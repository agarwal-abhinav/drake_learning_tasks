from typing import List
from omegaconf import DictConfig
import numpy as np
import time
import os
from enum import Enum
import matplotlib.pyplot as plt
import logging
import copy
import shutil
import yaml
from hydra.core.hydra_config import HydraConfig
import random
from tqdm import tqdm 
from collections import deque
import sys
import pickle 

from .base_task import BaseTask
from utils.iiwa_planner import IiwaPlanner
from utils.meshcat_utils import ConfigureAndStartMeshcat

# import relevant controllers 
from controllers.base_controller import BaseController
from controllers.ee_debug_controller import DebugEndEffectorController
from controllers.ee_gamepad_planar_controller import GamePadEndEffectorPlanarController
from controllers.ee_diffusion_planar_controller import PlanarDiffusionPolicyDrakeController


from pydrake.all import (
    AbstractValue, 
    Context, 
    DiagramBuilder,
    Frame,
    LeafSystem, 
    LogVectorOutput,
    ModelDirectives,
    MultibodyPlant,
    Parser,
    PortSwitch,
    ProcessModelDirectives,
    QueryObject,
    Quaternion,
    RigidTransform,
    RobotDiagram,
    Simulator,
    WeldJoint, 
    InputPortIndex, 
    HPolyhedron, 
    VPolytope, 
    RotationMatrix, 
    RandomGenerator
)

from utils.drake_utils import xyz_rpy_deg, change_camera_to_point_lighting, convert_rigid_transform_to_x_y_theta
from utils.trajectory_utils import is_stationary_and_close_2D, clip_start_end_idle
from utils.drake_utils import create_square_v_polytope, iiwa_ik_function


from manipulation.station import MakeHardwareStation
from manipulation.station import (
    LoadScenario,
    ConfigureParser,
)
from manipulation.scenarios import AddIiwaDifferentialIK, AddMultibodyTriad, AddIiwa

import time 

class Mode(Enum):
    REGULAR = 'regular'
    DATA_COLLECTION = 'data collection'

def pre_finalize_function(dict_of_bins): 
    def parser_pre_finalize_function(parser: Parser): 
        plant: MultibodyPlant = parser.plant()
        for bin_name, bin_properties in dict_of_bins.items(): 
            new_model_instance = parser.AddModelsFromUrl(url=bin_properties[0])[0]
            plant.RenameModelInstance(model_instance=new_model_instance, name=f"start_goal_{bin_name}")
            weld_joint = WeldJoint(
                name=f"{bin_name}_weld_joint",
                frame_on_parent_F=plant.world_frame(),
                frame_on_child_M=plant.GetFrameByName("box_center", new_model_instance),
                X_FM=xyz_rpy_deg(bin_properties[1], [0, 0, 0])
            )
            plant.AddJoint(weld_joint)

            if len(bin_properties) > 2:
                new_model_instance_2 = parser.AddModelsFromUrl(url=bin_properties[2])[0]
                plant.RenameModelInstance(model_instance=new_model_instance_2, name=f"pusher_goal_for_{bin_name}")
                weld_joint_2 = WeldJoint(
                    name=f"pusher_{bin_name}_weld_joint",
                    frame_on_parent_F=plant.world_frame(),
                    frame_on_child_M=plant.GetFrameByName("cylinder", new_model_instance_2),
                    X_FM=xyz_rpy_deg(bin_properties[3], [0, 0, 0])
                )
                plant.AddJoint(weld_joint_2)
    
    return parser_pre_finalize_function

class DiffusionToDiffIKBridge(LeafSystem): 
    def __init__(self, 
                 fixed_z_position, 
                 fixed_orientation, 
                 translation_offset=np.array([0.0, 0.0, 0.0])): 
        super().__init__()

        self.DeclareVectorInputPort(
            "diffusion_action", 
            2
        )

        self.DeclareAbstractOutputPort(
            "X_WG", 
            lambda: AbstractValue.Make(RigidTransform()), 
            self.CalcEEPose
        )

        self.z_position = fixed_z_position
        self.fixed_orientation = fixed_orientation
        self.translation_offset = translation_offset

    def CalcEEPose(self, context: Context, output):
        diffusion_action = self.get_input_port(0).Eval(context)
        target_pos = np.array([diffusion_action[0], diffusion_action[1], self.z_position]) + self.translation_offset

        target_pose = RigidTransform(Quaternion(self.fixed_orientation), target_pos)

        output.set_value(target_pose)

class KukaPlanarPusherLongContextBlock(BaseTask):
    compatible_controllers = {
        DebugEndEffectorController: lambda instance: instance._initialize_basic_non_leafsystem_controller(), 
        GamePadEndEffectorPlanarController: lambda instance: instance._initialize_basic_non_leafsystem_controller(),
        PlanarDiffusionPolicyDrakeController: lambda instance: instance._initialize_diffusion_planar_controller(), 
        # PlanarTrajectorySourceDrakeController: lambda instance: instance._initialize_trajectory_source_planar_controller()
    }

    def __init__(self, root_cfg: DictConfig, 
                 meshcat_initialized = None) -> None:
        ## Parse config
        super().__init__(root_cfg)

        self.dt = self.cfg.dt 
        self.save_data_dt = self.cfg.save_data_dt

        self.mode = Mode.REGULAR
        self.is_hardware = self.cfg.is_hardware
        assert self.is_hardware == False, "Hardware not supported for this task yet."

        self.reset_pose = RigidTransform(
            Quaternion(self.cfg.reset_orientation_wxyz),
            np.array(self.cfg.reset_translation),
        )

        ## Build diagram
        builder = DiagramBuilder()

        # Load scenario
        scenario = LoadScenario(data=self.cfg.scenario_data, scenario_name="overall")
        scenario = change_camera_to_point_lighting(scenario, main_camera_name="overhead_camera")

        # Meshcat
        if meshcat_initialized is None: 
            self.meshcat = ConfigureAndStartMeshcat(scenario)
        else:
            self.meshcat = meshcat_initialized
        # self.meshcat.AddButton("Stop Simulation", "Escape")

        package_path = os.path.abspath("models/package.xml")
        package_list = [package_path]
        self.station = MakeHardwareStation(
                scenario=scenario, 
                meshcat=self.meshcat, 
                package_xmls=package_list, 
                parser_prefinalize_callback=pre_finalize_function(self.cfg.bins),
            )
        self.station: RobotDiagram = builder.AddSystem(self.station)

        # Iiwa Planner
        self.nominal_joint_positions = np.array(self.cfg.nominal_joint_positions)
        if self.is_hardware:
            toppra_vel_lim = np.array(self.cfg.hw_toppra_velocity_limit)
            toppra_acc_lim = np.array(self.cfg.hw_toppra_acceleration_limit)
        else:
            toppra_vel_lim = np.array(self.cfg.sim_toppra_velocity_limit)
            toppra_acc_lim = np.array(self.cfg.sim_toppra_acceleration_limit)
        self.iiwa_planner: IiwaPlanner = builder.AddSystem(
            IiwaPlanner(
                robot_plant=self._load_robot_only(package_list),
                desired_pose = self.reset_pose,
                frame_name="iiwa_link_7",
                nominal_joint_positions=self.nominal_joint_positions,
                toppra_vel_lim=toppra_vel_lim,
                toppra_acc_lim=toppra_acc_lim,
                initial_delay=self.cfg.initial_delay,
                wait_diff_ik_delay=self.cfg.wait_diff_ik_delay,
            )
        )
        
        # Differential IK
        self.plant = self.station.plant()
        self.scene_graph = self.station.scene_graph()
        self.iiwa_controller = self.station.GetSubsystemByName("iiwa_controller_plant_pointer_system").get()
        self.diff_ik = AddIiwaDifferentialIK(
            builder, 
            self.iiwa_controller, 
            self.iiwa_controller.GetFrameByName("iiwa_link_7")
        )
        self.diff_ik.get_mutable_parameters().set_nominal_joint_position(
            self.nominal_joint_positions
        )

        # Port switch
        self.port_switch = builder.AddSystem(PortSwitch(7))

        ## Connect systems
        # Connect iiwa_planner
        builder.Connect(
            self.station.GetOutputPort("iiwa.position_measured"),
            self.iiwa_planner.GetInputPort("iiwa_position_measured"),
        )

        # Connect switch inputs
        builder.Connect(
            self.iiwa_planner.GetOutputPort("control_mode"),
            self.port_switch.get_port_selector_input_port(),
        )
        builder.Connect(
            self.diff_ik.get_output_port(),
            self.port_switch.DeclareInputPort("diff_ik_command")
        )
        builder.Connect(
            self.iiwa_planner.GetOutputPort("iiwa_position_command"),
            self.port_switch.DeclareInputPort("reset_command"),
        )

        # Connect switch output
        builder.Connect(
            self.port_switch.get_output_port(),
            self.station.GetInputPort("iiwa.position")
        )
        builder.Connect(
            self.iiwa_planner.GetOutputPort("reset_diff_ik"),
            self.diff_ik.GetInputPort("use_robot_state"),
        )
        
        # Diff IK connections
        builder.Connect(
            self.station.GetOutputPort("iiwa.state_estimated"), 
            self.diff_ik.GetInputPort("robot_state")
        )

        # Connect loggers
        self.state_estimated_logger = LogVectorOutput(
            self.station.GetOutputPort("iiwa.state_estimated"),
            builder,
        )

        self.torque_commanded_logger = LogVectorOutput(
            self.station.GetOutputPort("iiwa.torque_commanded"),
            builder,
        )
        self.switch_logger = LogVectorOutput(
            self.port_switch.get_output_port(),
            builder,
        )
        self.iiwa_planner_logger = LogVectorOutput(
            self.iiwa_planner.GetOutputPort("iiwa_position_command"),
            builder,
        )

        # if in debug mode, visualize camera feed 
        if self.cfg.display_camera_feed: 
            from utils.drake_utils import CameraSystem
            self.camera_systems = []
            for camera_name in scenario.cameras.keys(): 
                this_system = builder.AddSystem(CameraSystem())
                builder.Connect(
                    self.station.GetOutputPort(f"{camera_name}.rgb_image"), 
                    this_system.GetInputPort("camera_in")
                )
                self.camera_systems.append(this_system)

                if self.debug: 
                    camera_instance = self.plant.GetModelInstanceByName(camera_name)
                    AddMultibodyTriad(
                        self.plant.GetFrameByName("base", camera_instance),
                        self.scene_graph, 
                        length=0.1, 
                        radius=0.005
                    )
        self.scenario = scenario

        for camera_name in self.scenario.cameras.keys(): 
            builder.ExportOutput(self.station.GetOutputPort(f"{camera_name}.rgb_image"), f"{camera_name}")

        builder.ExportInput(self.diff_ik.GetInputPort("X_AE_desired"), "X_AE_desired")
        builder.ExportOutput(self.station.GetOutputPort("body_poses"), "body_poses")
        # builder.ExportOutput(self.iiwa_planner.GetOutputPort("control_mode"), "planner_control_mode")

        self.diagram = builder.Build()

        # collect reference names for data collection 
        self.pusher_frame = self.plant.GetFrameByName("pusher_end", self.plant.GetModelInstanceByName("pusher"))
        self.iiwa_frame = self.plant.GetFrameByName("iiwa_link_0", self.plant.GetModelInstanceByName("iiwa"))
        self.slider_frame = self.plant.GetFrameByName(self.cfg.slider_frame_name, self.plant.GetModelInstanceByName(self.cfg.slider_name))

        self.pusher_body = self.plant.GetBodyByName("pusher", self.plant.GetModelInstanceByName("pusher"))
        self.slider_body = self.plant.GetBodyByName(self.cfg.slider_body_name, self.plant.GetModelInstanceByName(self.cfg.slider_name))
        self.pusher_collision_geom_id = self.plant.GetCollisionGeometriesForBody(self.pusher_body)[0]
        self.slider_collision_geom_id = self.plant.GetCollisionGeometriesForBody(self.slider_body)[0]

        self.keypoint_frame_names = self.cfg.keypoint_frame_names

        self.keypoint_frame_refs: List[Frame] = []
        for name in self.keypoint_frame_names: 
            self.keypoint_frame_refs.append(self.plant.GetFrameByName(name, \
                                                                      self.plant.GetModelInstanceByName(self.cfg.slider_name)))

        self.save_data_every = int(self.save_data_dt / self.dt)
    
    # controller on-boarding methods 
    # for controllers which aren't leaf systems, simply initialize a simulator 
    # for controllers that are, create another diagram for controller side, link it with the main diagram, then simulator 
    def _initialize_basic_non_leafsystem_controller(self): 
        self.simulator = Simulator(self.diagram)
        self.simulator.set_target_realtime_rate(1.0)

        if self.cfg.export_diagram:
            self.export_diagram("kuka_pusher_diagram.pdf")

    def _initialize_diffusion_planar_controller(self): 
        from pydrake.all import ZeroOrderHold

        self.controller: PlanarDiffusionPolicyDrakeController 

        self.iiwa_base_translation_in_world = self.iiwa_frame.CalcPose(
            self.plant.CreateDefaultContext(), self.plant.world_frame()
        ).translation().flatten()

        # build a controller side diagram 
        control_side_diagram_builder = DiagramBuilder()
        control_side_diagram_builder.AddSystem(self.controller)
        zo_hold = control_side_diagram_builder.AddSystem(
            ZeroOrderHold(
                self.controller.cfg.controller_dt,  
                vector_size=2
            )
        )
        diff_to_diff_ik = control_side_diagram_builder.AddSystem(
            DiffusionToDiffIKBridge(
                fixed_z_position=self.reset_pose.translation()[2], 
                fixed_orientation=self.reset_pose.rotation().matrix(), 
                translation_offset=self.iiwa_base_translation_in_world
            )
        )
        control_side_diagram_builder.Connect(
            self.controller.GetOutputPort("planar_command_out"),
            zo_hold.get_input_port(0)
        )
        control_side_diagram_builder.Connect(
            zo_hold.get_output_port(0), 
            diff_to_diff_ik.GetInputPort("diffusion_action")
        )

        # add current diagram to controller side diagram 
        control_side_diagram_builder.AddSystem(self.diagram)
        control_side_diagram_builder.Connect(
            diff_to_diff_ik.GetOutputPort("X_WG"),
            self.diagram.GetInputPort("X_AE_desired")
        )
        control_side_diagram_builder.Connect(
            self.diagram.GetOutputPort("body_poses"),
            self.controller.GetInputPort("body_poses")
        )
        for camera_name in self.scenario.cameras.keys(): 
            control_side_diagram_builder.Connect(
                self.diagram.GetOutputPort(f"{camera_name}"),
                self.controller.GetInputPort(f"{camera_name}_in")
            )
        # control_side_diagram_builder.Connect(
        #     self.diagram.GetOutputPort("planner_control_mode"), 
        #     self.controller.GetInputPort("use_controller")
        # )

        control_side_diagram = control_side_diagram_builder.Build()

        self.simulator = Simulator(control_side_diagram)
        self.simulator.set_target_realtime_rate(1.0)

        self.initial_location_in_iiwa0 = self.reset_pose.translation().flatten()[:2] 

        self.controller.set_post_connection_values(
            initial_planar_command=self.initial_location_in_iiwa0,
            ee_body_index=self.pusher_body.index(), 
            translation_offset=self.iiwa_base_translation_in_world[:2]
        )
        self.controller.reset(
            meshcat=self.meshcat
        )

        if self.debug: 
            self.export_diagram("kuka_pusher_diffusion_controller_diagram.pdf", control_side_diagram)
            self.diff_to_diff_ik = diff_to_diff_ik
            self.zo_hold = zo_hold

    def reset_robot(self, seed: int = 42):
        np.random.seed(seed)
        random.seed(seed)
        self.seed = seed

        plant = self.plant
        iiwa_planner = self.iiwa_planner
        simulator = self.simulator
        diff_ik = self.diff_ik
        self.simulator.Initialize()

        if isinstance(self.controller, LeafSystem): 
            # set the controller to not be active during reset 
            self.controller.GetInputPort("use_controller").FixValue(
                self.controller.GetMyMutableContextFromRoot(simulator.get_mutable_context()), 
                InputPortIndex(0)
            )

        # Disable contact forces in pretty mode
        self.meshcat.SetProperty("/drake/contact_forces", "visible", False)

        context = simulator.get_mutable_context()
        plant_context = plant.GetMyMutableContextFromRoot(context)
        iiwa_planner_context = iiwa_planner.GetMyMutableContextFromRoot(context)
        diff_ik_context = diff_ik.GetMyMutableContextFromRoot(context)

        # Fix appropriate input ports
        q0 = plant.GetPositions(plant_context)
        self.iiwa_planner.GetInputPort("last_reset_time").FixValue(iiwa_planner_context, [context.get_time()])
        self.iiwa_planner.GetInputPort("reset_position").FixValue(iiwa_planner_context, q0[:7])

        # # Diff-IK uses the world frame as the reference frame.
        if not isinstance(self.controller, LeafSystem): 
            X_W_iiwa0 = self.iiwa_frame.CalcPose(plant_context, self.plant.world_frame())
            X_iiwa0_reset = self.reset_pose
            X_W_reset = X_W_iiwa0 @ X_iiwa0_reset
            self.diff_ik.GetInputPort("X_AE_desired").FixValue(diff_ik_context, X_W_reset)

        # Move to start
        self.simulator.AdvanceTo(context.get_time() + self.cfg.initial_delay + 0.5)
        traj_length = self.iiwa_planner.get_trajectory_length()
        self.simulator.AdvanceTo(context.get_time() + traj_length)

        # Print a random initial pose for the block
        initial_location_type = self.cfg.initial_location_type 
        initial_location_deltas = self.cfg.initial_location_deltas

        if initial_location_type is not None: 
            initial_location_translation = self.cfg.bins[initial_location_type][1]
            random_x = np.random.uniform(initial_location_translation[0]-initial_location_deltas[0], initial_location_translation[0]+initial_location_deltas[0])
            random_y = np.random.uniform(initial_location_translation[1]-initial_location_deltas[1], initial_location_translation[1]+initial_location_deltas[1])
            random_theta = np.random.uniform(-initial_location_deltas[2], initial_location_deltas[2])

        current_slider_world_pose = self.slider_frame.CalcPose(plant_context, self.plant.world_frame())
        new_slider_world_pose = xyz_rpy_deg([random_x, random_y, current_slider_world_pose.translation()[2]], [0, 0, random_theta])
        plant.SetFreeBodyPose(plant_context, self.slider_body, new_slider_world_pose)
        self.simulator.AdvanceTo(context.get_time() + 0.5)

        print("\nRandom initial slider pose:")
        print(f"x: {100*random_x:.2f}cm")
        print(f"y: {100*random_y:.2f}cm")
        print(f"theta: {round(random_theta)} degrees\n")

    def teleop(self):
        self.controller: BaseController
        self.controller.reset(meshcat=self.meshcat)

        plant = self.plant
        station = self.station
        simulator = self.simulator
        diff_ik = self.diff_ik
        scene_graph = self.scene_graph

        trajectory = {
            "time": [],
            "pusher_pos": [],
            "pusher_quat_wxyz": [],
            "time_wall": [], 
            "slider_pos": [], 
            "slider_quat_wxyz": [], 
            "contact_sdf": []
        }
        for frame_name in self.keypoint_frame_names: 
            trajectory[f"pusher_{frame_name}_pos"] = []
            trajectory[f"pusher_{frame_name}_quat_wxyz"] = []
        
        for camera_name in self.scenario.cameras.keys(): 
            trajectory[f"cam_rgb_{camera_name}"] = []

        running_index = 0

        terminate_teleop = False
        while not terminate_teleop:
            context = simulator.get_mutable_context()
            plant_context = plant.GetMyMutableContextFromRoot(context)
            diff_ik_context = diff_ik.GetMyMutableContextFromRoot(context)
            scene_graph_context = scene_graph.GetMyMutableContextFromRoot(context)
            
            # do the teleop work on the ee and iiwa
            pose = self.pusher_frame.CalcPose(plant_context, plant.world_frame())
            pos = np.array([pose.translation()])
            quat_wxyz = np.array([pose.rotation().ToQuaternion().wxyz()])
            obs_dict = dict(ee_pos=pos, ee_quat=quat_wxyz)
            target_pos, target_quat_wxyz, *args = self.controller.update(obs_dict)
            target_pose = RigidTransform(Quaternion(target_quat_wxyz[0]), target_pos[0])

            diff_ik.GetInputPort("X_AE_desired").FixValue(diff_ik_context, target_pose)
            
            # if this is the first iteration
            if len(trajectory['time']) == 0 and self.mode == Mode.DATA_COLLECTION:
                data_collection_start_index = running_index
            
            if (self.mode == Mode.DATA_COLLECTION and (running_index - data_collection_start_index)%self.save_data_every == 0):
                # get the pusher pose, velocity 
                pusher_pose = self.pusher_frame.CalcPose(plant_context, self.iiwa_frame)
                pusher_pos = np.array([pusher_pose.translation()])
                pusher_quat_wxyz = np.array([pusher_pose.rotation().ToQuaternion().wxyz()])

                # get the slider pose, velocity, and keypoint poses and velocities
                slider_pose = self.slider_frame.CalcPose(plant_context, self.iiwa_frame)
                slider_pos = np.array([slider_pose.translation()])
                slider_quat_wxyz = np.array([slider_pose.rotation().ToQuaternion().wxyz()])

                keypoints_pos = []
                keypoints_quat_wxyz = []
                for key_point_frame in self.keypoint_frame_refs: 
                    this_keypoint_pose = key_point_frame.CalcPose(plant_context, self.iiwa_frame)
                    keypoints_pos.append(this_keypoint_pose.translation())
                    keypoints_quat_wxyz.append(this_keypoint_pose.rotation().ToQuaternion().wxyz())

                # calculate the sdf between pusher and slider 
                sg_query_obj: QueryObject = scene_graph.get_query_output_port().Eval(scene_graph_context)
                collision_pairs = sg_query_obj.ComputeSignedDistancePairwiseClosestPoints()
                this_sdf = None
                for collision_pair in collision_pairs: 
                    if (collision_pair.id_A == self.pusher_collision_geom_id and collision_pair.id_B == self.slider_collision_geom_id) or \
                        (collision_pair.id_A == self.slider_collision_geom_id and collision_pair.id_B == self.pusher_collision_geom_id):
                        this_sdf = collision_pair.distance
                assert this_sdf is not None

                # this slider pose can be saved
                if len(trajectory['time']) == 0:
                    start_time = simulator.get_context().get_time()
                    start_time_macro = time.time()
                t = simulator.get_context().get_time() - start_time
                t_wall = time.time() - start_time_macro
                trajectory["time"].append(t)
                trajectory["time_wall"].append(t_wall)
                trajectory["pusher_pos"].append(pusher_pos)
                trajectory["pusher_quat_wxyz"].append(pusher_quat_wxyz)
                trajectory["slider_pos"].append(slider_pos)
                trajectory["slider_quat_wxyz"].append(slider_quat_wxyz)
                for i, frame_name in enumerate(self.keypoint_frame_names):
                    trajectory[f"pusher_{frame_name}_pos"].append(keypoints_pos[i])
                    trajectory[f"pusher_{frame_name}_quat_wxyz"].append(keypoints_quat_wxyz[i])
                trajectory["contact_sdf"].append(this_sdf)

                for camera_name in self.scenario.cameras.keys(): 
                    camera_rgb = copy.deepcopy(self.diagram.GetOutputPort(f"{camera_name}").Eval(context).data)
                    trajectory[f"cam_rgb_{camera_name}"].append(camera_rgb)

            enable, terminate, terminate_and_save, mode_switch = self.controller.get_info()

            if terminate:
                if self.mode == Mode.DATA_COLLECTION:
                    print("Terminating data collection without saving trajectory...")
                else:
                    print("Terminating teleop...")
                print("Reset mode to regular.")
                self.mode = Mode.REGULAR

            if terminate_and_save: 
                if self.mode == Mode.DATA_COLLECTION: 
                    print("Terminating and saving trajectory...")
                    self.save_trajectory(trajectory)
                else: 
                    print("Terminating teleop...")
                print("Reset mode to regular")
                self.mode = Mode.REGULAR

            if mode_switch:
                self.mode = Mode.DATA_COLLECTION if self.mode == Mode.REGULAR else Mode.REGULAR
                print(f"Switched to {self.mode.value} mode.")

            if terminate.any() or terminate_and_save.any(): 
                terminate_teleop = True

            simulator.AdvanceTo(simulator.get_context().get_time() + self.dt)
            running_index += 1

        print("\n")
        return trajectory
    
    def diffusion_rollout(self, max_time: float = 20.0, save_path = None, save_html: bool = False): 
        self.controller: PlanarDiffusionPolicyDrakeController
        if save_html == True or self.controller.cfg.save_trajectory == True: 
            assert save_path is not None, "Must provide save path if saving html or trajectory."

        self.controller.reset(
            self.meshcat
        )

        if isinstance(self.controller, LeafSystem): 
            self.controller.GetInputPort("use_controller").FixValue(
                self.controller.GetMyMutableContextFromRoot(self.simulator.get_mutable_context()), 
                InputPortIndex(1)
            )

        if save_html: 
            self.meshcat.StartRecording()

        if self.controller.cfg.save_trajectory: 
            trajectory = {
                "time": [],
                "pusher_pos": [],
                "slider_pos": [],
                "slider_quat_wxyz": [],
            }
        
        simulator = self.simulator
        plant = self.plant

        context = simulator.get_mutable_context()
        plant_context = plant.GetMyMutableContextFromRoot(context)

        if self.cfg.slider_is_square: 
            goal_box_v_poly = create_square_v_polytope(
                side_length=self.cfg.goal_side_length
            )
            slider_v_poly = create_square_v_polytope(
                side_length=self.cfg.slider_side_length
            )
            slider_base_volume = self.cfg.slider_side_length ** 2

            goal_hpoly_list = []
            for bin_name, bin_properties in self.cfg.bins.items():
                if bin_name not in self.controller.cfg.modes_to_eval: 
                    continue
                goal_frame = self.plant.GetFrameByName("box_center", self.plant.GetModelInstanceByName(f"start_goal_{bin_name}"))
                goal_frame_in_iiwa0 = goal_frame.CalcPose(
                    plant_context, self.iiwa_frame
                )
                _, _, _, rot_mat, transl = convert_rigid_transform_to_x_y_theta(goal_frame_in_iiwa0)
                goal_box_v_poly_rotated_translated = rot_mat @ goal_box_v_poly.vertices() + transl
                goal_box_h_poly = HPolyhedron(VPolytope(goal_box_v_poly_rotated_translated))
                goal_hpoly_list.append((bin_name, goal_box_h_poly, goal_box_v_poly_rotated_translated))

            center_goal_frame = self.plant.GetFrameByName("box_center", self.plant.GetModelInstanceByName("shadow_box_center"))
            center_goal_frame_in_iiwa0 = center_goal_frame.CalcPose(
                plant_context, self.iiwa_frame
            )
            _, _, _, rot_mat, transl = convert_rigid_transform_to_x_y_theta(center_goal_frame_in_iiwa0)
            center_goal_box_v_poly_rotated_translated = rot_mat @ goal_box_v_poly.vertices() + transl
            center_goal_h_poly = HPolyhedron(VPolytope(center_goal_box_v_poly_rotated_translated))
        else: 
            raise NotImplementedError("Only square sliders supported for eval currently.")
        
        goal_cylinder_pos_list = []
        for bin_name, bin_properties in self.cfg.bins.items(): 
            if bin_name not in self.controller.cfg.modes_to_eval: 
                continue
            goal_cylinder_frame = self.plant.GetFrameByName("cylinder", self.plant.GetModelInstanceByName(f"pusher_goal_for_{bin_name}"))
            goal_cylinder_frame_in_iiwa0 = goal_cylinder_frame.CalcPose(
                plant_context, self.iiwa_frame  
            )
            goal_cylinder_pos = goal_cylinder_frame_in_iiwa0.translation()[:2]
            goal_cylinder_pos_list.append((bin_name, goal_cylinder_pos))

        center_cylinder_frame = self.plant.GetFrameByName("cylinder", self.plant.GetModelInstanceByName(f"cylinder_goal"))
        center_cylinder_frame_in_iiwa0 = center_cylinder_frame.CalcPose(
            plant_context, self.iiwa_frame  
        )
        center_cylinder_pos = center_cylinder_frame_in_iiwa0.translation()[:2]
    
        reached_mid = False 
        mild_return_to_a_box = False # returned to a box > 95% at rest, but not pusher 
        return_to_a_box = False # returned to a box > 80% and pusher slider both at rest 
        mild_success = False # returned to correct box > 95% at rest, but not pusher 
        success = False # returned to correct box > 80%, and pusher slider both at rest 

        mid_overlap_area = None 
        final_overlap_area = None 

        num_steps = int(max_time / self.dt)

        ee_pos_log = deque([], maxlen=10)
        slider_pos_log = deque([], maxlen=5)

        index_of_mid = None 
        index_of_mild_return = None 
        index_of_strong_return = None 

        for step in range(num_steps): 
            simulator.AdvanceTo(simulator.get_context().get_time() + self.dt)

            ee_pos = self.pusher_frame.CalcPose(plant_context, self.iiwa_frame).translation()[:2]
            ee_pos_log.append(ee_pos)

            slider_pose = self.slider_frame.CalcPose(plant_context, self.iiwa_frame)
            _, _, _, rot_mat, transl = convert_rigid_transform_to_x_y_theta(slider_pose)
            slider_v_poly_rotated_translated = rot_mat @ slider_v_poly.vertices() + transl
            slider_h_poly = HPolyhedron(VPolytope(slider_v_poly_rotated_translated))

            if self.controller.cfg.save_trajectory:
                trajectory["time"].append(simulator.get_context().get_time())
                trajectory["pusher_pos"].append(ee_pos)
                trajectory["slider_pos"].append(transl)
                trajectory["slider_quat_wxyz"].append(slider_pose.rotation().ToQuaternion().wxyz())

            slider_pos_log.append(transl)

            # check if reached center 
            if not reached_mid: 
                center_intersect = center_goal_h_poly.Intersection(slider_h_poly)

                ee_poss = np.stack(ee_pos_log, axis=0)
                pairwise_distances = np.linalg.norm(ee_poss[:, None, :] - ee_poss[None, :, :], axis=-1)

                distance_to_center_cylinder = np.linalg.norm(ee_pos - center_cylinder_pos)

                if not center_intersect.IsEmpty() and np.all(pairwise_distances) < 1e-4 and distance_to_center_cylinder < self.cfg.eval_center_cylinder_threshold: 
                    intersection_volume = center_intersect.CalcVolumeViaSampling(RandomGenerator(0), 0.01, 10000)
                    this_mid_overlap_area = intersection_volume.volume / slider_base_volume
                    if this_mid_overlap_area >= self.cfg.slider_mid_overlap_threshold:
                        reached_mid = True 
                        print("Reached center position of slider.")
                        mid_overlap_area = this_mid_overlap_area
                        index_of_mid = step
            else: 
                ee_poss = np.stack(ee_pos_log, axis=0)
                pairwise_distances = np.linalg.norm(ee_poss[:, None, :] - ee_poss[None, :, :], axis=-1)

                slider_poss = np.stack(slider_pos_log, axis=0)
                slider_pairwise_distances = np.linalg.norm(slider_poss[:, None, :] - slider_poss[None, :, :], axis=-1)

                if np.all(slider_pairwise_distances < 1e-4) and not mild_return_to_a_box: 
                    # slider at rest 
                    # check if it intersects with any of the goal boxes 
                    if self.debug: 
                        print(f"Slider at rest, checking for mild return to a box...{step}")
                        import time 
                        start = time.time()
                    
                    for bin_name, goal_box_h_poly, _ in goal_hpoly_list: 
                        goal_intersect = goal_box_h_poly.Intersection(slider_h_poly)
                        if not goal_intersect.IsEmpty(): 
                            overlap_volume = goal_intersect.CalcVolumeViaSampling(RandomGenerator(0), 0.01, 10000)
                            if (overlap_volume.volume / slider_base_volume) >= self.cfg.mild_overlap_volume_threshold: 
                                mild_return_to_a_box = True 
                                print("Mildly returned slider to a box.")
                                index_of_mild_return = step
                                if bin_name == self.cfg.initial_location_type: 
                                    mild_success = True 
                                    print("Mildly successfully returned slider to goal box.")
                            break
                    
                    if self.debug:
                        end = time.time()
                        print(f"Time taken for mild return to a box check: {end - start:.4f} seconds")

                if np.all(pairwise_distances < 1e-4) and mild_return_to_a_box:
                    # pusher at rest 

                    # check if slider is inside a bin 
                    if self.debug: 
                        print(f"RUNNING THIS CHECK {step}")
                        import time 
                        start = time.time()
                    
                    goal_box_index = -1 
                    for bin_name, goal_box_h_poly, _ in goal_hpoly_list: 
                        goal_intersect = goal_box_h_poly.Intersection(slider_h_poly)
                        if not goal_intersect.IsEmpty(): 
                            overlap_volume = goal_intersect.CalcVolumeViaSampling(RandomGenerator(0), 0.01, 10000)
                            if (overlap_volume.volume / slider_base_volume) >= self.cfg.strong_overlap_volume_threshold: 
                                goal_box_index = bin_name 
                                break
                    if self.debug: 
                        end = time.time()
                        print(f"Time taken for final box check: {end - start:.4f} seconds")
                    
                    # check if pusher is at rest at return location 
                    distance_to_goal_cylinder = np.inf
                    goal_cylinder_index = -1
                    if self.debug: 
                        import time 
                        start = time.time()
                    for bin_name, goal_cylinder_pos in goal_cylinder_pos_list:
                        this_distance = np.linalg.norm(ee_pos - goal_cylinder_pos)
                        if this_distance < distance_to_goal_cylinder:
                            distance_to_goal_cylinder = this_distance
                            if distance_to_goal_cylinder < self.cfg.eval_goal_cylinder_threshold:
                                goal_cylinder_index = bin_name
                                break
                    if self.debug: 
                        end = time.time()
                        print(f"Time taken for final cylinder check: {end - start:.4f} seconds")

                    if goal_box_index != -1 and goal_cylinder_index != -1:
                        return_to_a_box = True 
                        final_overlap_area = overlap_volume.volume / slider_base_volume 
                        print("Returned slider to a box.")
                        index_of_strong_return = step
                    
                        if goal_box_index == self.cfg.initial_location_type: 
                            success = True 
                            print("Successfully returned slider to goal box.")
                        
                        break               

        print("Middle overlap area:", mid_overlap_area)
        print("Final overlap area:", final_overlap_area)
        print("Success:", success)
        print("Mild Success:", mild_success)
        print("Returned to a box:", return_to_a_box)
        print("Mild return to a box:", mild_return_to_a_box)

        if save_html: 
            self.meshcat.StopRecording()
            self.meshcat.PublishRecording()
            html = self.meshcat.StaticHtml() 
            with open(os.path.join(save_path, "rollout.html"), "w") as f: 
                f.write(html)
            self.meshcat.DeleteRecording()
            import time 
            time.sleep(1)

        if self.controller.cfg.save_trajectory:
            for key in trajectory.keys(): 
                this_save_path = f'{save_path}/{key}.npy'
                assert len(trajectory[key]) > 0, "No data collected for this trajectory key!"
                np.save(this_save_path, np.array(trajectory[key]))
            
            dict_of_metadata = {
                "mid_overlap_area": mid_overlap_area,
                "final_overlap_area": final_overlap_area,
                "success": success,
                "mild_success": mild_success,
                "return_to_a_box": return_to_a_box,
                "mild_return_to_a_box": mild_return_to_a_box,
                "index_of_mid": index_of_mid,
                "index_of_mild_return": index_of_mild_return,
                "index_of_strong_return": index_of_strong_return,
            }

            with open(f'{save_path}/metadata.pkl', 'wb') as f:
                pickle.dump(dict_of_metadata, f)

        return (mid_overlap_area, final_overlap_area), (success, mild_success), (return_to_a_box, mild_return_to_a_box)

    def plot_state_log(self, log):
        t = log.sample_times()
        data = log.data()
        for dim in range(7):
            plt.plot(t, data[dim], label=f'Joint {dim+1}')
        plt.xlabel('Time')
        plt.ylabel('Position')
        plt.grid(True)
        plt.legend()
        plt.savefig('position_plot.pdf')
        plt.show()
        plt.clf()
        
        for dim in range(7,14):
            plt.plot(t, data[dim], label=f'Joint {dim+1-7}')
        plt.xlabel('Time')
        plt.ylabel('Velocity')
        plt.savefig('velocity_plot.pdf')
        plt.legend()
        plt.grid(True)
        plt.show()
        plt.clf()
        
    def plot_log(self, log, name):
        t = log.sample_times()
        data = log.data()
        for dim in range(data.shape[0]):
            plt.plot(t, data[dim], label=f'Joint {dim+1}')
        plt.xlabel('Time')
        plt.grid(True)
        plt.legend()
        plt.savefig(name)
        plt.show()
        plt.clf()

    def _load_robot_only(self, package_xmls: list): 
        scenario = LoadScenario(data=self.cfg.scenario_data, scenario_name="robot-only")

        # Create the multibody plant
        plant = MultibodyPlant(time_step=self.dt)
        plant.set_name("iiwa_planner_plant")

        parser = Parser(plant)
        for p in package_xmls:
            parser.package_map().AddPackageXml(p)
        ConfigureParser(parser)

        # Add model directives
        _ = ProcessModelDirectives(
            directives=ModelDirectives(directives=scenario.directives),
            parser=parser,
        )
        plant.Finalize()
        print(plant.num_positions())
        return plant

    def save_trajectory(self, trajectory): 
        if self.cfg.initial_location_type is not None: 
            subdir = f"start_bin_{self.cfg.initial_location_type}"
        else: 
            subdir = "general"

        save_path = f'data/{self.cfg.data_collection_run_folder_name}/{subdir}'

        i = self.seed
        os.makedirs(f'{save_path}/{i}')

        for key in trajectory.keys(): 
            this_save_path = f'{save_path}/{i}/{key}.npy'
            assert len(trajectory[key]) > 0, "No data collected for this trajectory key!"
            np.save(this_save_path, np.array(trajectory[key]))

        shutil.copytree(HydraConfig.get().runtime.output_dir, f'{save_path}/{i}/hydra_logs')

    def run_eval(self): 
        logging.getLogger(
            "drake"
        ).setLevel(logging.INFO)
        logging.getLogger(
            "utils/iiwa_planner"
        ).setLevel(logging.DEBUG)

        from utils.logging_utils import IterTee

        assert self.controller is not None, "Evaluation controller algorithm not set for the task."

        if self.controller.cfg.eval_min_seed is not None: 
            seeds = list(range(self.controller.cfg.eval_min_seed, self.controller.cfg.eval_max_seed))
        else: 
            seeds = [42]

        output_dir = HydraConfig.get().runtime.output_dir 
        global_log_path = os.path.join(output_dir, "eval_log.txt")
        
        # Open global log. If it exists, open for append and write a start marker;
        # otherwise create it and write a creation marker.
        if os.path.exists(global_log_path):
            global_log = open(global_log_path, "a")
            global_log.write(f"\n\n=== Appending new eval run started: {time.asctime()} ===\n")
        else:
            global_log = open(global_log_path, "w")
            global_log.write(f"=== Eval log created: {time.asctime()} ===\n")
        global_log.flush()

        tee = IterTee(sys.stdout, global_log)
        sys.stdout = tee 

        m = 0
        total_success = 0 
        total_mild_success = 0 
        total_return_to_box = 0 
        total_mild_return_to_box = 0 

        total_mid_area = 0 
        total_final_area = 0 
        num_mid_area = 0 
        num_final_area = 0
        while m < len(seeds): 
            # create the logging for this particular loop 
            os.mkdir(os.path.join(output_dir, f"eval_seed_{seeds[m]}"))
            iter_log_path = os.path.join(output_dir, f"eval_seed_{seeds[m]}", "eval_log.txt")
            iter_log = open(iter_log_path, "w")
            tee.set_iter_file(iter_log)

            seed = seeds[m]
            if m < self.controller.cfg.save_html_first: 
                save_html = True 
            else: 
                save_html = False
            m += 1 
            print(f"Starting new eval with seed {seed}...\n")
            self.reset_robot(seed)
            areas, successes, correct_returns = self.diffusion_rollout(self.controller.cfg.eval_max_time, 
                                                                       save_path=os.path.join(output_dir, f"eval_seed_{seed}"), 
                                                                       save_html=save_html)

            if areas[0] is not None: 
                total_mid_area += areas[0] 
                num_mid_area += 1 
            
            if areas[1] is not None: 
                total_final_area += areas[1]
                num_final_area += 1

            if successes[0] == True: 
                total_success += 1 
            if successes[1] == True: 
                total_mild_success += 1
            
            if correct_returns[0] == True: 
                total_return_to_box += 1
            if correct_returns[1] == True: 
                total_mild_return_to_box += 1

            iter_log.close()
            tee.set_iter_file(None)

        print("\n================ Evaluation Summary ================\n")
        print(f"Evaluation for: {self.cfg.initial_location_type} and seed: {self.controller.cfg.eval_min_seed} to {self.controller.cfg.eval_max_seed-1}")
        print(f"Average mid overlap area: {total_mid_area/num_mid_area if num_mid_area > 0 else 0}")
        print(f"Average final overlap area: {total_final_area/num_final_area if num_final_area > 0 else 0}")
        print(f"Total success count: {total_success} out of {len(seeds)}")
        print(f"Total mild success count: {total_mild_success} out of {len(seeds)}")
        print(f"Total return to a box: {total_return_to_box} out of {len(seeds)}")
        print(f"Total mild return to a box: {total_mild_return_to_box} out of {len(seeds)}")
        print("\n====================================================\n")

        global_log.close()

    def run_teleop(self):
        logging.getLogger(
            "drake"
        ).setLevel(logging.INFO)
        logging.getLogger(
            "utils/iiwa_planner"
        ).setLevel(logging.DEBUG)

        assert self.controller is not None, "Teleop controller algorithm not set for the task."

        if type(self.controller) == GamePadEndEffectorPlanarController: 
            assert(self.reset_pose.translation()[2] == self.controller.cfg.z_values[0])

        if self.cfg.teleop_min_seed is not None: 
            seeds = list(range(self.cfg.teleop_min_seed, self.cfg.teleop_max_seed))
        else: 
            seeds = [42]

        m = 0
        while True:
            if m < len(seeds):
                seed = seeds[m]
                m += 1
            else:
                seed = random.randint(0, 10000)
            print(f"Starting new teleop with seed {seed}...\n")
            self.reset_robot(seed)
            trajectory = self.teleop()

    def check_saved_trajectory_images(self): 
        import cv2 

        if self.cfg.initial_location_type is not None: 
            subdir = f"start_bin_{self.cfg.initial_location_type}"
        else: 
            subdir = "general"

        data_path = f'data/{self.cfg.data_collection_run_folder_name}/{subdir}'

        traj_dir_list = [
            name for name in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, name))
        ]    

        for traj_dir in tqdm(traj_dir_list):
            loaded_images = []
            for camera_name in self.scenario.cameras.keys(): 
                loaded_images_this = np.load(os.path.join(data_path, traj_dir, f"cam_rgb_{camera_name}.npy"))
                loaded_images.append(loaded_images_this)

                for frame in loaded_images_this: 
                    cv2.imshow('video', cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGBA2BGR))
                    if cv2.waitKey(30) & 0xFF == ord('q'):
                        break   
        cv2.destroyAllWindows()
        breakpoint()
    
    def _create_fake_ik_plant(self): 
        scenario = LoadScenario(data=self.cfg.scenario_data, scenario_name="robot-only")
        plant = MultibodyPlant(time_step=self.dt)

        parser = Parser(plant)
        ConfigureParser(parser)
        parser.package_map().AddPackageXml("models/package.xml")
        _ = ProcessModelDirectives(
            directives=ModelDirectives(directives=scenario.directives),
            parser=parser,
        )
        plant.Finalize()

        return plant

    def render_mirror_with_forced_publish(self): 
        import time 
        if self.cfg.initial_location_type is not None: 
            subdir = f"start_bin_{self.cfg.initial_location_type}"
        else: 
            subdir = "general"

        data_path = f'data/{self.cfg.data_collection_run_folder_name}/{subdir}'

        traj_dir_list = [
            name for name in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, name))
        ] 

        diagram_context = self.diagram.CreateDefaultContext()
        plant_context = self.plant.GetMyMutableContextFromRoot(diagram_context)
        
        ik_plant = self._create_fake_ik_plant()
        ik_plant_context = ik_plant.CreateDefaultContext()

        iiwa_model_instance = self.plant.GetModelInstanceByName("iiwa")

        X_W_iiwa0 = self.iiwa_frame.CalcPose(plant_context, self.plant.world_frame())

        if self.cfg.initial_location_type == 0: 
            this_location_type = 4 
        elif self.cfg.initial_location_type == 4: 
            this_location_type = 0
        elif self.cfg.initial_location_type == 1: 
            this_location_type = 3
        elif self.cfg.initial_location_type == 3: 
            this_location_type = 1
        else:
            raise NotImplementedError("Only swapping between 0 and 4, or 1 and 3 supported for mirroring currently.")
        
        for traj_dir in tqdm(traj_dir_list): 
            loaded_pos = np.load(os.path.join(data_path, traj_dir, "slider_pos.npy"))
            loaded_quat = np.load(os.path.join(data_path, traj_dir, "slider_quat_wxyz.npy"))

            loaded_ee_pos = np.load(os.path.join(data_path, traj_dir, "pusher_pos.npy"))

            with open(os.path.join(data_path, traj_dir, "hydra_logs/.hydra/config.yaml"), 'r') as f: 
                run_config = yaml.safe_load(f)

            trajectory = {
                "pusher_pos": [],
                "slider_pos": [],
                "slider_quat_wxyz": [],
            }
            for camera_name in self.scenario.cameras.keys(): 
                trajectory[f"cam_rgb_{camera_name}"] = []

            save_path = f'data/{self.cfg.data_collection_run_folder_name}/start_bin_{this_location_type}_via_mirror/{traj_dir}'
            
            os.makedirs(os.path.join(save_path, "hydra_logs", ".hydra"))
            with open(os.path.join(save_path, "hydra_logs/.hydra/config.yaml"), 'w') as f: 
                yaml.dump(run_config, f)
            
            current_iiwa_joints = np.array(
                self.cfg.nominal_joint_positions
            )

            # deal with the slider 
            for m in range(loaded_pos.shape[0]):
                slider_pose_in_iiwa0 = RigidTransform(
                    Quaternion(loaded_quat[m].T),
                    loaded_pos[m].flatten()
                )
                slider_pose = X_W_iiwa0 @ slider_pose_in_iiwa0

                # calculate the mirrored pose 
                slider_pose_translation = slider_pose.translation().flatten()
                slider_pose_rotation = slider_pose.rotation().matrix()

                slider_pose_rotation[0, 1] = -slider_pose_rotation[0, 1]
                slider_pose_rotation[1, 0] = -slider_pose_rotation[1, 0]

                slider_pose_mirrored = RigidTransform(
                    RotationMatrix(slider_pose_rotation), 
                    np.array([
                        slider_pose_translation[0],
                        -slider_pose_translation[1],
                        slider_pose_translation[2]
                    ])
                )
                self.plant.SetFreeBodyPose(
                    plant_context,
                    self.slider_body,
                    slider_pose_mirrored
                )
                
                ee_pos_in_iiwa0 = loaded_ee_pos[m].flatten()
                ee_pos_in_world = ee_pos_in_iiwa0 + X_W_iiwa0.translation().flatten()
                ee_pos_in_world[-1] = self.cfg.reset_translation[2]
                ee_pos_in_world[1] = -ee_pos_in_world[1]  # mirror y coordinate
                ee_pose_in_world = RigidTransform(
                    Quaternion(self.cfg.reset_orientation_wxyz),
                    ee_pos_in_world
                )

                iiwa_joints = iiwa_ik_function(
                    ee_pose_in_world, 
                    ik_plant, 
                    ik_plant_context, 
                    current_iiwa_joints
                )

                self.plant.SetPositions(
                    plant_context, 
                    iiwa_model_instance, 
                    iiwa_joints.flatten()
                )

                self.diagram.ForcedPublish(diagram_context)
                current_iiwa_joints = iiwa_joints

                ee_pos = self.pusher_frame.CalcPose(plant_context, self.iiwa_frame).translation()
                slider_pose = self.slider_frame.CalcPose(plant_context, self.iiwa_frame)
                trajectory["pusher_pos"].append(np.array([ee_pos]))
                trajectory["slider_pos"].append(np.array([slider_pose.translation()]))
                trajectory["slider_quat_wxyz"].append(np.array([slider_pose.rotation().ToQuaternion().wxyz()]))

                for camera_name in self.scenario.cameras.keys():
                    camera_rgb = copy.deepcopy(self.diagram.GetOutputPort(f"{camera_name}").Eval(diagram_context).data)
                    trajectory[f"cam_rgb_{camera_name}"].append(camera_rgb)

            # save the mirrored trajectory
            os.makedirs(save_path, exist_ok=True)
            for key in trajectory.keys(): 
                this_save_path = f'{save_path}/{key}.npy'
                assert len(trajectory[key]) > 0, "No data collected for this trajectory key!"
                np.save(this_save_path, np.array(trajectory[key]))
                
    def _print_data_overlap_statistics(self): 

        if self.cfg.initial_location_type is not None: 
            subdir = f"start_bin_{self.cfg.initial_location_type}"
        else: 
            subdir = "general"

        data_path = f'data/{self.cfg.data_collection_run_folder_name}/{subdir}'

        traj_dir_list = [
            name for name in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, name))
        ]    

        if self.cfg.slider_is_square: 
            slider_v_poly = create_square_v_polytope(
                side_length=self.cfg.slider_side_length
            )
            goal_box_v_poly = create_square_v_polytope(
                side_length=self.cfg.goal_side_length
            )

            goal_frame = self.plant.GetFrameByName("box_center", self.plant.GetModelInstanceByName(f"start_goal_{self.cfg.initial_location_type}"))
            goal_frame_in_iiwa0 = goal_frame.CalcPose(
                self.plant.CreateDefaultContext(), self.iiwa_frame
            )
            R_goal = goal_frame_in_iiwa0.rotation().matrix()
            theta_goal = np.arctan2(R_goal[1,0], R_goal[0,0])
            x_goal = goal_frame_in_iiwa0.translation()[0]
            y_goal = goal_frame_in_iiwa0.translation()[1]

            rotation_matrix = np.array([
                [np.cos(theta_goal), -np.sin(theta_goal)],
                [np.sin(theta_goal), np.cos(theta_goal)]
            ])
            goal_box_v_poly_rotated_translated = rotation_matrix @ goal_box_v_poly.vertices() + np.array([[x_goal], [y_goal]])
            goal_h_poly = HPolyhedron(VPolytope(goal_box_v_poly_rotated_translated))

            slider_base_volume = self.cfg.slider_side_length **2

            center_goal_frame = self.plant.GetFrameByName("box_center", self.plant.GetModelInstanceByName("shadow_box_center"))
            center_goal_frame_in_iiwa0 = center_goal_frame.CalcPose(
                self.plant.CreateDefaultContext(), self.iiwa_frame
            )
            R_center_goal = center_goal_frame_in_iiwa0.rotation().matrix()
            theta_center_goal = np.arctan2(R_center_goal[1,0], R_center_goal[0,0])
            x_center_goal = center_goal_frame_in_iiwa0.translation()[0]
            y_center_goal = center_goal_frame_in_iiwa0.translation()[1]
            rotation_matrix_center = np.array([
                [np.cos(theta_center_goal), -np.sin(theta_center_goal)],
                [np.sin(theta_center_goal), np.cos(theta_center_goal)]
            ])
            center_goal_box_v_poly_rotated_translated = rotation_matrix_center @ goal_box_v_poly.vertices() + np.array([[x_center_goal], [y_center_goal]])
            center_goal_h_poly = HPolyhedron(VPolytope(center_goal_box_v_poly_rotated_translated))

        else: 
            raise NotImplementedError("Only square sliders supported for overlap statistics currently.")
        
        center_cylinder_frame = self.plant.GetFrameByName("cylinder", self.plant.GetModelInstanceByName("cylinder_goal"))
        center_cylinder_frame_in_iiwa0 = center_cylinder_frame.CalcPose(
            self.plant.CreateDefaultContext(), self.iiwa_frame
        )
        center_cylinder_pos = center_cylinder_frame_in_iiwa0.translation()[:2]

        average_overlap = 0.0
        average_overlap_at_center = 0.0

        for traj_dir in tqdm(traj_dir_list):
            loaded_pos = np.load(os.path.join(data_path, traj_dir, "slider_pos.npy"))
            loaded_quat = np.load(os.path.join(data_path, traj_dir, "slider_quat_wxyz.npy"))

            last_frame_pos = loaded_pos[-1]
            last_frame_quat = loaded_quat[-1]

            R = RotationMatrix(Quaternion(last_frame_quat.T))
            theta = np.arctan2(R.matrix()[1,0], R.matrix()[0,0])
            x = last_frame_pos[0, 0]
            y = last_frame_pos[0, 1]

            rotation_matrix = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]
            ])
            slider_box_v_poly_rotated_translated = rotation_matrix @ slider_v_poly.vertices() + np.array([[x], [y]])
            slider_h_poly = HPolyhedron(VPolytope(slider_box_v_poly_rotated_translated))

            intersection: HPolyhedron = goal_h_poly.Intersection(slider_h_poly)
            intersection_volume = intersection.CalcVolumeViaSampling(RandomGenerator(0), 0.01, 10000).volume

            average_overlap += intersection_volume / slider_base_volume

            # loaded_pos_pusher = np.load(os.path.join(data_path, traj_dir, "pusher_pos.npy"))[..., :2]
            # loaded_pos_pusher = np.squeeze(loaded_pos_pusher, axis=1)
            # first_idx, last_idx = clip_start_end_idle(
            #     loaded_pos_pusher, 
            #     1e-9, 
            #     keep_idle=0
            # )
            # loaded_pos_pusher = loaded_pos_pusher[first_idx:last_idx]
            # # breakpoint()

            # center_index = is_stationary_and_close_2D(
            #     loaded_pos_pusher, center_cylinder_pos
            # )
            # assert center_index is not -1, "Did not find stationary position near goal cylinder!"
            # center_slider_pos = loaded_pos[center_index]
            # center_slider_quat = loaded_quat[center_index]
            # R = RotationMatrix(Quaternion(center_slider_quat.T))
            # theta = np.arctan2(R.matrix()[1,0], R.matrix()[0,0])
            # x = center_slider_pos[0, 0]
            # y = center_slider_pos[0, 1]

            # rotation_matrix = np.array([
            #     [np.cos(theta), -np.sin(theta)],
            #     [np.sin(theta), np.cos(theta)]
            # ])
            # slider_center_v_poly_rotated_translated = rotation_matrix @ slider_v_poly.vertices() + np.array([[x], [y]])
            # slider_center_h_poly = HPolyhedron(VPolytope(slider_center_v_poly_rotated_translated))
            # intersection_at_center: HPolyhedron = center_goal_h_poly.Intersection(slider_center_h_poly)
            # intersection_volume_at_center = intersection_at_center.CalcVolumeViaSampling(RandomGenerator(0), 0.01, 10000).volume
            # average_overlap_at_center += intersection_volume_at_center / slider_base_volume

        average_overlap /= len(traj_dir_list)
        # average_overlap_at_center /= len(traj_dir_list)
        print(f"Average overlap with goal region over {len(traj_dir_list)} trajectories: {average_overlap*100:.2f}%")
        # print(f"Average overlap at center over {len(traj_dir_list)} trajectories: {average_overlap_at_center*100:.2f}%")