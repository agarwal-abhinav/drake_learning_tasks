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
)

from utils.drake_utils import xyz_rpy_deg, change_camera_to_point_lighting

from manipulation.station import MakeHardwareStation
from manipulation.station import (
    LoadScenario,
    ConfigureParser,
)
from manipulation.scenarios import AddIiwaDifferentialIK, AddMultibodyTriad

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
        PlanarDiffusionPolicyDrakeController: lambda instance: instance._initialize_diffusion_planar_controller()
    }

    def __init__(self, root_cfg: DictConfig) -> None:
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
        self.meshcat = ConfigureAndStartMeshcat(scenario)
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

        control_side_diagram = control_side_diagram_builder.Build()

        self.simulator = Simulator(control_side_diagram)
        self.simulator.set_target_realtime_rate(1.0)

        self.initial_location_in_iiwa0 = self.reset_pose.translation().flatten()[:2] - self.iiwa_base_translation_in_world[:2]
        self.controller.reset(
            initial_planar_command=self.initial_location_in_iiwa0,
            ee_body_index=self.pusher_body.index(),
            meshcat=self.meshcat, translation_offset=self.iiwa_base_translation_in_world[:2]
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

        # Print a random initial pose for the T
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

        # Move to start
        self.simulator.AdvanceTo(context.get_time() + self.cfg.initial_delay + 0.5)
        traj_length = self.iiwa_planner.get_trajectory_length()
        self.simulator.AdvanceTo(context.get_time() + traj_length)

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
    
    def diffusion_rollout(self, max_time: float = 20.0): 
        self.controller: PlanarDiffusionPolicyDrakeController
        self.controller.reset(
            initial_planar_command=self.initial_location_in_iiwa0,
            ee_body_index=self.pusher_body.index(), 
            meshcat=self.meshcat, 
            translation_offset=self.iiwa_base_translation_in_world[:2]
        )

        simulator = self.simulator
        plant = self.plant
        scene_graph = self.scene_graph

        context = simulator.get_mutable_context()
        plant_context = plant.GetMyMutableContextFromRoot(context)
        scene_graph_context = scene_graph.GetMyMutableContextFromRoot(context)

        if self.debug: 
            diff_to_diff_ik = self.diff_to_diff_ik
            diff_to_diff_ik_context = diff_to_diff_ik.GetMyMutableContextFromRoot(context)
            controller = self.controller
            controller_context = controller.GetMyMutableContextFromRoot(context)

            zo_hold = self.zo_hold
            zo_hold_context = zo_hold.GetMyMutableContextFromRoot(context)

        num_steps = int(max_time / self.dt)

        for step in range(num_steps): 
            simulator.AdvanceTo(simulator.get_context().get_time() + self.dt)

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

        assert self.controller is not None, "Evaluation controller algorithm not set for the task."

        if self.controller.cfg.eval_min_seed is not None: 
            seeds = list(range(self.controller.cfg.eval_min_seed, self.controller.cfg.eval_max_seed))
        else: 
            seeds = [42]
        
        m = 0
        while True: 
            if m < len(seeds): 
                seed = seeds[m]
                m += 1 
            else: 
                seed = random.randint(0, 10000)
            print(f"Starting new eval with seed {seed}...\n")
            self.reset_robot(seed)
            self.diffusion_rollout(self.controller.cfg.eval_max_time)

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
        from tqdm import tqdm 

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




