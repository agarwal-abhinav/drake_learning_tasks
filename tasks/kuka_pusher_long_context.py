from typing import List
from omegaconf import DictConfig
import numpy as np
import time
import os
from enum import Enum
import matplotlib.pyplot as plt
import logging

from .base_task import BaseTask
from utils.iiwa_planner import IiwaPlanner
from utils.meshcat_utils import ConfigureAndStartMeshcat

# import relevant controllers 
from controllers.base_controller import BaseController
from controllers.ee_debug_controller import DebugEndEffectorController
from controllers.ee_gamepad_planar_controller import GamePadEndEffectorPlanarController

from pydrake.all import (
    DiagramBuilder,
    Frame,
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
    
    return parser_pre_finalize_function

class KukaPlanarPusherLongContextBlock(BaseTask):
    compatible_algorithms = {
        DebugEndEffectorController, 
        GamePadEndEffectorPlanarController,
    }

    def __init__(self, root_cfg: DictConfig) -> None:
        ## Parse config
        super().__init__(root_cfg)

        self.dt = self.cfg.dt 

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

        # Hardware Station (TODO: camera should be a part of this, find a way to do that automatically) 
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

        self.diagram = builder.Build()
        self.simulator = Simulator(self.diagram)
        self.simulator.set_target_realtime_rate(1.0)

        # collect reference names for data collection 
        self.pusher_frame = self.plant.GetFrameByName("pusher_end", self.plant.GetModelInstanceByName("pusher"))
        self.iiwa_frame = self.plant.GetFrameByName("iiwa_link_0", self.plant.GetModelInstanceByName("iiwa"))
        self.slider_frame = self.plant.GetFrameByName("box_center", self.plant.GetModelInstanceByName("object_red"))

        self.pusher_body = self.plant.GetBodyByName("pusher", self.plant.GetModelInstanceByName("pusher"))
        self.slider_body = self.plant.GetBodyByName("box_link", self.plant.GetModelInstanceByName("object_red"))
        self.pusher_collision_geom_id = self.plant.GetCollisionGeometriesForBody(self.pusher_body)[0]
        self.slider_collision_geom_id = self.plant.GetCollisionGeometriesForBody(self.slider_body)[0]

        self.keypoint_frame_names = ["object_red_keypoint_plus_x", "object_red_keypoint_minus_x", \
                                     "object_red_keypoint_plus_y", "object_red_keypoint_minus_y", \
                                    "object_red_keypoint_plus_x_y", "object_red_keypoint_minus_x_y", \
                                    "object_red_keypoint_plus_x_minus_y", "object_red_keypoint_minus_x_plus_y"]
        self.keypoint_frame_refs: List[Frame] = []
        for name in self.keypoint_frame_names: 
            self.keypoint_frame_refs.append(self.plant.GetFrameByName(name, \
                                                                      self.plant.GetModelInstanceByName("object_red")))

        if self.cfg.export_diagram:
            self.export_diagram("kuka_pusher_diagram.pdf")
    
    def reset_robot(self):
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

        # Diff-IK uses the world frame as the reference frame.
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
            random_theta = np.random.uniform(-initial_location_deltas[0], initial_location_deltas[0])

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

        # station_context = station.GetMyMutableContextFromRoot(context)
        # color_image = self.station.GetOutputPort("camera0.rgb_image").Eval(station_context).data.copy()
        # self.ax1 = plt.subplot(1, 1, 1)
        # self.color_plot = self.ax1.imshow(color_image)
        # plt.ion()

    def teleop(self):
        from pydrake.all import RigidTransform, Quaternion

        self.controller: BaseController
        self.controller.reset(meshcat=self.meshcat)

        plant = self.plant
        station = self.station
        simulator = self.simulator
        diff_ik = self.diff_ik
        scene_graph = self.scene_graph

        trajectory = {
            "time": [],
            "ee_pos": [],
            "ee_quat_wxyz": [],
            "pusher_vel_translational": [],
            "pusher_vel_rotational": [],
            "time_wall": [], 
            "block_pos": [], 
            "block_quat_wxyz": [], 
            "block_vel_translational": [], 
            "block_vel_rotational": [], 
            "contact_sdf": []
        }
        for frame_name in self.keypoint_frame_names: 
            trajectory[f"{frame_name}_pos"] = []
            trajectory[f"{frame_name}_quat_wxyz"] = []
            trajectory[f"{frame_name}_vel_translational"] = []
            trajectory[f"{frame_name}_vel_rotational"] = []
        # trajectory.update({f"cam_rgb_{k}": [] for k in self.cameras.keys()})

        # terminate = np.array([False])
        terminate_teleop = False
        while not terminate_teleop:
            context = simulator.get_mutable_context()
            station_context = station.GetMyMutableContextFromRoot(context)
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

            if self.mode == Mode.DATA_COLLECTION:
                pose = self.pusher_frame.CalcPose(plant_context, self.iiwa_frame)
                pos = np.array([pose.translation()])
                quat_wxyz = np.array([pose.rotation().ToQuaternion().wxyz()])
                # get the slider pose, velocity, and keypoint poses and velocities
                slider_pose = self.slider_frame.CalcPose(plant_context, self.iiwa_frame)
                slider_velocity = self.slider_frame.CalcRelativeSpatialVelocity(plant_context, self.iiwa_frame, self.iiwa_frame, self.iiwa_frame)

                pusher_vel = self.pusher_frame.CalcRelativeSpatialVelocity(plant_context, self.iiwa_frame, self.iiwa_frame, self.iiwa_frame)

                pusher_vel_translational = np.array([pusher_vel.translational()])
                pusher_vel_rotational = np.array([pusher_vel.rotational()])

                block_pos = np.array([slider_pose.translation()])
                block_quat_wxyz = np.array([slider_pose.rotation().ToQuaternion().wxyz()])
                block_vel = np.array([slider_velocity.translational()])
                block_ang_vel = np.array([slider_velocity.rotational()])

                keypoints_pos = []
                keypoints_quat_wxyz = []
                keypoints_vel = []
                keypoints_ang_vel = []
                for key_point_frame in self.keypoint_frame_refs: 
                    this_keypoint_pose = key_point_frame.CalcPose(plant_context, self.iiwa_frame)
                    this_keypoint_velocity = key_point_frame.CalcRelativeSpatialVelocity(plant_context, self.iiwa_frame, self.iiwa_frame, self.iiwa_frame)

                    keypoints_pos.append(this_keypoint_pose.translation())
                    keypoints_quat_wxyz.append(this_keypoint_pose.rotation().ToQuaternion().wxyz())
                    keypoints_vel.append(this_keypoint_velocity.translational())
                    keypoints_ang_vel.append(this_keypoint_velocity.rotational())

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
                trajectory["ee_pos"].append(pos)
                trajectory["ee_quat_wxyz"].append(quat_wxyz)
                trajectory["block_pos"].append(block_pos)
                trajectory["block_quat_wxyz"].append(block_quat_wxyz)
                trajectory["pusher_vel_translational"].append(pusher_vel_translational)
                trajectory["pusher_vel_rotational"].append(pusher_vel_rotational)
                trajectory["block_vel_translational"].append(block_vel)
                trajectory["block_vel_rotational"].append(block_ang_vel)
                for i, frame_name in enumerate(self.keypoint_frame_names):
                    trajectory[f"{frame_name}_pos"].append(keypoints_pos[i])
                    trajectory[f"{frame_name}_quat_wxyz"].append(keypoints_quat_wxyz[i])
                    trajectory[f"{frame_name}_vel_translational"].append(keypoints_vel[i])
                    trajectory[f"{frame_name}_vel_rotational"].append(keypoints_ang_vel[i])
                trajectory["contact_sdf"].append(this_sdf)
                # TODO: figure out camera interface
                # for serial, cam in self.cameras.items():
                #     frame = cam.wait_for_frames()
                #     rgb = np.array(frame.get_color_frame().get_data())
                #     trajectory[f"cam_rgb_{serial}"].append(rgb)
                #     if self.cfg.camera.display: 
                #         cv2.imshow(f"cam_rgb_{serial}", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

            enable, terminate, terminate_and_save, mode_switch = self.controller.get_info()
            # if self.cfg.camera.display: 
            #     cv2.waitKey(1)

            # color_image = self.station.GetOutputPort("camera0.rgb_image").Eval(station_context).data.copy()
            # self.color_plot.set_data(color_image)
            # plt.pause(0.01)

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

        print("\n")
        return trajectory
    
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
        save_path = f'data/{self.cfg.data_collection_run_folder_name}/{self.cfg.data_collection_type}'
        
        i = 0
        while os.path.exists(f'{save_path}/{i}'): 
            i += 1

        os.makedirs(f'{save_path}/{i}')

        for key in trajectory.keys(): 
            this_save_path = f'{save_path}/{i}/{key}.npy'
            np.save(this_save_path, np.array(trajectory[key]))

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

        while True:
            self.reset_robot()
            trajectory = self.teleop()
