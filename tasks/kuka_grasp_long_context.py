from .base_task import BaseTask

from controllers.base_controller import BaseController

from utils.drake_utils import change_camera_to_point_lighting

from omegaconf import DictConfig, OmegaConf
import os

from manipulation.station import LoadScenario, MakeHardwareStation

from pydrake.all import (
    DiagramBuilder, 
    MultibodyPlant, 
    RobotDiagram, 
    StartMeshcat
)

class KukaGraspLongContextBlock(BaseTask): 
    def __init__(self, root_cfg: DictConfig, 
                 X_WO, 
                 meshcat_initialized = None, 
                 ) -> None:
        super().__init__(root_cfg)

        self.dt = self.cfg.dt 
        self.save_data_dt = self.cfg.save_data_dt 

        builder = DiagramBuilder()

        scenario = LoadScenario(filename=os.path.abspath(self.cfg.scenario_path), scenario_name="overall")
        traj_opt_scenario = LoadScenario(
            filename=os.path.abspath(self.cfg.scenario_path), scenario_name="for_traj_opt"
        )

        # make changes for camera lighting 
        scenario = change_camera_to_point_lighting(scenario, main_camera_name="camera0")

        if meshcat_initialized is None: 
            self.meshcat = StartMeshcat()
        else: 
            self.meshcat = meshcat_initialized

        # make hardware station 
        self.station: RobotDiagram = builder.AddSystem(
                MakeHardwareStation(
                scenario=scenario, 
                meshcat=self.meshcat, 
                package_xmls=self.package_list, 
            )
        )

        self.plant: MultibodyPlant = self.station.GetSubsystemByName("plant")
        self.plant.SetDefaultFreeBodyPose(
            self.plant.GetBodyByName("base_link"), X_WO["initial"]
        )

        temp_station_context = self.station.CreateDefaultContext()
        temp_plant_context = self.plant.GetMyContextFromRoot(temp_station_context)
        X_WO["initial"] = self.plant.EvalBodyPoseInWorld(
            temp_plant_context, 
            self.plant.GetBodyByName("base_link")
        )

        

