from pydrake.all import (
    RigidTransform, 
    RollPitchYaw, 
    RenderEngineVtkParams, 
    LightParameter, 
    LeafSystem, 
    Value, 
    Image, 
    PixelType, 
    HPolyhedron, 
    VPolytope, 
    InverseKinematics, 
    MultibodyPlant, 
    Context, 
    Solve
)
import matplotlib.pyplot as plt

from manipulation.station import Scenario

import numpy as np 

def xyz_rpy_deg(xyz, rpy_deg):
    """Shorthand for defining a pose."""
    rpy_deg = np.asarray(rpy_deg)
    return RigidTransform(RollPitchYaw(rpy_deg * np.pi / 180), xyz)

def change_camera_to_point_lighting(scenario: Scenario, main_camera_name: str = "camera0"): 
    for camera_config in scenario.cameras.values(): 
        if camera_config.name == main_camera_name:
            camera_config.renderer_name = "RenderEngineVtk"
            this_params = RenderEngineVtkParams()
            this_params.lights = [LightParameter(type="point")]
            camera_config.renderer_class = this_params
        else: 
            camera_config.renderer_name = "RenderEngineVtk"

    return scenario

def create_square_v_polytope(side_length: float) -> VPolytope:
    "returns a vpolytope in the frame of the center of the square"
    L = side_length / 2.0

    V = np.array([
        [-L, -L], 
        [L, -L], 
        [L, L], 
        [-L, L]
    ]).T

    return VPolytope(V)

def convert_rigid_transform_to_x_y_theta(transform: RigidTransform): 
    R = transform.rotation().matrix()
    theta = np.arctan2(R[1, 0], R[0, 0])
    x = transform.translation()[0]
    y = transform.translation()[1]

    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    return x, y, theta, rotation_matrix, np.array([[x], [y]])

class CameraSystem(LeafSystem):
    def __init__(self): 
        super().__init__()

        self._rgb_in = self.DeclareAbstractInputPort(
            name="camera_in", 
            model_value=Value(Image[PixelType.kRgba8U]()),
        ) 

        self.DeclarePeriodicPublishEvent(period_sec=0.1, 
                                         offset_sec=0, 
                                         publish=self.Publish)
        
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.im = self.ax.imshow(np.zeros((480, 640, 3)))
        
    def Publish(self, context): 
        rgb_image_object = self._rgb_in.Eval(context)

        rgb_image = rgb_image_object.data

        self.im.set_data(rgb_image)
        plt.draw()
        plt.pause(0.01)

def iiwa_ik_function(pose: RigidTransform, 
                     plant: MultibodyPlant, 
                     plant_context: Context, 
                     q0): 
    ik = InverseKinematics(plant, plant_context)
    ik.AddPositionConstraint(
        plant.GetFrameByName("iiwa_link_7"),
        [0, 0, 0], 
        plant.world_frame(), 
        pose.translation(), 
        pose.translation()
    )
    ik.AddOrientationConstraint(
        plant.GetFrameByName("iiwa_link_7"),
        RigidTransform().rotation(),
        plant.world_frame(),
        pose.rotation(),
        0.01
    )
    prog = ik.get_mutable_prog()
    q = ik.q() 
    prog.AddQuadraticErrorCost(np.identity(len(q)), q0, q)
    prog.SetInitialGuess(q, q0) 

    result = Solve(ik.prog())

    assert result.is_success(), "IK did not succeed"
    q_sol = result.GetSolution(q)
    return q_sol
