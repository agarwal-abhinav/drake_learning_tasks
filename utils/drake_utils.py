from pydrake.all import (
    RigidTransform, 
    RollPitchYaw, 
    RenderEngineVtkParams, 
    LightParameter, 
    LeafSystem, 
    Value, 
    Image, 
    PixelType
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