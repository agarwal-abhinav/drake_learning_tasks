from .base_controller import BaseController
import numpy as np

class DebugEndEffectorController(BaseController): 
    def __init__(self, root_cfg): 
        super().__init__(root_cfg)
        self.translation = self.cfg.translation
        self.orientation = self.cfg.orientation 
        self.grasp = None 

    def reset(self, meshcat): 
        ... 

    def update(self, obs_dict, *args, **kwargs):
        return self.translation, self.orientation, self.grasp
    
    def get_info(self): 
        return np.array([True]), np.array([False]), np.array([False]), np.array([False])