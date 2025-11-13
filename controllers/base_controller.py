from abc import ABC, abstractmethod
from dataclasses import dataclass

from typing import Dict

import numpy as np 

@dataclass
class BatchedPose: 
    pos: np.ndarray
    rot: np.ndarray

class BaseController(ABC): 
    def __init__(self, root_cfg=None):
        self.root_cfg = root_cfg
        self.cfg = root_cfg.controller
        self.debug = root_cfg.debug 

    @abstractmethod
    def reset(self, *args, **kwargs) -> None: 
        raise NotImplementedError

    @abstractmethod
    def update(self, obs_dict: Dict, *args, **kwargs) -> None: 
        raise NotImplementedError