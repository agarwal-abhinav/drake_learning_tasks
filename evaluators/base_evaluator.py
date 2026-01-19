from abc import ABC, abstractmethod
from omegaconf import DictConfig
from hydra.utils import get_class
from typing import Type, Callable, Dict 

from tasks.base_task import BaseTask

class BaseEvaluator(ABC):
    def __init__(self, 
                 root_cfg: DictConfig): 
        self.root_cfg = root_cfg
        self.cfg = root_cfg.evaluator
        
        self.task_class = get_class(self.root_cfg.task._target_)
        self.controller_class = get_class(self.root_cfg.controller._target_)