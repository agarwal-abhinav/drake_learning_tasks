from abc import ABC, abstractmethod
from omegaconf import DictConfig
from typing import Type
import pydot

from controllers.base_controller import BaseController

class BaseTask(ABC): 

    compatible_controllers: set[Type[BaseController]] = set()

    def __init__(self, 
                 root_cfg: DictConfig
                 ) -> None:
        
        self.root_cfg = root_cfg
        self.cfg = root_cfg.task
        self.debug = root_cfg.debug 

        self._controller = None 

    @property
    def controller(self) -> BaseController: 
        return self._controller 
    
    @controller.setter
    def controller(self, value: BaseController) -> None: 

        if self._controller is not None: 
            raise ValueError("Controller has already been set for this task.")
        
        compat = getattr(self, "compatible_controllers", set())
        if compat and not any(isinstance(value, c) for c in compat): 
            allowed = [c.__name__ for c in compat]
            raise ValueError(
                f"Controller of type {type(value).__name__} is not compatible with this task. "
                f"Allowed controller types: {allowed}"
            )
        
        self._controller = value 

    @abstractmethod
    def reset_robot(self) -> None: 
        pass 

    def exec_method(self, method: str) -> None: 
        # run a specific task specified by the method 

        if hasattr(self, method) and callable(getattr(self, method)): 
            print("Executing method: ", f"{method}")
            getattr(self, method)()
        else: 
            raise ValueError(f"Method {method} is not defined in {self.__class__.__name__}.")
        
    def export_diagram(self, filename: str):
        pydot.graph_from_dot_data(self.diagram.GetGraphvizString())[0].write_pdf(  # type: ignore
            filename
        )
        print(f"Saved diagram to: {filename}")

