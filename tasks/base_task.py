from abc import ABC, abstractmethod
from omegaconf import DictConfig
from typing import Type, Callable, Dict
import pydot
import os

from controllers.base_controller import BaseController

class BaseTask(ABC): 

    compatible_controllers: Dict[Type[BaseController], Callable] = {}

    def __init__(self, 
                 root_cfg: DictConfig
                 ) -> None:
        
        self.root_cfg = root_cfg
        self.cfg = root_cfg.task
        self.debug = root_cfg.debug 

        self._controller = None 
        self.simulator = None # when controller is set, simulator must be initialized

        # every task is going to have at list this package 
        basic_package_path = os.path.abspath("models/package.xml")
        self.package_list = [basic_package_path]

    @property
    def controller(self) -> BaseController: 
        return self._controller 
    
    @controller.setter
    def controller(self, value: BaseController) -> None: 

        if self._controller is not None: 
            raise ValueError("Controller has already been set for this task.")
        
        compat = getattr(self, "compatible_controllers", dict())
        if compat and not any(isinstance(value, c) for c in compat.keys()): 
            allowed = [c.__name__ for c in compat.keys()]
            raise ValueError(
                f"Controller of type {type(value).__name__} is not compatible with this task. "
                f"Allowed controller types: {allowed}"
            )
        
        self._controller = value 
        self.compatible_controllers[type(value)](self)

        assert self.simulator is not None, "Simulator must be initialized in the task before setting the controller."
    
    def remove_controller(self) -> None: 
        self._controller = None 

    def clear_attrs(self, *, exclude=(), close_resources=True, skip_private=True):
        """
        Set instance attributes to None to help release references.

        Args:
            exclude: iterable of attribute names to keep.
            close_resources: if True, call .close() or .shutdown() on values if present.
            skip_private: if True, skip attributes starting with '_' (including dunder-ish).
        """
        exclude = set(exclude or ())

        def _try_close(val):
            try:
                if hasattr(val, "close") and callable(val.close):
                    val.close()
                elif hasattr(val, "shutdown") and callable(val.shutdown):
                    val.shutdown()
            except Exception:
                pass

        # normal instance dict attributes
        d = getattr(self, "__dict__", {})
        for name, val in list(d.items()):
            if name in exclude: 
                continue
            if skip_private and name.startswith("_"):
                continue
            if close_resources and val is not None:
                _try_close(val)
            try:
                setattr(self, name, None)
            except Exception:
                try:
                    delattr(self, name)
                except Exception:
                    pass

        # handle __slots__ if present
        slots = getattr(self.__class__, "__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        for name in slots:
            if not name or name in exclude:
                continue
            if skip_private and name.startswith("_"):
                continue
            if hasattr(self, name):
                try:
                    val = getattr(self, name)
                    if close_resources and val is not None:
                        _try_close(val)
                    setattr(self, name, None)
                except Exception:
                    try:
                        delattr(self, name)
                    except Exception:
                        pass

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
        
    def export_diagram(self, filename: str, diagram=None) -> None:
        if diagram is None: 
            diagram = self.diagram
        pydot.graph_from_dot_data(diagram.GetGraphvizString())[0].write_pdf(  # type: ignore
            filename
        )
        print(f"Saved diagram to: {filename}")

