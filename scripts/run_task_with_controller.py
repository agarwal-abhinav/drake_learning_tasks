import pathlib, sys 

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import hydra 
from hydra.utils import get_class
from omegaconf import DictConfig, OmegaConf

from tasks.base_task import BaseTask
from controllers.base_controller import BaseController

@hydra.main(
    version_base=None, 
    config_path="../configurations", 
    config_name="run_task_with_controller.yaml"
)
def main(cfg: DictConfig) -> None: 
    task_class = get_class(cfg.task._target_)
    task: BaseTask = task_class(root_cfg=cfg)

    controller_class = get_class(cfg.controller._target_)
    controller: BaseController = controller_class(root_cfg=cfg)

    task.controller = controller

    for todo in cfg.run_methods: 
        task.exec_method(todo)

if __name__ == "__main__":
    main()