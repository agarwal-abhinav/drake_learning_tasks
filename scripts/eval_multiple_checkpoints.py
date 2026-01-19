import pathlib, sys 

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import hydra 
from hydra.utils import get_class
from omegaconf import DictConfig, OmegaConf

from evaluators.base_evaluator import BaseEvaluator

@hydra.main(
    version_base=None, 
    config_path="../configurations", 
    config_name="eval_multiple_checkpoints.yaml"
)
def main(cfg: DictConfig) -> None: 
    evaluator_class = get_class(cfg.evaluator._target_)
    evaluator: BaseEvaluator = evaluator_class(root_cfg=cfg)

    evaluator.run_eval()

if __name__ == "__main__":
    main()