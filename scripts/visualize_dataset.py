import pathlib, sys 

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import hydra 
from omegaconf import DictConfig, OmegaConf 

import numpy as np 
import cv2
import os 

@hydra.main(
    version_base=None, 
    config_path="../configurations", 
    config_name="visualize_dataset.yaml"
)
def main(cfg: DictConfig) -> None: 
    data_dir = os.path.join(cfg.data_root_dir, cfg.dataset)

    traj_dir_list = [
        name for name in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, name))
    ]

    for traj_dir in traj_dir_list: 
        loaded_images = []
        for camera_name in cfg.cameras: 
            loaded_images_this = np.load(os.path.join(data_dir, traj_dir, f"cam_rgb_{camera_name}.npy"))
            loaded_images.append(loaded_images_this)

            for frame in loaded_images_this: 
                cv2.imshow('video', cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGBA2BGR))
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    break   
    cv2.destroyAllWindows()
    breakpoint()