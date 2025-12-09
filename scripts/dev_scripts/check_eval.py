import matplotlib.pyplot as plt 
import numpy as np 

import cv2 

import pathlib, sys
import pickle 

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

if __name__ == "__main__": 
    eval_dir = "eval/long_context_planar_pushing/two_modes/unet_cross_attention/constant_obs_steps/48_obs/latest/start_0"

    eval_path = pathlib.Path(eval_dir)
    if not eval_path.is_absolute():
        eval_path = PROJECT_ROOT / eval_path

    if not eval_path.exists():
        raise FileNotFoundError(f"Directory not found: {eval_path}")

    dir_list = [str(p) for p in sorted(eval_path.iterdir()) if p.is_dir()]

    middle_overlap_area = 0 
    num_middle_overlap = 0 

    final_overlap_area = 0 
    num_final_overlap = 0 

    total_sucees = 0 
    total_mild_success = 0 
    total_a_box = 0 
    total_mild_a_box = 0 
    
    for dir in dir_list: 
        with open(pathlib.Path(dir) / "metadata.pkl", "rb") as f: 
            eval_data = pickle.load(f)

        overhead_images = eval_data["overhead_images"]