import zarr 
import matplotlib.pyplot as plt 
import numpy as np 

import cv2 

import pathlib, sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

if __name__ == "__main__":
    zarr_path = "data/kuka_pusher_long_context_different_returns/start_bin_0_via_mirror.zarr"

    root = zarr.open_group(zarr_path, mode='r')
    data_group = root['data']
    meta_group = root['meta']

    overhead_images = data_group["overhead_camera"]
    wrist_images = data_group["wrist_camera"]
    proprioceptions = data_group["state"]
    actions = data_group["action"]

    breakpoint()

    for i in range(0, overhead_images.shape[0]): 
        cv2.imshow('video', cv2.cvtColor(np.array(overhead_images[i]).astype(np.uint8), cv2.COLOR_RGB2BGR))

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    
    breakpoint()