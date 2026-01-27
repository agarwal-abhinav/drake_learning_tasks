import pathlib, sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import hydra 
from hydra.utils import get_class
from omegaconf import DictConfig, OmegaConf

import numpy as np
import random 
from typing import Dict, List, Tuple
import os 
from tqdm import tqdm 
import cv2
import copy 
import zarr
import yaml 

from utils.trajectory_utils import clip_start_end_idle

def calculate_traj_dir_list(data_paths: Dict[str, int], sort=True) -> List[str]: 
    traj_dir_list = []
    for data_path in data_paths.keys(): 
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data path {data_path} does not exist.")
        # collect directories and sort them numerically if possible
        dirs = [name for name in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, name))]
        try:
            dirs = sorted(dirs, key=lambda n: int(os.path.basename(n)))
        except ValueError:
            dirs = sorted(dirs)  # fallback to lexicographic if names aren't pure integers
        for i, name in enumerate(dirs):
            traj_dir_list.append(os.path.join(data_path, name))
            if i >= data_paths[data_path]:
                break

    if sort: 
        # traj_dir_list already ordered by numeric basename per-source; if you want global numeric sort:
        try:
            traj_dir_list = sorted(traj_dir_list, key=lambda p: int(os.path.basename(p)))
        except ValueError:
            traj_dir_list = sorted(traj_dir_list)

    return traj_dir_list

@hydra.main(
    version_base=None, 
    config_path="../configurations", 
    config_name="convert_data_to_zarr.yaml"
)
def main(cfg: DictConfig) -> None: 
    np.random.seed(42)
    random.seed(42)

    data_paths: Dict[str, int] = OmegaConf.to_container(cfg.data_paths, resolve=True)
    zarr_path: str = cfg.zarr_path
    zarr_dt: float = cfg.zarr_dt

    if os.path.exists(zarr_path):
        raise FileExistsError(f"Zarr path '{zarr_path}' already exists. Aborting to avoid overwrite.")
    
    image_label_file_map: Dict[str, str] = OmegaConf.to_container(cfg.image_label_file_map, resolve=True)
    proprioception_files: List[Tuple[str, int]] = OmegaConf.to_container(cfg.proprioception_files, resolve=True)

    if cfg.action_as_shifted_prop: 
        action_files = None
    else: 
        action_files: List[str] = OmegaConf.to_container(cfg.action_files, resolve=True)

    traj_dir_list = calculate_traj_dir_list(data_paths=data_paths)

    image_labels = list(image_label_file_map.keys())

    concatenated_image_lists = {label:[] for label in image_labels}
    concatenated_proprioceptions = []
    concatenated_actions = []
    episode_ends = []
    current_end = 0
    
    print(traj_dir_list)
    if cfg.save_clipping_indices: 
        clipping_indices_save_path = zarr_path[:-5] + "_clipping_indices.yaml"
        clipping_indices_dict = {'start': [], 'end': []}

    use_clipping_indices = cfg.get("use_clipping_indices", None)
    if use_clipping_indices is not None: 
        with open(cfg.use_clipping_indices, 'r') as f: 
            clipping_indices_loaded = yaml.safe_load(f)

    traj_dir_num = 0
    for traj_dir in tqdm(traj_dir_list): 
        loaded_proprioception = []
        for prop_file in proprioception_files: 
            prop_data = np.load(os.path.join(traj_dir, prop_file[0]))
            loaded_proprioception.append(prop_data[..., :prop_file[1]])

        loaded_images = {}
        for image_name in image_labels: 
            image_file = image_label_file_map[image_name]
            image_data = np.load(os.path.join(traj_dir, image_file))
            loaded_images[image_name] = image_data
        
        if action_files is not None:
            loaded_actions = []
            for action_file in action_files: 
                action_data = np.load(os.path.join(traj_dir, action_file))
                loaded_actions.append(action_data)
            loaded_actions = np.concatenate(loaded_actions, axis=-1)
        
        with open(os.path.join(traj_dir, "hydra_logs/.hydra/config.yaml"), 'r') as f: 
            run_config = yaml.safe_load(f)
        data_save_dt = run_config["task"]["save_data_dt"]

        idx = 0 
        this_proprioception = []
        if not cfg.action_as_shifted_prop:
            this_action = []
        this_image_lists = {label:[] for label in image_labels}
        save_division = int(zarr_dt / data_save_dt)

        while idx < len(loaded_proprioception[0]): 
            if idx % save_division == 0: 
                this_proprioception.append(
                    np.hstack([prop[idx].flatten() for prop in loaded_proprioception]) # we use [0] here because prop[idx] is a row vector of shape (1, n)
                )

                for image_name in image_labels: 
                    this_image_lists[image_name].append(
                        cv2.resize(
                            loaded_images[image_name][idx][...,:3], 
                            (cfg.zarr_image_width, cfg.zarr_image_height)
                        )
                    )
                
                if action_files is not None: 
                    raise NotImplementedError("Action files processing not implemented yet.")
            
            idx += 1 
        
        this_proprioception = np.array(this_proprioception)

        if cfg.action_as_shifted_prop:
            this_action = copy.deepcopy(this_proprioception)
            this_action = np.concatenate([this_action[1:, :], this_action[-1:, :]], axis=0)
        else:
            this_action = np.array(this_action)
        
        for image_name in image_labels: 
            this_image_lists[image_name] = np.array(this_image_lists[image_name])
        
        if cfg.remove_start_end_idling: 
            if use_clipping_indices is not None:
                first_idx = clipping_indices_loaded['start'][traj_dir_num]
                last_idx = clipping_indices_loaded['end'][traj_dir_num]

            else: 
                first_idx, last_idx = clip_start_end_idle(
                    this_proprioception, 
                    1e-9, 
                    keep_idle=1
                )
            
            if cfg.save_clipping_indices: 
                clipping_indices_dict['start'].append(int(first_idx))
                clipping_indices_dict['end'].append(int(last_idx))
            
            this_proprioception = this_proprioception[first_idx:last_idx]
            this_action = this_action[first_idx:last_idx]
            for image_name in image_labels: 
                this_image_lists[image_name] = this_image_lists[image_name][first_idx:last_idx]

        concatenated_proprioceptions.append(this_proprioception)
        concatenated_actions.append(this_action)
        for image_name in image_labels: 
            concatenated_image_lists[image_name].append(this_image_lists[image_name])
        
        episode_ends.append(current_end + len(this_proprioception))
        current_end += len(this_proprioception)

        traj_dir_num += 1

    root = zarr.open_group(zarr_path, mode='w')
    data_group = root.create_group('data')
    meta_group = root.create_group('meta')

    prop_chunk_size = (1024, this_proprioception.shape[1])
    action_chunk_size = (1024, this_action.shape[1])
    image_chunk_size = (128, *this_image_lists.values().__iter__().__next__()[0].shape)

    concatenated_proprioceptions = np.concatenate(concatenated_proprioceptions, axis=0)
    concatenated_actions = np.concatenate(concatenated_actions, axis=0)
    for image_name in image_labels: 
        concatenated_image_lists[image_name] = np.concatenate(concatenated_image_lists[image_name], axis=0)
        assert concatenated_image_lists[image_name].shape[0] == concatenated_proprioceptions.shape[0]
    
    # this is simply added to the dataset to maintain backward compatibility 
    # some dataloaders need a target even though the policy doesn't use it 
    concatenated_targets = np.zeros_like(concatenated_actions)
    target_chunk_size = action_chunk_size

    episode_ends = np.array(episode_ends)

    # some checks 
    assert episode_ends[-1] == concatenated_proprioceptions.shape[0]
    assert concatenated_proprioceptions.shape[0] == concatenated_actions.shape[0]
    
    data_group.create_dataset(
        'state', 
        data=concatenated_proprioceptions, 
        chunks=prop_chunk_size
    )
    data_group.create_dataset(
        'action', 
        data=concatenated_actions, 
        chunks=action_chunk_size
    )
    data_group.create_dataset(
        'target', 
        data=concatenated_targets, 
        chunks=target_chunk_size
    )

    for image_name in image_labels:
        data_group.create_dataset(
            image_name, 
            data=concatenated_image_lists[image_name], 
            chunks=image_chunk_size
        )
    
    meta_group.create_dataset(
        'episode_ends', 
        data=episode_ends
    )

    if cfg.save_clipping_indices: 
        with open(clipping_indices_save_path, 'w') as f: 
            yaml.dump(clipping_indices_dict, f)

if __name__ == "__main__":
    main()