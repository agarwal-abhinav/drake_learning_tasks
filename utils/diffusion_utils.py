import torch 
import dill 

from omegaconf import OmegaConf
import hydra 

import os 

def load_policy(policy_name: str, 
                dataset_zarr: str = None, 
                load_normalizer_from_file: bool = False, 
                infer_frozen_policy: bool = False): 
    print(f"Loading policy from: {policy_name}")
    payload = torch.load(open(policy_name, "rb"), pickle_module=dill)

    model_cfg = payload["cfg"]

    if infer_frozen_policy: 
        OmegaConf.set_struct(model_cfg.policy, False)
        model_cfg.policy.inference_loading = True
        OmegaConf.set_struct(model_cfg.policy, True)

    model_workspace_cls = hydra.utils.get_class(model_cfg._target_)
    model_workspace = model_workspace_cls(model_cfg)
    model_workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    if load_normalizer_from_file: 
        normalizer_path = os.path.join(os.path.dirname(os.path.dirname(policy_name)), "normalizer.pt")
        normalizer = torch.load(normalizer_path, weights_only=False)
    else:
        if dataset_zarr is not None: 
            model_cfg.task.dataset.zarr_path = dataset_zarr
        dataset = hydra.utils.instantiate(model_cfg.task.dataset)
        normalizer = dataset.get_normalizer()

    policy = model_workspace.model 
    policy.set_normalizer(normalizer)
    if model_cfg.training.use_ema: 
        policy = model_workspace.ema_model 
    policy.set_normalizer(normalizer)

    policy.eval()
    return policy, model_cfg