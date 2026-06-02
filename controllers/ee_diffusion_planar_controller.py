from pydrake.all import(
    LeafSystem, MultibodyPlant, 
    AbstractValue, 
    RigidTransform,
    PixelType, 
    Value, 
    Image, 
    Context, 
    Rgba, 
    Sphere, 
    InputPortIndex
) 

from collections import deque
import cv2
import numpy as np 
import sys 

from utils.diffusion_utils import load_policy
from .base_controller import BaseControllerLeafSystem 

import torch 

custom_dataset_functions = []

class PlanarDiffusionPolicyDrakeController(BaseControllerLeafSystem):
    dataset_type_to_controller_init_map = {
        "planar_pushing_attention_dataset_constant_then_skip_frames": lambda instance: instance._preprocess_constant_then_skip_frames(instance.training_cfg),
        "planar_pushing_attention_dataset_skip_frames": lambda instance: instance._preprocess_skip_frames(instance.training_cfg), 
        "planar_pushing_attention_dataset_extra_frame": lambda instance: instance._preprocess_extra_frame(instance.training_cfg)
    }
    def __init__(self,
                 root_cfg,
                 policy_and_cfg = None
    ):
        super().__init__(root_cfg)

        cfg = self.cfg
        relative_path_to_diffusion_model = cfg.relative_path_to_diffusion_model
        if relative_path_to_diffusion_model not in sys.path: 
            sys.path.insert(0, cfg.relative_path_to_diffusion_model)

        load_normalizer_from_file = cfg.get("load_normalizer_from_file", True)
        infer_frozen_policy = cfg.get("infer_frozen_policy", False)
        self.use_ptp_realign = cfg.get("use_ptp_realign", False)
        
        if policy_and_cfg is not None:
            self.policy, self.training_cfg = policy_and_cfg
        else:
            self.policy, self.training_cfg = load_policy(cfg.checkpoint_path, 
                                                    load_normalizer_from_file=load_normalizer_from_file, 
                                                    infer_frozen_policy=infer_frozen_policy)

        special_sequencing_dataset_type = self.training_cfg.task.dataset._target_.split(".")[-2]
        if special_sequencing_dataset_type in self.dataset_type_to_controller_init_map: 
            self.dataset_type_to_controller_init_map[special_sequencing_dataset_type](self)
        else: 
            self._preprocess_default(self.training_cfg)

        self.n_action_steps = cfg.n_action_steps_override
        self._actions = deque([], maxlen=self.n_action_steps)

        self.obs_names = list(self.training_cfg.shape_meta.obs.keys())
        self.obs_shapes = self.training_cfg.shape_meta.obs

        # define dequeues for policy input 
        self._dict_of_obs_buffers = {}
        for obs_name_key in self.obs_names: 
            self._dict_of_obs_buffers[obs_name_key] = deque([], maxlen=self.n_obs_steps)

        # declare input ports 
        self._body_poses_in = self.DeclareAbstractInputPort(
            "body_poses", AbstractValue.Make([RigidTransform()])
        )

        self._dict_of_camera_input_ports = {}
        for obs_name in self.obs_names: 
            if "camera" in obs_name: 
                self._dict_of_camera_input_ports[obs_name] = self.DeclareAbstractInputPort(
                    name=f"{obs_name}_in", 
                    model_value=Value(Image[PixelType.kRgba8U]())
                )

        # if the controller is always supposed to be active, set the value for this to 1 permanently 
        self.use_controller_in = self.DeclareAbstractInputPort(
            "use_controller", 
            AbstractValue.Make(InputPortIndex(0)), 
        )

        # declare output ports 
        self.output = self.DeclareVectorOutputPort(
            "planar_command_out", 
            2, 
            self.DoCalcOutput
        )

        self._device = cfg.policy_device
        self.policy.to(self._device)

        self.current_action = cfg.initial_planar_command

        self.use_DDIM = cfg.get("use_DDIM", False)

        self.horizon=self.policy.horizon
        _n_past = getattr(self.policy, "n_past_action_steps", None)
        if _n_past is not None: 
            self.past_predicted = _n_past - 1
        else: 
            self.past_predicted = self.policy.n_obs_steps - 1
        
        if self.use_ptp_realign: 
            self._executed_actions = deque([], maxlen=self.past_predicted)

    def DoCalcOutput(self, context: Context, output): 
        time = context.get_time()

        use_controller = self.use_controller_in.Eval(context)

        if use_controller != InputPortIndex(1): 
            output.set_value(self.initial_planar_command)
            return 
    
        # wait time at the start 
        while len(self._dict_of_obs_buffers[self.obs_names[0]]) < self.n_obs_steps - 1: 
            self._update_deques(context) 
            output.set_value(self.current_action)
        
        self._update_deques(context)

        if len(self._actions) == 0: 
            obs_dict = self._deque_to_dict()
            with torch.inference_mode(): 
                predicted_actions_base = self.policy.predict_action(
                    obs_dict, 
                    use_DDIM=self.use_DDIM
                )['action_pred'].cpu().numpy()

            if (self.use_ptp_realign and self.cfg.num_samples > 1 and len(self._executed_actions) == self.past_predicted): 
                # print("using ptp")
                past_cmds = np.stack(list(self._executed_actions), axis=0)
                pred_past = predicted_actions_base[:, :self.past_predicted, :]
                dists = ((pred_past - past_cmds[None]) ** 2).sum(axis=(1,2))
                best_idx = int(np.argmin(dists))
            else: 
                best_idx = 0
            
            predicted_actions = predicted_actions_base[best_idx]

            if self.cfg.visualize_actions: 
                self._visualize_trajectory(predicted_actions_base)
            
            self._actions.extend(predicted_actions[self.past_predicted:self.past_predicted + self.n_action_steps])

        self.current_action = self._actions.popleft()
        if self.use_ptp_realign: 
            self._executed_actions.append(np.asarray(self.current_action, dtype=np.float32).copy())
        # if time > 20: 
        #     from PIL import Image 
        #     for ll, image in enumerate(self._dict_of_obs_buffers['overhead_camera']): 
        #         img = Image.fromarray(image.astype(np.uint8))
        #         img.save(f"temp_images/over_image_{ll+10}.png")
            
        #     for ll, image in enumerate(self._dict_of_obs_buffers['wrist_camera']): 
        #         img = Image.fromarray(image.astype(np.uint8))
        #         img.save(f"temp_images/wrist_image_{ll+10}.png")

        #     breakpoint()
        output.set_value(self.current_action) 

    def _visualize_trajectory(self, trajectory: np.ndarray):
        """Visualize predicted trajectory in meshcat with points at each timestep."""
        assert self.meshcat is not None, "Meshcat instance is not set for visualization."
        # Clear previous trajectory visualization
        self.meshcat.Delete("predicted_trajectory")

        # Draw each action as a sphere
        for j, traj in enumerate(trajectory):
            for i, action in enumerate(traj):
                # Green for executed actions (first n_action_steps), yellow for others
                if i < self.past_predicted + self.n_action_steps and i >= self.past_predicted:
                    color = Rgba(0.0, 1.0, 0.0, 0.8)  # Green
                else:
                    color = Rgba(1.0, 1.0, 0.0, 0.8)  # Yellow

                sphere_path = f"predicted_trajectory/point_{j}_{i}"
                self.meshcat.SetObject(sphere_path, Sphere(0.005), color)
                # Position at (x, y, z=0.0) - assuming planar pushing
                self.meshcat.SetTransform(sphere_path, RigidTransform([action[0]+self.translation_offset[0], 
                                                                    action[1]+self.translation_offset[1], 
                                                                    0.0]))
        if self.debug: 
            self.meshcat.Delete("agent_pos")
            for j, traj in enumerate(self._dict_of_obs_buffers['agent_pos']): 
                
                color = Rgba(0.0, 0.0, 1.0, 0.8)  # Blue for agent positions in buffer
                sphere_path = f"agent_pos/point_{j}"
                self.meshcat.SetObject(sphere_path, Sphere(0.005), color)
                self.meshcat.SetTransform(sphere_path, RigidTransform([traj[0]+self.translation_offset[0], 
                                                                    traj[1]+self.translation_offset[1], 
                                                                    0.0]))
                if j in self.indices_to_keep: 
                    color = Rgba(1.0, 0.0, 0.0, 0.8)  # Red for kept agent positions
                    sphere_path = f"agent_pos/point_{j}_extra"
                    self.meshcat.SetObject(sphere_path, Sphere(0.005), color)
                    self.meshcat.SetTransform(sphere_path, RigidTransform([traj[0]+self.translation_offset[0], 
                                                                        traj[1]+self.translation_offset[1], 
                                                                        0.0]))
                    
            import matplotlib.pyplot as plt 
            plt.figure()
            for j, traj in enumerate(self._dict_of_obs_buffers['agent_pos']):
                if j in self.indices_to_keep: 
                    plt.scatter(float(traj[0]), float(traj[1]), 'ro')  # red for kept positions
                else: 
                    plt.scatter(float(traj[0]), float(traj[1]), 'bo')  # blue for other positions
            plt.title("Agent positions in buffer (red=kept, blue=discarded)")
            plt.show()

    def _update_deques(self, context: Context): 
        camera_images = {}

        for name in self.obs_names: 
            if "camera" in name: 
                camera_images[name] = self._dict_of_camera_input_ports[name].Eval(context).data 
        
        planar_reference_body_pose = self._body_poses_in.Eval(context)[
            int(self._planar_body_index)
        ]
        
        planar_translation = planar_reference_body_pose.translation().flatten()

        for name in self.obs_names: 
            if "camera" in name: 
                self._dict_of_obs_buffers[name].append(
                    cv2.resize(camera_images[name][...,:self.obs_shapes[name].shape[0]], (self.obs_shapes[name].shape[1], self.obs_shapes[name].shape[2]))
                ) # this assumes RGBA iput and shape specific for CHW format 
            
            else: 
                self._dict_of_obs_buffers[name].append(
                    planar_translation[: self.obs_shapes[name].shape[0]] - self.translation_offset
                )
    
    def _deque_to_dict(self): 
        data = {'obs': {}}

        for name in self.obs_names: 
            if "camera" in name: 
                this_processed_buffer = torch.cat(
                    [
                        torch.from_numpy(np.moveaxis(self._dict_of_obs_buffers[name][k], -1, -3) / 255.0)
                        for k in self.indices_to_keep
                    ], 
                    dim=0
                ).reshape(
                    1, 
                    self.policy.n_obs_steps,
                    self.obs_shapes[name].shape[0],
                    self.obs_shapes[name].shape[1],
                    self.obs_shapes[name].shape[2]
                ).to(self._device)

                data['obs'][name] = this_processed_buffer.repeat(self.cfg.num_samples, 1, 1, 1, 1).float()

            else: 
                data['obs'][name] = torch.from_numpy(np.stack(
                    [
                        self._dict_of_obs_buffers[name][k] for k in self.indices_to_keep
                    ]
                )).unsqueeze(0).to(self._device).repeat(self.cfg.num_samples, 1, 1).float()
        
        return data 
    
    def reset(self, meshcat): 
        self._actions.clear()
        for key in self._dict_of_obs_buffers.keys(): 
            self._dict_of_obs_buffers[key].clear()
        
        self.meshcat = meshcat

        self.current_action = self.initial_planar_command

        if self.use_ptp_realign: 
            self._executed_actions.clear()

    def close(self): 
        if hasattr(self, "policy") and self.policy is not None: 
            self.policy.to("cpu")
        del self.policy 
        del self._dict_of_obs_buffers
        self._dict_of_obs_buffers = None 
        self.policy = None 
        if self._device == "cuda": 
            torch.cuda.empty_cache()

    def set_post_connection_values(self, initial_planar_command, ee_body_index, translation_offset=np.array([0.0, 0.0])): 
        self._planar_body_index = ee_body_index
        self.initial_planar_command = initial_planar_command
        self.translation_offset = translation_offset # add to go from iiwa to world, subtract for world to iiwa 

    def update(self):
        print("Update method called - no operation defined.")

    # pre and post processing functions for skip connection long context policies         

    def _preprocess_extra_frame(self, policy_cfg): 
        print("Running preprocessing for extra frame policy...")
        self.n_obs_steps = policy_cfg.task.dataset.n_obs_steps 
        
        continuous_till = policy_cfg.task.dataset.continuous_till

        self.indices_to_keep = np.concatenate([np.array([0]), np.arange(self.n_obs_steps - continuous_till, self.n_obs_steps)])

        assert len(self.indices_to_keep) == self.policy.n_obs_steps, "Number of indices to keep must match the policy's n_obs_steps."

    def _preprocess_constant_then_skip_frames(self, policy_cfg): 
        print("Running preprocessing for constant then skip frames policy...")
        self.n_obs_steps = policy_cfg.task.dataset.n_obs_steps

        H = self.n_obs_steps 
        K = policy_cfg.task.dataset.skip_every 
        R = policy_cfg.task.dataset.constant_upto 

        if H - R - 2 >= 0: 
            older = np.arange(H-R-2, -1, -K)[::-1]
        else: 
            older = np.empty(0, dtype=int)
        
        recent = np.arange(H-R, H)

        self.indices_to_keep = np.concatenate([older, recent])

        assert len(self.indices_to_keep) == self.policy.n_obs_steps, "Number of indices to keep must match the policy's n_obs_steps."

    def _preprocess_skip_frames(self, policy_cfg): 
        print("Running preprocessing for skip frames policy...")
        self.n_obs_steps = policy_cfg.task.dataset.n_obs_steps

        H = self.n_obs_steps 
        K = policy_cfg.task.dataset.skip_every 

        self.indices_to_keep = np.arange(H-1, -1, -K)[::-1]

        assert len(self.indices_to_keep) == self.policy.n_obs_steps, "Number of indices to keep must match the policy's n_obs_steps."
    
    def _preprocess_default(self, policy_cfg): 
        print("No special preprocessing for policy - using default last n_obs_steps frames as input.")
        self.n_obs_steps = policy_cfg.n_obs_steps 

        self.indices_to_keep = np.arange(self.n_obs_steps)




    
