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

class PlanarDiffusionPolicyDrakeController(BaseControllerLeafSystem):
    def __init__(self,
                 root_cfg,
    ):
        super().__init__(root_cfg)

        cfg = self.cfg
        sys.path.insert(0, cfg.relative_path_to_diffusion_model)

        load_normalizer_from_file = cfg.get("load_normalizer_from_file", True)
        infer_frozen_policy = cfg.get("infer_frozen_policy", False)

        self.policy, training_cfg = load_policy(cfg.checkpoint_path, 
                                                load_normalizer_from_file=load_normalizer_from_file, 
                                                infer_frozen_policy=infer_frozen_policy)

        self.n_action_steps = cfg.n_action_steps_override
        self.n_obs_steps = self.policy.n_obs_steps
        self._actions = deque([], maxlen=self.n_action_steps)

        self.obs_names = list(training_cfg.shape_meta.obs.keys())
        self.obs_shapes = training_cfg.shape_meta.obs

        # define dequeues for policy input 
        self._dict_of_obs_buffers = {}
        for obs_name_key in self.obs_names: 
            self._dict_of_obs_buffers[obs_name_key] = deque([], maxlen=self.policy.n_obs_steps)

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

    def DoCalcOutput(self, context: Context, output): 
        time = context.get_time()

        use_controller = self.use_controller_in.Eval(context)

        if use_controller != InputPortIndex(1): 
            output.set_value(self.initial_planar_command)
            return 
    
        # wait time at the start 
        while len(self._dict_of_obs_buffers[self.obs_names[0]]) < self.policy.n_obs_steps - 1: 
            self._update_deques(context) 
            output.set_value(self.current_action)
        
        self._update_deques(context)

        if len(self._actions) == 0: 
            obs_dict = self._deque_to_dict()
            with torch.no_grad(): 
                predicted_actions = self.policy.predict_action(
                    obs_dict, 
                    use_DDIM=self.use_DDIM
                )['action_pred'][0].cpu().numpy()

            if self.debug: 
                self._visualize_trajectory(predicted_actions)
            
            self._actions.extend(predicted_actions[self.n_obs_steps-1:self.n_obs_steps-1 + self.n_action_steps])

        self.current_action = self._actions.popleft()
        output.set_value(self.current_action) 

    def _visualize_trajectory(self, trajectory: np.ndarray):
        """Visualize predicted trajectory in meshcat with points at each timestep."""
        assert self.meshcat is not None, "Meshcat instance is not set for visualization."
        # Clear previous trajectory visualization
        self.meshcat.Delete("predicted_trajectory")

        # Draw each action as a sphere
        for i, action in enumerate(trajectory):
            # Green for executed actions (first n_action_steps), yellow for others
            if i < self.n_obs_steps -1 + self.n_action_steps and i >= self.n_obs_steps-1:
                color = Rgba(0.0, 1.0, 0.0, 0.8)  # Green
            else:
                color = Rgba(1.0, 1.0, 0.0, 0.8)  # Yellow

            sphere_path = f"predicted_trajectory/point_{i}"
            self.meshcat.SetObject(sphere_path, Sphere(0.005), color)
            # Position at (x, y, z=0.0) - assuming planar pushing
            self.meshcat.SetTransform(sphere_path, RigidTransform([action[0]+self.translation_offset[0], 
                                                                   action[1]+self.translation_offset[1], 
                                                                   0.0]))

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
                        torch.from_numpy(np.moveaxis(img, -1, -3) / 255.0)
                        for img in self._dict_of_obs_buffers[name]
                    ], 
                    dim=0
                ).reshape(
                    1, 
                    self.policy.n_obs_steps,
                    self.obs_shapes[name].shape[0],
                    self.obs_shapes[name].shape[1],
                    self.obs_shapes[name].shape[2]
                ).to(self._device)

                data['obs'][name] = this_processed_buffer

            else: 
                data['obs'][name] = torch.from_numpy(np.stack(
                    self._dict_of_obs_buffers[name]
                )).unsqueeze(0).to(self._device)
        
        return data 
    
    def reset(self, meshcat): 
        self._actions.clear()
        for key in self._dict_of_obs_buffers.keys(): 
            self._dict_of_obs_buffers[key].clear()
        
        self.meshcat = meshcat

        self.current_action = self.initial_planar_command

    def set_post_connection_values(self, initial_planar_command, ee_body_index, translation_offset=np.array([0.0, 0.0])): 
        self._planar_body_index = ee_body_index
        self.initial_planar_command = initial_planar_command
        self.translation_offset = translation_offset # add to go from iiwa to world, subtract for world to iiwa 

    def update(self):
        print("Update method called - no operation defined.")

    
