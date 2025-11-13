from typing import Dict
import numpy as np
from dataclasses import dataclass

from .base_controller import BaseController, BatchedPose

from pydrake.all import (
    StartMeshcat,
    Rotation,
    RotationMatrix,
)

class GamePadEndEffectorPlanarController(BaseController):
    def __init__(self, root_cfg):
        super().__init__(root_cfg)
        cfg = self.cfg
        self.translation_scale = cfg.translation_scale
        self.deadzone = cfg.deadzone
        self.orientation = np.array(cfg.orientation) # we fix the ee orientation
        self.z_values = np.array(cfg.z_values)
        self.gamepad_orientation = np.array(
            [
                [cfg.gamepad_orientation.x[0], cfg.gamepad_orientation.y[0]],
                [cfg.gamepad_orientation.x[1], cfg.gamepad_orientation.y[1]],
            ]
        )
        self.meshcat = None

        if self.z_values <= 0.15:
            response = input("z_value is less than 15cm. This may cause collisions with the table for the cylinder end effector. Continue? (y/n)")
            if response.lower() != 'y':
                exit()

        # get_info variables
        self.prev_button_values = [0.0 for _ in range(17)]
        self.button_index = {            
            'A': 0,
            'B': 1,
            'Y': 2,
            'X': 3,
            'LB': 4,
            'RB': 5,
            'LT': 6,
            'RT': 7,
            'BACK': 8,
            'START': 9,
            'L_JOY': 10,
            'R_JOY': 11,
            'UP': 12,
            'DOWN': 13,
            'LEFT': 14,
            'RIGHT': 15,
            'LOGO': 16,
        }
        self.ENABLE_INDEX = self.button_index[cfg.enable_button]
        self.TERMINATE_INDEX = self.button_index[cfg.terminate_button]
        self.MODE_SWITCH_INDEX = self.button_index[cfg.mode_switch_button]
        self.TERMINATE_AND_SAVE_INDEX = self.button_index[cfg.terminate_and_save_button]

        self.grasp = None

    def reset(self, meshcat):
        self.meshcat = meshcat
        self.init_pose = None

        if self.meshcat.GetGamepad().index != None:
            print("Gamepad connected.")
        else:
            print("Please connect Gamepad.")
            while self.meshcat.GetGamepad().index == None:
                continue
            print("Gamepad connected.")
            
    def create_stick_dead_zone(self, x, y): 
        stick = np.array([x, y])
        m = np.linalg.norm(stick)

        if m < self.deadzone: 
            return np.array([0, 0])
        over = (m - self.deadzone) / (1 - self.deadzone)
        return stick * over / m

    def get_offset_ee(self): 
        gamepad = self.meshcat.GetGamepad()
        position = self.create_stick_dead_zone(gamepad.axes[0], gamepad.axes[1])
        pos_offset = self.translation_scale * self.gamepad_orientation @ position
        pos_offset = np.array([*pos_offset, 0.0]) # hardcode z offset to 0
        rot_offset = RotationMatrix().matrix()
        return BatchedPose(pos_offset, rot_offset)

    def update(self, obs_dict: Dict, *args, **kwargs):
        ee_pos = obs_dict["ee_pos"]
        ee_quat = obs_dict["ee_quat"]
        
        # # Get offset from gamepad
        offset_ee: BatchedPose = self.get_offset_ee()

        # Compute target pose
        if self.init_pose is None: 
            self.init_pose = ee_pos
        target_pos = self.init_pose + offset_ee.pos
        target_pos[:,-1] = self.z_values # hard code z values
        self.init_pose = target_pos

        return target_pos, self.orientation, self.grasp
    
    def get_info(self):
        button_values = self.meshcat.GetGamepad().button_values

        enable = np.array([False])
        terminate = np.array([False])
        mode_switch = np.array([False])
        terminate_and_save = np.array([False])

        if self._pressed_enable(button_values):
            enable = np.array([True])
        if self._pressed_terminate(button_values):
            terminate = np.array([True])
        if self._pressed_terminate_and_save(button_values): 
            terminate_and_save = np.array([True])
        # mode switch upon button release
        if self._is_close(self.prev_button_values[self.MODE_SWITCH_INDEX], 1.0):
            if not self._pressed_mode_switch(button_values):
                mode_switch = np.array([True])

        self.prev_button_values = button_values
        return enable, terminate, terminate_and_save, mode_switch

    def _pressed_enable(self, button_values):
        return self._is_close(button_values[self.ENABLE_INDEX], 1.0, 1e-3)
    
    def _pressed_terminate(self, button_values):
        return self._is_close(button_values[self.TERMINATE_INDEX], 1.0, 1e-3)
    
    def _pressed_terminate_and_save(self, button_values): 
        return self._is_close(button_values[self.TERMINATE_AND_SAVE_INDEX], 1.0, 1e-3)
    
    def _pressed_mode_switch(self, button_values):
        return self._is_close(button_values[self.MODE_SWITCH_INDEX], 1.0, 1e-3)

    def _is_close(self, a, b, eps=1e-3):
        return abs(a - b) < eps