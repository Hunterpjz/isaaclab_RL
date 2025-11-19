from __future__ import annotations
import re
import torch
from typing import TYPE_CHECKING

from isaaclab.utils.math import euler_xyz_from_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.assets import Articulation, RigidObject
from isaaclab.sensors import ContactSensorCfg
from isaaclab.managers import SceneEntityCfg

def time_out(env: ManagerBasedRLEnv, max_episode_length: float) -> torch.Tensor:
    """Terminate if the episode length exceeds max_episode_length."""
    return env.episode_length_buf > max_episode_length


def roll_pitch_exceeded(env, roll_threshold=0.8, pitch_threshold=1.0):
    """Terminate if robot base tilts too much (roll/pitch)."""
    base_quat = env.scene["robot"].data.root_quat_w  # shape: (num_envs, 4)
    roll, pitch, yaw = euler_xyz_from_quat(base_quat)
    terminated = torch.logical_or(torch.abs(roll) > roll_threshold,
                                  torch.abs(pitch) > pitch_threshold)
    # print(f"[DEBUG] Environments terminated due to roll/pitch: {terminated.nonzero(as_tuple=True)[0]}")
    return terminated

