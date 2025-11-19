from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def alive_reward(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    return torch.ones(env.num_envs, dtype=torch.float, device=env.device)


def tracking_lin_vel_reward(env, sensor_cfg: SceneEntityCfg, sigma: float = 0.25) -> torch.Tensor:
    lin_vel_func = env.observation_manager.cfg.policy.base_lin_vel.func
    lin_vel = lin_vel_func(env) 
    target_velocity = torch.tensor([1.0, 0.0, 0.0], device=env.device)

    lin_vel_error = torch.sum((target_velocity - lin_vel) ** 2, dim=1)

    return torch.exp(-lin_vel_error / sigma)


def lin_vel_z_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # 获取线性速度的 Z 分量
    lin_vel_func = env.observation_manager.cfg.policy.base_lin_vel.func
    lin_vel = lin_vel_func(env)
    return torch.square(lin_vel[:, 2])


def ang_vel_xy_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # 获取角速度的 XY 分量
    ang_vel_func = env.observation_manager.cfg.policy.base_ang_vel.func
    ang_vel = ang_vel_func(env)
    return torch.sum(torch.square(ang_vel[:, :2]), dim=1)


def orientation_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # 获取重力方向投影的前两维
    gravity_func = env.observation_manager.cfg.policy.projected_gravity.func
    gravity = gravity_func(env)
    return torch.sum(torch.square(gravity[:, :2]), dim=1)

def termination_reward(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # 获取重置标志并返回奖励
    reset_buf_func = env.observation_manager.cfg.policy.reset_buf.func
    reset_buf = reset_buf_func(env)
    return reset_buf 

def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)

def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward

def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward