from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.utils.math import matrix_from_quat, subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def _ensure_float_tensor(t: torch.Tensor) -> torch.Tensor:
    """保证为 float32 并在正确 device 上（兼容 IsaacLab 的 tensors）。"""
    if not torch.is_tensor(t):
        t = torch.tensor(t, dtype=torch.float32)
    return t.to(dtype=torch.float32)


def robot_body_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """
    各 body 相对于 root 的位置（展平成向量）。
    兼容性：基于 robot.data 而非 motion reference，num_bodies 从实际数据推断。
    返回 shape: (num_envs, num_bodies*3)
    """
    command = env.command_manager.get_term(command_name)
    robot = command.robot

    body_pos_w = robot.data.body_pos_w  # (num_envs, num_bodies, 3)
    num_bodies = body_pos_w.shape[1]

    root_pos_w = _ensure_float_tensor(robot.data.root_pos_w)  # (num_envs, 3)
    root_quat_w = _ensure_float_tensor(robot.data.root_quat_w)  # (num_envs, 4)
    body_quat_w = _ensure_float_tensor(robot.data.body_quat_w)

    # repeat anchor -> (num_envs, num_bodies, 3/4)
    anchor_pos = root_pos_w[:, None, :].repeat(1, num_bodies, 1)
    anchor_quat = root_quat_w[:, None, :].repeat(1, num_bodies, 1)

    pos_b, _ = subtract_frame_transforms(anchor_pos, anchor_quat, body_pos_w, body_quat_w)
    return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """
    各 body 相对于 root 的朝向表示（使用 quaternion->矩阵并取前两列，保持原代码习惯）。
    返回 shape: (num_envs, num_bodies * 3 * 2) -> 因为取矩阵前两列，每列3维。
    """
    command = env.command_manager.get_term(command_name)
    robot = command.robot

    body_pos_w = robot.data.body_pos_w
    num_bodies = body_pos_w.shape[1]

    root_pos_w = _ensure_float_tensor(robot.data.root_pos_w)
    root_quat_w = _ensure_float_tensor(robot.data.root_quat_w)
    body_quat_w = _ensure_float_tensor(robot.data.body_quat_w)

    anchor_pos = root_pos_w[:, None, :].repeat(1, num_bodies, 1)
    anchor_quat = root_quat_w[:, None, :].repeat(1, num_bodies, 1)

    _, ori_b = subtract_frame_transforms(anchor_pos, anchor_quat, body_pos_w, body_quat_w)

    mat = matrix_from_quat(ori_b)  # shape: (num_envs, num_bodies, 3, 3)
    return mat[..., :2].reshape(mat.shape[0], -1)


def root_lin_vel_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """
    根部线速度，body/frame 表示（优先使用 robot.data.root_lin_vel_b，如果不存在回退到 world 并转换）。
    返回 (num_envs, 3)
    """
    command = env.command_manager.get_term(command_name)
    robot = command.robot

    # 尝试直接拿 body-frame 的 root 线速度
    if hasattr(robot.data, "root_lin_vel_b"):
        vel = robot.data.root_lin_vel_b
    else:
        # 回退：world 速度 -> body frame via quat
        vel_w = _ensure_float_tensor(robot.data.root_lin_vel_w)  # (N,3)
        quat = _ensure_float_tensor(robot.data.root_quat_w)  # (N,4)
        # 转换：v_b = R(q)^T * v_w
        R = matrix_from_quat(quat)  # (N,3,3)
        vel = torch.bmm(R.transpose(1, 2), vel_w.unsqueeze(-1)).squeeze(-1)

    return _ensure_float_tensor(vel)


def root_ang_vel_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """
    根部角速度，body-frame 表示。返回 (num_envs, 3)
    """
    command = env.command_manager.get_term(command_name)
    robot = command.robot

    if hasattr(robot.data, "root_ang_vel_b"):
        ang = robot.data.root_ang_vel_b
    else:
        ang = _ensure_float_tensor(robot.data.root_ang_vel_w)  # 直接返回 world 表示（agent 可学到）
    return _ensure_float_tensor(ang)


def joint_pos(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """关节角，(num_envs, num_joints)"""
    command = env.command_manager.get_term(command_name)
    robot = command.robot
    return _ensure_float_tensor(robot.data.joint_pos)


def joint_vel(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """关节角速度，(num_envs, num_joints)"""
    command = env.command_manager.get_term(command_name)
    robot = command.robot
    return _ensure_float_tensor(robot.data.joint_vel)


def target_lin_vel(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """
    从 MotionCommand 读取当前目标线速度（num_envs, 3）。
    保证 shape 与 device 正确。
    """
    command = env.command_manager.get_term(command_name)
    # command.command() 或 command.command 属性在你的 MotionCommand 中返回 target_lin_vel
    t = command.command
    # 若 cfg 中为单个常量，扩展到 num_envs（防护）
    if t.dim() == 1:
        t = t.unsqueeze(0).expand(env.num_envs, -1)
    return _ensure_float_tensor(t)


def rl_velocity_tracking_obs(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """
    汇总观测：为线速度跟踪任务定制的 observation。
    拼接顺序（可在 ObservationManager 中分别使用这些单项，也可直接使用此汇总项）：
      [root_lin_vel_b (3), root_ang_vel_b (3), body_pos_b (num_bodies*3),
       body_ori_b (num_bodies*3*2), joint_pos (num_joints), joint_vel (num_joints),
       target_lin_vel (3)]
    返回 (num_envs, obs_dim)
    """
    # get pieces
    rl_root_lin = root_lin_vel_b(env, command_name)
    rl_root_ang = root_ang_vel_b(env, command_name)
    body_pos = robot_body_pos_b(env, command_name)
    body_ori = robot_body_ori_b(env, command_name)
    jpos = joint_pos(env, command_name)
    jvel = joint_vel(env, command_name)
    tgt = target_lin_vel(env, command_name)

    # concat (保证 dtype/device 一致)
    pieces = [rl_root_lin, rl_root_ang, body_pos, body_ori, jpos, jvel, tgt]
    pieces = [p if p is not None else torch.zeros((env.num_envs, 0), device=rl_root_lin.device) for p in pieces]
    obs = torch.cat(pieces, dim=-1)
    return obs
