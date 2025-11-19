from __future__ import annotations

import torch
from typing import TYPE_CHECKING,Sequence
from dataclasses import MISSING
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

class MotionCommand(CommandTerm):
    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.target_lin_vel = torch.zeros(self.num_envs, 3, device=self.device)
        self._resample_command(torch.arange(self.num_envs, device=self.device))
        
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_lin_vel"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.target_lin_vel

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.body_indexes]

    def _update_metrics(self):
        # 计算线速度误差
        self.metrics["error_body_lin_vel"] = torch.norm(self.body_lin_vel_w.mean(dim=1) - self.cfg.target_lin_vel, dim=-1)

    def _update_command(self):
        # 每个episode定期重采样目标
        self.time_steps += 1
        env_ids = torch.where(self.time_steps >= self.cfg.motion_time_step_total)[0]
        if len(env_ids) > 0:
            self._resample_command(env_ids)
            self.time_steps[env_ids] = 0

    def _resample_command(self, env_ids: Sequence[int]):
        """采样新的目标线速度"""
        if len(env_ids) == 0:
            return
        vel_x = sample_uniform(
            self.cfg.velocity_range["x"][0], self.cfg.velocity_range["x"][1], (len(env_ids), 1), device=self.device
        )
        vel_y = sample_uniform(
            self.cfg.velocity_range["y"][0], self.cfg.velocity_range["y"][1], (len(env_ids), 1), device=self.device
        )
        vel_z = torch.zeros((len(env_ids), 1), device=self.device)
        self.target_lin_vel[env_ids] = torch.cat([vel_x, vel_y, vel_z], dim=-1)


    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            # 调试可视化：显示目标速度等
            pass

@configclass
class MotionCommandCfg(CommandTermCfg):

    class_type: type = MotionCommand
    asset_name: str = MISSING
    anchor_body_name: str = MISSING
    body_names: list[str] = MISSING
    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}
    joint_position_range: tuple[float, float] = (-0.52, 0.52)
    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001
    target_lin_vel: torch.Tensor = 1 # 目标线速度
    motion_time_step_total: int = 1000  # 每个episode的时间步长（比如1000步）
