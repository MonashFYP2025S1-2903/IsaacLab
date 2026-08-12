# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)

    return 1 - torch.tanh(object_ee_distance / std)


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w, dim=1)
    # rewarded if the object is lifted above the threshold
    return (object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))


# ============================================================
# LEARNED REWARD TERM
# ============================================================
# This is the core integration: a reward term that calls your
# trained preference-based reward model instead of hand-crafted
# reward components.

class LearnedRewardTerm:
    """
    Stateful reward term that loads a trained reward model checkpoint
    and computes rewards from camera images + proprioception.

    This is initialised once when the environment is created, and
    called every step by the RewardManager.
    """

    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        from isaaclab_reward_wrapper import LearnedRewardFunction
        self.reward_fn = LearnedRewardFunction.from_checkpoint(
            checkpoint_path, device=device,
        )
        self.device = device

        # Read camera config from checkpoint
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.camera_config = ckpt.get("camera_config", "front")
        
        self.camera_names = "camera_ext1"

        print(f"[LearnedRewardTerm] Loaded: {checkpoint_path}")
        print(f"  Camera config: {self.camera_config} → {self.camera_names}")

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ) -> torch.Tensor:
        """
        Called by RewardManager each step.

        Gathers camera images and proprioception from the environment,
        runs them through the learned reward model, returns per-env rewards.
        """
        num_envs = env.num_envs
        robot: RigidObject = env.scene[robot_cfg.name]
        object: RigidObject = env.scene[object_cfg.name]

        # ----------------------------------------------------------
        # 1. Gather camera images: (num_envs, num_cams, 3, H, W)
        # ----------------------------------------------------------
        camera_images = []
        for cam_name in self.camera_names:
            sensor_name = cam_name
            if sensor_name in env.scene.sensors:
                cam = env.scene.sensors[sensor_name]
                # cam.data.output["rgb"] is (num_envs, H, W, 4) RGBA uint8
                rgb = cam.data.output["rgb"][..., :3]  # drop alpha
                # Rearrange to (num_envs, 3, H, W)
                rgb = rgb.permute(0, 3, 1, 2).contiguous()
                camera_images.append(rgb)
            else:
                raise RuntimeError(
                    f"Camera 'cam_{cam_name}' not found in scene sensors. "
                    f"Available: {list(env.scene.sensors.keys())}. "
                    f"Checkpoint expects camera_config='{self.camera_config}'."
                )

        # Stack: (num_envs, num_cams, 3, H, W)
        images = torch.stack(camera_images, dim=1)

        # ----------------------------------------------------------
        # 2. Gather proprioception: (num_envs, proprio_dim)
        # ----------------------------------------------------------
        # Match the proprio features used during reward model training.
        # From your data collection: joint_pos(7) + joint_vel(7) + ee_pos(3) = 17
        joint_pos = robot.data.joint_pos[:, :7]     # 7 arm joints
        joint_vel = robot.data.joint_vel[:, :7]      # 7 arm joints

        
        object_pos_w = object.data.root_pos_w[:, :3]
        object_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, object_pos_w)

        proprio = torch.cat([joint_pos, joint_vel, object_pos_b], dim=-1)

        # ----------------------------------------------------------
        # 3. Gather actions: (num_envs, action_dim)
        # ----------------------------------------------------------
        actions = env.action_manager.action  # (num_envs, action_dim)

        # ----------------------------------------------------------
        # 4. Compute learned reward
        # ----------------------------------------------------------
        rewards = self.reward_fn(images, proprio, actions)

        return rewards


# Global singleton — initialised lazily on first call
_learned_reward_term: LearnedRewardTerm | None = None


def learned_reward(
    env: ManagerBasedRLEnv,
    checkpoint_path: str = "",
) -> torch.Tensor:
    """
    MDP reward function that calls the learned reward model.

    This is the function passed to RewTerm. It lazily initialises the
    reward model on first call.
    """
    global _learned_reward_term
    if _learned_reward_term is None:
        device = str(env.device)
        _learned_reward_term = LearnedRewardTerm(checkpoint_path, device)
    return _learned_reward_term(env)
