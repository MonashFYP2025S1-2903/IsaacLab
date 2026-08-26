# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka Reach variant with an added "preflog" observation group, for the preference-learning
investigation's second simpler-task control (see FYP2025S1-2903_deployment_setup_guide.md,
2026-08-27 "Simpler-task controls" entry). Mirrors the Cartpole-PrefLearning precedent: subclasses
the existing, unmodified FrankaReachEnvCfg rather than editing it, adding only what
on_policy_runner.py/play_collect_pref_data.py need (a "preflog" group) plus what
task_adapters.py's calculate_reward_reach needs to recompute the real reward offline.

Reach's live PolicyCfg already has joint_pos, joint_vel, pose_command, actions -- fully sufficient
for the privileged-obs composition (obs_privileged mirrors it exactly, see task_adapters.py's
REACH_OBS_PRIVILEGED_FIELDS comment on why this must match exactly, not just be "at least as
much"). What policy does NOT expose is the actual end-effector pose or the command's target pose
in WORLD frame -- reach/mdp/rewards.py's own position_command_error/orientation_command_error
compute these via isaaclab.utils.math.combine_frame_transforms/quat_mul before taking the error,
so this preflog group logs the SAME already-transformed world-frame quantities directly (reusing
those exact utility functions), rather than re-deriving the transform offline in
calculate_reward_reach where a mistake would be easy to make and hard to notice.
"""

import torch

from isaaclab.assets import RigidObject
from isaaclab.envs import mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import combine_frame_transforms, quat_mul

from isaaclab_tasks.manager_based.manipulation.reach.config.franka.joint_pos_env_cfg import FrankaReachEnvCfg

_EE_ASSET_CFG = SceneEntityCfg("robot", body_names=["panda_hand"])


def log_ee_pos_w(env, asset_cfg: SceneEntityCfg = _EE_ASSET_CFG) -> torch.Tensor:
    """End-effector (panda_hand) position in world frame -- same source
    position_command_error() itself reads (asset.data.body_pos_w)."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.body_pos_w[:, asset_cfg.body_ids[0]]


def log_ee_quat_w(env, asset_cfg: SceneEntityCfg = _EE_ASSET_CFG) -> torch.Tensor:
    """End-effector (panda_hand) orientation in world frame -- same source
    orientation_command_error() itself reads (asset.data.body_quat_w)."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.body_quat_w[:, asset_cfg.body_ids[0]]


def log_command_target_pos_w(env, command_name: str, asset_cfg: SceneEntityCfg = _EE_ASSET_CFG) -> torch.Tensor:
    """Command target position transformed into world frame -- identical computation to
    position_command_error()'s own des_pos_w, logged directly instead of only being used
    internally to compute a scalar error."""
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)
    return des_pos_w


def log_command_target_quat_w(env, command_name: str, asset_cfg: SceneEntityCfg = _EE_ASSET_CFG) -> torch.Tensor:
    """Command target orientation transformed into world frame -- identical computation to
    orientation_command_error()'s own des_quat_w."""
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_quat_w, des_quat_b)
    return des_quat_w


@configclass
class FrankaReachPrefLearningObservationsCfg:
    """The existing PolicyCfg group, unmodified, plus a new PrefLogCfg group (dict-form,
    unconcatenated) for preference-learning trajectory collection.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Identical to FrankaReachEnvCfg's own PolicyCfg -- duplicated here (not imported) only
        because IsaacLab's @configclass observation groups are defined inline per env cfg; the
        values themselves are the same functions/params the base task uses.
        """

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "ee_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PrefLogCfg(ObsGroup):
        """Preference-learning trajectory logging group (mirrors the Cartpole-PrefLearning and
        Franka Lift PrefLogCfg convention -- unconcatenated dict output).
        """

        obs_joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        obs_joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        obs_pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "ee_pose"})
        current_action = ObsTerm(func=mdp.last_action)

        # World-frame quantities needed for calculate_reward_reach (task_adapters.py) --
        # not derivable from the above without redoing the frame transform offline.
        ee_pos_w = ObsTerm(func=log_ee_pos_w)
        ee_quat_w = ObsTerm(func=log_ee_quat_w)
        target_pos_w = ObsTerm(func=log_command_target_pos_w, params={"command_name": "ee_pose"})
        target_quat_w = ObsTerm(func=log_command_target_quat_w, params={"command_name": "ee_pose"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    preflog: PrefLogCfg = PrefLogCfg()


@configclass
class FrankaReachPrefLearningEnvCfg(FrankaReachEnvCfg):
    """FrankaReachEnvCfg (imported, unmodified) with the preflog observation group added."""

    observations: FrankaReachPrefLearningObservationsCfg = FrankaReachPrefLearningObservationsCfg()
