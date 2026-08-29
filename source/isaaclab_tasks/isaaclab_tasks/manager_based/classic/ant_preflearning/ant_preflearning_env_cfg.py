# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Ant variant with an added "preflog" observation group, for the IsaacLab-Ant preference-learning
cross-check (see FYP2025S1-2903_deployment_setup_guide.md, 2026-08-28 "MuJoCo/Gymnasium
cross-check" entry -- this task closes the open "did you run IsaacLab's own Ant" question raised
there). Standard Isaac-Ant-v0 (isaaclab_tasks.manager_based.classic.ant) has no "preflog"
observation group -- play_collect_pref_data.py hardcodes a read of
infos["observations"]["preflog"], so plain Ant would KeyError there. Rather than touch
on_policy_runner.py or the existing ant_env_cfg.py, this adds the missing group via subclassing,
entirely additively -- same pattern as cartpole_preflearning_env_cfg.py.

Unlike Cartpole's fully-observed 4-dim MDP, Ant's PolicyCfg has 10 terms (partially-observed,
much higher-dimensional, matching Lift's situation more than Cartpole's) -- the preflog group
mirrors every one of them in un-concatenated (dict) form using the SAME generic
isaaclab_tasks.manager_based.classic.humanoid.mdp functions/params the base AntEnvCfg's own
PolicyCfg already uses (imported, not reimplemented), so pref_learning/task_adapters.py's
calculate_reward_ant() can reconstruct all 7 of Ant's reward terms (progress, alive, upright,
move_to_target, action_l2, energy, joint_pos_limits) from logged fields alone -- every raw
quantity those reward functions depend on (base height/lin_vel/ang_vel/yaw_roll/angle_to_target/
up_proj/heading_proj, joint_pos_norm, joint_vel_rel, feet_body_forces, last action) is present.

One extra preflog-only field beyond the PolicyCfg mirror (same category as Reach's own
ee_pos_w/target_pos_w preflog additions -- needed for offline reward reconstruction, not part of
the live policy's own observation): humanoid/mdp/rewards.py's `progress_reward` needs raw
world-frame root XY position each step (potential = -norm(target_xy - root_xy)/step_dt, a
step-to-step delta), which none of the standard PolicyCfg terms expose -- they only expose
DERIVED angle/projection features (base_angle_to_target, base_heading_proj), never the raw
position itself. isaaclab.envs.mdp's own generic `root_pos_w` function subtracts env_origins,
which would NOT match progress_reward's actual computation (it uses asset.data.root_pos_w
directly, un-adjusted) -- so a tiny local function is added below instead of reusing that one.
"""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.classic.humanoid.mdp as mdp
from isaaclab_tasks.manager_based.classic.ant.ant_env_cfg import AntEnvCfg


def raw_root_pos_w(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """Raw (non-origin-subtracted) world-frame root position -- matches EXACTLY what
    humanoid/mdp/rewards.py's progress_reward and observations.py's base_heading_proj/
    base_angle_to_target use internally (asset.data.root_pos_w[:, :3]), unlike
    isaaclab.envs.mdp.root_pos_w which subtracts env_origins. See module docstring above.
    """
    asset = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, :3]


@configclass
class AntPrefLearningObservationsCfg:
    """Observation specifications: the existing PolicyCfg group, unmodified, plus a new PrefLogCfg
    group (dict-form, unconcatenated) for preference-learning trajectory collection.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Identical to AntEnvCfg's own PolicyCfg -- duplicated here (not imported) only because
        IsaacLab's @configclass observation groups are defined inline per env cfg; every term uses
        the same generic, unmodified humanoid.mdp functions the base task uses.
        """

        base_height = ObsTerm(func=mdp.base_pos_z)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        base_yaw_roll = ObsTerm(func=mdp.base_yaw_roll)
        base_angle_to_target = ObsTerm(func=mdp.base_angle_to_target, params={"target_pos": (1000.0, 0.0, 0.0)})
        base_up_proj = ObsTerm(func=mdp.base_up_proj)
        base_heading_proj = ObsTerm(func=mdp.base_heading_proj, params={"target_pos": (1000.0, 0.0, 0.0)})
        joint_pos_norm = ObsTerm(func=mdp.joint_pos_limit_normalized)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.2)
        feet_body_forces = ObsTerm(
            func=mdp.body_incoming_wrench,
            scale=0.1,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", body_names=["front_left_foot", "front_right_foot", "left_back_foot", "right_back_foot"]
                )
            },
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class PrefLogCfg(ObsGroup):
        """Preference-learning trajectory logging group (added, mirrors cartpole_preflearning's
        own PrefLogCfg convention -- unconcatenated dict output, same funcs/params as PolicyCfg
        above so every raw quantity calculate_reward_ant() needs is present in the log.
        """

        obs_base_height = ObsTerm(func=mdp.base_pos_z)
        obs_base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        obs_base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        obs_base_yaw_roll = ObsTerm(func=mdp.base_yaw_roll)
        obs_base_angle_to_target = ObsTerm(func=mdp.base_angle_to_target, params={"target_pos": (1000.0, 0.0, 0.0)})
        obs_base_up_proj = ObsTerm(func=mdp.base_up_proj)
        obs_base_heading_proj = ObsTerm(func=mdp.base_heading_proj, params={"target_pos": (1000.0, 0.0, 0.0)})
        obs_joint_pos_norm = ObsTerm(func=mdp.joint_pos_limit_normalized)
        obs_joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.2)
        obs_feet_body_forces = ObsTerm(
            func=mdp.body_incoming_wrench,
            scale=0.1,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", body_names=["front_left_foot", "front_right_foot", "left_back_foot", "right_back_foot"]
                )
            },
        )
        current_action = ObsTerm(func=mdp.last_action)
        obs_root_pos_w = ObsTerm(func=raw_root_pos_w)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    preflog: PrefLogCfg = PrefLogCfg()


@configclass
class AntPrefLearningEnvCfg(AntEnvCfg):
    """AntEnvCfg (imported, unmodified) with the preflog observation group added."""

    observations: AntPrefLearningObservationsCfg = AntPrefLearningObservationsCfg()
