# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# DELTA (relative) joint-position action variant of the Franka cube-lift task.
# ADDITIVE: subclasses FrankaCubeLiftEnvCfg and ONLY swaps the arm action to a RELATIVE (delta) joint
# position command (q_target = q_current + scale*action). Everything else is inherited UNCHANGED:
# gravity-on cube, rewards, and position-only obs via the LIFT_INCLUDE_JOINT_VEL env var.
# Registered as a SEPARATE task (Isaac-Lift-Cube-Franka-Delta-v0) so existing pipelines are untouched.
#
# Why: the absolute action (q_default + 0.5*a) commands large jumps the slow/rate-limited real controller
# cannot follow -> jerky, poor sim2real. Delta actions bound each step -> smoother, more transferable.

from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.lift import mdp

from .joint_pos_env_cfg import FrankaCubeLiftEnvCfg


@configclass
class FrankaCubeLiftDeltaEnvCfg(FrankaCubeLiftEnvCfg):
    """Franka cube-lift with RELATIVE (delta) joint-position actions; gravity_on + rewards inherited."""

    def __post_init__(self):
        # parent builds the whole env (gravity_on cube, absolute action, rewards, obs)
        super().__post_init__()
        # swap ABSOLUTE -> RELATIVE (delta): q_target = q_current + scale*action.
        # small scale keeps per-step motion bounded so a slow/rate-limited real arm can track it.
        self.actions.arm_action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot", joint_names=["panda_joint.*"], scale=0.05, use_zero_offset=True
        )


@configclass
class FrankaCubeLiftDeltaEnvCfg_PLAY(FrankaCubeLiftDeltaEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
