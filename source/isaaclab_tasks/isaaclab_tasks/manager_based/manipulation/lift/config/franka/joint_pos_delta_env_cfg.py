# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# DELTA (relative) joint-position action + position-only obs variant of the Franka cube-lift task.
# ADDITIVE + SELF-CONTAINED: subclasses FrankaCubeLiftEnvCfg and only (1) swaps the arm action to a
# RELATIVE (delta) joint command (q_target = q_current + scale*action) and (2) drops joint_vel from the
# policy obs (-> 27-dim, matching deploy_zmq.py). Gravity-on cube + rewards inherited. No shared-file edits.
# Registered as a SEPARATE task (Isaac-Lift-Cube-Franka-Delta-v0) so existing pipelines are untouched.
#
# Why: absolute action (q_default + 0.5*a) commands large jumps the slow/rate-limited real controller cannot
# follow -> jerky, poor sim2real. Delta actions bound each step -> smoother, more transferable to the real arm.

import os
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.lift import mdp

from .joint_pos_env_cfg import FrankaCubeLiftEnvCfg


@configclass
class FrankaCubeLiftDeltaEnvCfg(FrankaCubeLiftEnvCfg):
    """Franka cube-lift, RELATIVE (delta) joint-position actions + position-only (27-dim) obs; gravity_on inherited."""

    def __post_init__(self):
        super().__post_init__()
        # (1) ABSOLUTE -> RELATIVE (delta): q_target = q_current + scale*action; small scale = bounded, trackable step
        self.actions.arm_action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot", joint_names=["panda_joint.*"], scale=float(os.environ.get("DELTA_SCALE", "0.05")), use_zero_offset=True
        )
        # (2) position-only obs (match 27-dim deployment): drop joint velocity from the policy obs
        self.observations.policy.joint_vel = None


@configclass
class FrankaCubeLiftDeltaEnvCfg_PLAY(FrankaCubeLiftDeltaEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
