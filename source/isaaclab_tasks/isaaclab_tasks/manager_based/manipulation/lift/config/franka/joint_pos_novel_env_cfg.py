# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# ABSOLUTE joint-position action + position-only obs, GRAVITY-ON.
# The fair apples-to-apples BASELINE for the delta experiment (Isaac-Lift-Cube-Franka-Delta-v0):
# base FrankaCubeLiftEnvCfg (absolute JointPositionActionCfg scale=0.5, use_default_offset, gravity ON)
# with joint_vel dropped from the policy obs (36 -> 27), matching the delta obs and the deployed no_vel_gravon.
# ADDITIVE + SELF-CONTAINED: subclass only; no shared-file edits. Registered as Isaac-Lift-Cube-Franka-NoVel-v0.

from isaaclab.utils import configclass

from .joint_pos_env_cfg import FrankaCubeLiftEnvCfg


@configclass
class FrankaCubeLiftNoVelEnvCfg(FrankaCubeLiftEnvCfg):
    """Absolute joint-position action, gravity-on, position-only (27-dim) obs. Delta-experiment baseline."""

    def __post_init__(self):
        super().__post_init__()
        # drop joint velocity from the policy obs (36 -> 27); absolute action + gravity_on inherited from base
        self.observations.policy.joint_vel = None
