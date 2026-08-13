# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# DELTA action + SLOW ARM (actuator velocity cap) + longer episode. The sim2real controller-gap fix.
# The plain-delta policy assumed a fast-tracking arm (~2.8 rad/s) and went unstable on the real slow
# joint_pos_runner (TRACK_VMAX ~0.3 rad/s). Here we cap the SIM Franka arm joint velocity at RL_VMAX so the
# sim arm moves like the real one, while keeping the PLAIN delta action (scale 0.10, which learns well) so
# the action stays linear (gradients flow) -- unlike clamping the action, which saturated -> NaN + no learning.
# This is the physically-faithful model of the runner: drive toward a position target at max RL_VMAX.
# Position-only obs (27-dim). Params via env vars: DELTA_SCALE(0.10), RL_VMAX(0.30 rad/s), EP_LEN_S(10s).
# ADDITIVE + SELF-CONTAINED (deep-copies the actuators before editing -> no global mutation). Registered as
# Isaac-Lift-Cube-Franka-Delta-RL-v0.

import copy
import os

from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.lift import mdp

from .joint_pos_env_cfg import FrankaCubeLiftEnvCfg


@configclass
class FrankaCubeLiftDeltaRateLimitedEnvCfg(FrankaCubeLiftEnvCfg):
    """Delta action + Franka arm joint-velocity capped at RL_VMAX (sim model of runner TRACK_VMAX); 27-dim obs."""

    def __post_init__(self):
        super().__post_init__()
        # plain DELTA action (relative), scale 0.10 -- linear, learns well
        self.actions.arm_action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            scale=float(os.environ.get("DELTA_SCALE", "0.10")),
            use_zero_offset=True,
        )
        # position-only obs (match 27-dim deployment)
        self.observations.policy.joint_vel = None
        # SLOW the sim arm to the real runner speed: cap arm joint velocity at RL_VMAX (deep-copy first so we
        # never mutate the shared FRANKA_PANDA_HIGH_PD_CFG actuator objects).
        vmax = float(os.environ.get("RL_VMAX", "0.30"))
        self.scene.robot.actuators = copy.deepcopy(self.scene.robot.actuators)
        self.scene.robot.actuators["panda_shoulder"].velocity_limit_sim = vmax
        self.scene.robot.actuators["panda_forearm"].velocity_limit_sim = vmax
        # the slow arm needs time to reach + grasp + lift
        self.episode_length_s = float(os.environ.get("EP_LEN_S", "10.0"))


@configclass
class FrankaCubeLiftDeltaRateLimitedEnvCfg_PLAY(FrankaCubeLiftDeltaRateLimitedEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
