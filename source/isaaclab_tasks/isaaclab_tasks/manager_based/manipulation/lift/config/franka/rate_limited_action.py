# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# Rate-limited DELTA joint-position action = RelativeJointPositionAction whose per-control-step joint
# change is clamped to vmax * step_dt. This replicates the REAL joint_pos_runner TRACK_VMAX rate limit
# INSIDE the simulator, closing the sim2real controller gap that made the plain-delta policy unstable on
# the slow real arm (the policy commanded ~5 rad/s; the real runner caps at ~0.3 rad/s -> divergence).
# vmax is a cfg field so it can be FIXED (match the runner) or randomized per-episode later (control-dyn DR).

import torch

from isaaclab.envs.mdp.actions.joint_actions import RelativeJointPositionAction
from isaaclab.envs.mdp.actions.actions_cfg import RelativeJointPositionActionCfg
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass


class RateLimitedRelativeJointPositionAction(RelativeJointPositionAction):
    """Relative (delta) joint-position action with a per-step travel cap of vmax * step_dt (rad)."""

    cfg: "RateLimitedRelativeJointPositionActionCfg"

    def apply_actions(self):
        # processed_actions = scale * raw  (use_zero_offset -> offset 0). Clamp the per-step delta to the
        # arm max travel per control step, then apply RELATIVE to the MEASURED joints (as RelativeJointPos).
        max_step = self.cfg.vmax * self._env.step_dt
        limited = torch.clamp(self.processed_actions, -max_step, max_step)
        target = limited + self._asset.data.joint_pos[:, self._joint_ids]
        self._asset.set_joint_position_target(target, joint_ids=self._joint_ids)


@configclass
class RateLimitedRelativeJointPositionActionCfg(RelativeJointPositionActionCfg):
    """Rate-limited relative joint-position action (models joint_pos_runner TRACK_VMAX)."""

    class_type: type[ActionTerm] = RateLimitedRelativeJointPositionAction

    vmax: float = 0.30
    """Max joint speed (rad/s). Per-step delta is clamped to vmax * step_dt. Match the real runner TRACK_VMAX."""
