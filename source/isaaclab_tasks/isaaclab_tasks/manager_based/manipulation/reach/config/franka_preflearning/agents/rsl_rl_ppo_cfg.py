# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka Reach PPO runner config with the learned-reward config surface attached -- same pattern
as CartpolePrefLearningPPORunnerCfg (see cartpole_preflearning/agents/rsl_rl_ppo_cfg.py), applied
proactively this time (found the hard way for Cartpole, job 30570365: use_learned_reward/
learned_reward are a LiftCubePPORunnerCfg-specific extension, not a generic
RslRlOnPolicyRunnerCfg field). LearnedRewardSettings itself is entirely generic -- reused here via
import, not duplicated a second time.
"""

from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.lift.config.franka.agents.rsl_rl_ppo_cfg import (
    LearnedRewardSettings,
)
from isaaclab_tasks.manager_based.manipulation.reach.config.franka.agents.rsl_rl_ppo_cfg import (
    FrankaReachPPORunnerCfg,
)


@configclass
class FrankaReachPrefLearningPPORunnerCfg(FrankaReachPPORunnerCfg):
    """FrankaReachPPORunnerCfg (imported, unmodified) with the same learned-reward config surface
    LiftCubePPORunnerCfg has -- everything else (num_steps_per_env, policy/algorithm
    hyperparameters) is inherited unchanged.
    """

    use_learned_reward: bool = False
    learned_reward: LearnedRewardSettings = LearnedRewardSettings()
