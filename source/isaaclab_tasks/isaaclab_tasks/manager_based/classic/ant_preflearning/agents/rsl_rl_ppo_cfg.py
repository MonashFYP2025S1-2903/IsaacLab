# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Ant PPO runner config with the learned-reward config surface attached -- same rationale as
cartpole_preflearning/agents/rsl_rl_ppo_cfg.py: use_learned_reward/learned_reward are not generic
RslRlOnPolicyRunnerCfg fields, only present on task RunnerCfgs that explicitly add them.
LearnedRewardSettings itself is entirely generic (no Lift-specific content) -- reused here via
import, not duplicated, same as Cartpole's variant does.
"""

from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.classic.ant.agents.rsl_rl_ppo_cfg import AntPPORunnerCfg
from isaaclab_tasks.manager_based.manipulation.lift.config.franka.agents.rsl_rl_ppo_cfg import (
    LearnedRewardSettings,
)


@configclass
class AntPrefLearningPPORunnerCfg(AntPPORunnerCfg):
    """AntPPORunnerCfg (imported, unmodified) with the same learned-reward config surface
    LiftCubePPORunnerCfg/CartpolePrefLearningPPORunnerCfg have -- everything else
    (num_steps_per_env, policy/algorithm hyperparameters) is inherited unchanged.
    """

    use_learned_reward: bool = False
    learned_reward: LearnedRewardSettings = LearnedRewardSettings()
