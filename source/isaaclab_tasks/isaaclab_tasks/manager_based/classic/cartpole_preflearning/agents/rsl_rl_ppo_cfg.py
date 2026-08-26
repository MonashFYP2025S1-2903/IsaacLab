# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cartpole PPO runner config with the learned-reward config surface attached.

Discovered 2026-08-27 (job 30570365): `use_learned_reward`/`learned_reward` are NOT generic
RslRlOnPolicyRunnerCfg fields -- they were added specifically to LiftCubePPORunnerCfg (see
lift/config/franka/agents/rsl_rl_ppo_cfg.py). Hydra's struct-mode override checking rejects
`agent.use_learned_reward=true` against any RunnerCfg that doesn't declare the field, which is
why job 30570365's Stage 5/6 PPO-with-learned-reward runs failed against the plain
CartpolePPORunnerCfg. `LearnedRewardSettings` itself is entirely generic (no Lift-specific
content at all) -- reused here via import, not duplicated.
"""

from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.classic.cartpole.agents.rsl_rl_ppo_cfg import CartpolePPORunnerCfg
from isaaclab_tasks.manager_based.manipulation.lift.config.franka.agents.rsl_rl_ppo_cfg import (
    LearnedRewardSettings,
)


@configclass
class CartpolePrefLearningPPORunnerCfg(CartpolePPORunnerCfg):
    """CartpolePPORunnerCfg (imported, unmodified) with the same learned-reward config surface
    LiftCubePPORunnerCfg has -- everything else (num_steps_per_env, policy/algorithm
    hyperparameters) is inherited unchanged.
    """

    use_learned_reward: bool = False
    learned_reward: LearnedRewardSettings = LearnedRewardSettings()
