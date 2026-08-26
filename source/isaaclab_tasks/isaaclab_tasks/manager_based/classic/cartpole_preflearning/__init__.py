# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cartpole with a preflog observation group, registered as its own gym id -- see
cartpole_preflearning_env_cfg.py. Entirely new package, auto-discovered by
isaaclab_tasks/__init__.py's import_packages() the same way every other task package is; does not
modify isaaclab_tasks.manager_based.classic.cartpole or its __init__.py.

rsl_rl_cfg_entry_point points at this package's own agents.rsl_rl_ppo_cfg:CartpolePrefLearningPPORunnerCfg
(not the base task's CartpolePPORunnerCfg) -- found 2026-08-27 that use_learned_reward/learned_reward
are not generic RslRlOnPolicyRunnerCfg fields, only present on task RunnerCfgs that explicitly add
them (as LiftCubePPORunnerCfg does); CartpolePrefLearningPPORunnerCfg adds the same fields, inherited
from CartpolePPORunnerCfg otherwise unchanged, so both plain GT-oracle PPO (use_learned_reward
defaults False) and learned-reward PPO work through the one gym id.
"""

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-Cartpole-PrefLearning-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cartpole_preflearning_env_cfg:CartpolePrefLearningEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CartpolePrefLearningPPORunnerCfg",
    },
)
