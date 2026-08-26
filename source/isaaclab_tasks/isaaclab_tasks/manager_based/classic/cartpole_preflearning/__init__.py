# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cartpole with a preflog observation group, registered as its own gym id -- see
cartpole_preflearning_env_cfg.py. Entirely new package, auto-discovered by
isaaclab_tasks/__init__.py's import_packages() the same way every other task package is; does not
modify isaaclab_tasks.manager_based.classic.cartpole or its __init__.py.
"""

import gymnasium as gym

gym.register(
    id="Isaac-Cartpole-PrefLearning-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cartpole_preflearning_env_cfg:CartpolePrefLearningEnvCfg",
        "rsl_rl_cfg_entry_point": (
            "isaaclab_tasks.manager_based.classic.cartpole.agents.rsl_rl_ppo_cfg:CartpolePPORunnerCfg"
        ),
    },
)
