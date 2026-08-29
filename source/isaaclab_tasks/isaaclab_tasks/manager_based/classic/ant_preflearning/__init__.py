# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Ant with a preflog observation group, registered as its own gym id -- see
ant_preflearning_env_cfg.py. Entirely new package, auto-discovered by
isaaclab_tasks/__init__.py's import_packages() the same way every other task package is; does not
modify isaaclab_tasks.manager_based.classic.ant or its __init__.py. Mirrors
manager_based/classic/cartpole_preflearning/ exactly, adapted for Ant's own (larger) observation
set -- see that package's __init__.py docstring for the use_learned_reward/learned_reward
rationale, which applies identically here.
"""

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-Ant-PrefLearning-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ant_preflearning_env_cfg:AntPrefLearningEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntPrefLearningPPORunnerCfg",
    },
)
