# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka Reach with a preflog observation group and the learned-reward config surface,
registered as its own gym id -- see reach_preflearning_env_cfg.py and
agents/rsl_rl_ppo_cfg.py. Entirely new package, auto-discovered by isaaclab_tasks/__init__.py's
import_packages() the same way every other task package is; does not modify
isaaclab_tasks.manager_based.manipulation.reach or its config/franka/__init__.py.
"""

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-Reach-Franka-PrefLearning-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.reach_preflearning_env_cfg:FrankaReachPrefLearningEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaReachPrefLearningPPORunnerCfg",
    },
)
