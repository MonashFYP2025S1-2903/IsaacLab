# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone launcher for DIAYN-style, reward-free skill-discovery pretraining.

Mirrors train.py's environment-construction boilerplate exactly (same AppLauncher/Hydra/env
creation pattern, so a DIAYN run uses the SAME env config and already-tuned PPO/policy
hyperparameters as every other training run in this codebase), but swaps OnPolicyRunner for the
new, standalone DiaynRunner + DiaynVecEnvWrapper after env construction.

Added 2026-08-26 as an entirely new, additive file -- does not modify train.py or any other
existing script. See rsl_rl/rsl_rl/diayn/ and FYP2025S1-2903_deployment_setup_guide.md
(2026-08-26 "bigger DIAYN investment" entry) for the full design rationale: this produces a
genuinely diverse, reward-free-discovered trajectory set as an alternative UPSTREAM data source
for the existing preference-learning pipeline, to be compared against GT-oracle-checkpoint-based
collection -- it does not touch or replace any part of the existing PPO/reward-model/online-PL
machinery.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="DIAYN-style reward-free skill-discovery pretraining.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="Training iterations.")
parser.add_argument("--n_skills", type=int, default=8, help="Number of discrete DIAYN skills.")
parser.add_argument("--disc_hidden_dim", type=int, default=128, help="Discriminator MLP hidden width.")
parser.add_argument("--disc_lr", type=float, default=3e-4, help="Discriminator learning rate.")
parser.add_argument("--disc_train_epochs", type=int, default=4, help="Discriminator epochs per PPO iteration.")
parser.add_argument("--disc_batch_size", type=int, default=4096, help="Discriminator minibatch size.")
parser.add_argument("--disc_buffer_size", type=int, default=200000,
                     help="Discriminator replay buffer capacity (transitions).")
parser.add_argument("--collect_traj_dir", type=str, default="",
                     help="If set, collect trajectories here (same npz schema as the existing pipeline).")
parser.add_argument("--collect_traj_iterations", type=str, default="",
                     help="Comma-separated iteration numbers to open a collection window at.")
parser.add_argument("--collect_traj_window", type=int, default=250,
                     help="How many iterations each collection window stays open.")
parser.add_argument("--collect_traj_env_ids", type=str, default="0,1,2,3,4,5,6,7,8,9",
                     help="Comma-separated env indices to collect from.")
# append RSL-RL cli arguments (unchanged, shared with train.py)
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
from datetime import datetime

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.io import dump_pickle, dump_yaml

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

from rsl_rl.diayn.env_wrapper import DiaynVecEnvWrapper
from rsl_rl.diayn.runner import DiaynRunner

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Pretrain a DIAYN skill-conditioned policy (reward-free)."""
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl, then wrap AGAIN for DIAYN skill-conditioning
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    diayn_env = DiaynVecEnvWrapper(
        env,
        n_skills=args_cli.n_skills,
        discriminator_hidden_dim=args_cli.disc_hidden_dim,
        discriminator_lr=args_cli.disc_lr,
        discriminator_train_epochs=args_cli.disc_train_epochs,
        discriminator_batch_size=args_cli.disc_batch_size,
        discriminator_buffer_size=args_cli.disc_buffer_size,
        device=agent_cfg.device,
    )
    print(f"[DIAYN] {args_cli.n_skills} skills, discriminator hidden_dim={args_cli.disc_hidden_dim}, "
          f"policy obs dim {env.num_obs} -> {diayn_env.num_obs} (skill one-hot appended).")

    runner = DiaynRunner(diayn_env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

    if args_cli.collect_traj_dir:
        target_iterations = [int(x) for x in args_cli.collect_traj_iterations.split(",") if x.strip()]
        env_ids = [int(x) for x in args_cli.collect_traj_env_ids.split(",") if x.strip()]
        run_tag = agent_cfg.experiment_name
        runner.setup_traj_collector(
            out_dir=args_cli.collect_traj_dir, env_ids=env_ids,
            target_iterations=target_iterations, window=args_cli.collect_traj_window, run_tag=run_tag,
        )
        print(f"[TrajCollector] Collecting DIAYN on-policy trajectories to "
              f"{args_cli.collect_traj_dir}/{run_tag} for envs {env_ids} at iterations "
              f"{target_iterations} (window={args_cli.collect_traj_window}).")

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # randomize initial episode lengths (for exploration), same as train.py's init_at_random_ep_len
    diayn_env.episode_length_buf = torch.randint_like(
        diayn_env.episode_length_buf, high=int(diayn_env.max_episode_length)
    )

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
