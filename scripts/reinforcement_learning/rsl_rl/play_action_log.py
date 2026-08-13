# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# Instrumented copy of scripts/reinforcement_learning/rsl_rl/play.py that logs the RAW policy action
# (deterministic inference mean, pre-env-clip) every step to a .npy file, then exits after N steps.
#
# WHY: training logs only expose "Mean action noise std" (the PPO exploration std, a training-time
# hyperparameter of the Gaussian policy), not the actual action values the policy commands during
# rollout. This script measures the real thing: does the delta-mode policy output near-constant,
# state-independent actions (saturated), which combined with q_target = q_current + scale*action would
# make the resulting joint commands essentially open-loop and unresponsive to the actual robot/object state?
#
# Usage (headless, no video):
#   ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_action_log.py \
#       --task <task-name> --checkpoint <path/to/model_*.pt> --num_envs 32 --n_steps 300 \
#       --out /path/to/actions_<label>.npy --headless

import argparse

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Log raw policy actions for an RSL-RL checkpoint.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--n_steps", type=int, default=300, help="Number of env steps to log before exiting.")
parser.add_argument("--out", type=str, required=True, help="Output .npy path for the logged action tensor.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import os
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg


def main():
    task_name = args_cli.task.split(":")[-1]
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(task_name, args_cli)

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    obs, _ = env.get_observations()
    logged_actions = []
    logged_dq = []  # the *actual* joint-position delta applied this step, if resolvable from the action term
    for step in range(args_cli.n_steps):
        with torch.inference_mode():
            actions = policy(obs)
            logged_actions.append(actions.detach().cpu().numpy().copy())
            obs, _, _, _ = env.step(actions)
        if step % 50 == 0:
            print(f"[INFO] step {step}/{args_cli.n_steps}")

    arr = np.stack(logged_actions, axis=0)  # (n_steps, num_envs, action_dim)
    np.save(args_cli.out, arr)
    print(f"[INFO] Saved raw action log: shape={arr.shape} -> {args_cli.out}")

    # quick summary printed immediately (full analysis done offline from the .npy)
    flat = arr.reshape(-1, arr.shape[-1])
    print("\n--- Per-dim action stats across all (step, env) samples ---")
    print(f"mean: {flat.mean(axis=0)}")
    print(f"std : {flat.std(axis=0)}")
    print(f"min : {flat.min(axis=0)}")
    print(f"max : {flat.max(axis=0)}")
    # temporal variation: std of the PER-ENV-MEAN action across time, vs std across envs at a fixed time.
    # low time-variation + low cross-env variation for a given dim = that dim is ~constant regardless of
    # state (both across different envs' different object/robot configs, AND across the rollout) = saturated/open-loop.
    time_std = arr.mean(axis=1).std(axis=0)  # variation over steps of the across-env mean action
    env_std = arr.std(axis=1).mean(axis=0)  # variation across envs, averaged over steps
    print(f"\nstd OVER TIME of the cross-env-mean action (does the action change as the rollout progresses?): {time_std}")
    print(f"std ACROSS ENVS, averaged over time (does the action differ for different envs' states?)       : {env_std}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
