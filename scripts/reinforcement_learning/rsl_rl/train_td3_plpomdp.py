# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Trains PL-POMDP's own (ported) TD3 agent against a learned reward model on an IsaacLab task,
run as-is with no KL-anchor-to-reference mechanism -- Phase 2 of the PL-POMDP migration roadmap
(reward-model core, then PPO, now TD3/SAC). See the approved plan
(C:\\Users\\MEN119\\.claude\\plans\\vivid-fluttering-hare.md, 2026-08-28) for full rationale.

num_envs=1 only this phase (vectorizing TD3 is deferred to Phase 3, once this single-env version
is validated). Structurally simpler than the PPO port's orchestration loop: off-policy methods
store flat (obs, act, rew, obs2, done) transitions with no episode-boundary bookkeeping at
buffer-write time (no GAE, no finish_path-equivalent) -- IsaacLab's own vectorized env auto-resets
internally on episode end, same as every other script in this investigation relies on.

Reuses the same building blocks as train_ppo_plpomdp.py: RslRlVecEnvWrapper (tensor-interface
adapter only, not "using rsl_rl's algorithm"), LearnedRewardWrapper (unchanged), and the
Mean reward / Mean GT reward print convention -- logged every --log_interval steps instead of
per-epoch, since off-policy has no natural epoch boundary.

Usage:
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train_td3_plpomdp.py \\
        --task Isaac-Cartpole-PrefLearning-v0 --headless \\
        --rm_checkpoint /path/to/plpomdp_mlp_.../reward_model.pt \\
        --total_steps 50000
"""

"""Launch Isaac Sim Simulator first."""

import argparse
from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Train PL-POMDP's own TD3 against a learned reward, no KL anchor.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=1, help="Only num_envs=1 is supported this "
                     "phase -- vectorizing TD3 is deferred to Phase 3.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--rm_checkpoint", type=str, default="", help="PL-POMDP-trained reward model "
                     "checkpoint. Required unless --train_on_gt_reward is set.")
parser.add_argument("--reward_weight", type=float, default=0.1, help="Matches every other run this "
                     "session's agent.learned_reward.reward_weight.")
parser.add_argument("--train_on_gt_reward", action="store_true", default=False,
                     help="Train directly on the environment's own ground-truth reward instead of "
                     "a learned reward model -- bypasses --rm_checkpoint/LearnedRewardWrapper "
                     "entirely. Same correctness-check rationale as train_ppo_plpomdp.py's flag "
                     "(added 2026-08-28 after that flag caught a real metric-logging bug in the "
                     "PPO port): expect convergence toward the established GT-oracle ceiling "
                     "(~4.9-4.95 final GT episode return on Cartpole) if the implementation and "
                     "logging are both correct.")
parser.add_argument("--total_steps", type=int, required=True, help="Total environment steps.")
parser.add_argument("--start_steps", type=int, default=10000, help="PL-POMDP's own default -- "
                     "pure-random exploration warmup before the policy is used at all.")
parser.add_argument("--update_after", type=int, default=1000, help="PL-POMDP's own default.")
parser.add_argument("--update_every", type=int, default=50, help="PL-POMDP's own default.")
parser.add_argument("--log_interval", type=int, default=1000, help="Print Mean reward/Mean GT "
                     "reward every this many steps (off-policy has no natural epoch boundary).")
parser.add_argument("--recompute_reward_in_backup", action="store_true", default=False,
                     help="See td3_plpomdp.py's module docstring -- off by default, numerically "
                          "identical to on for this investigation's fixed-reward-checkpoint scope.")
parser.add_argument("--strip_last_action_from_actor_obs", action="store_true", default=False,
                     help="Arm P1, ported from rsl_rl (added 2026-08-31).")
parser.add_argument("--strip_last_action_from_critic_obs", action="store_true", default=False,
                     help="Arm C1, ported from rsl_rl (added 2026-08-31). Independent of "
                     "--strip_last_action_from_actor_obs -- combine both for the P1+C1 test.")
parser.add_argument("--seed", type=int, default=42)
cli_args.add_rsl_rl_args(parser)  # only used here to obtain agent_cfg.clip_actions
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import sys
import time
from collections import deque

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "pref_learning"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "rsl_rl", "rsl_rl"))

import td3_plpomdp
from runners.learned_reward import LearnedRewardCfg, LearnedRewardWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import parse_env_cfg


def main():
    assert args_cli.num_envs == 1, "Only num_envs=1 is supported this phase."
    if not args_cli.train_on_gt_reward:
        assert args_cli.rm_checkpoint, "--rm_checkpoint is required unless --train_on_gt_reward is set."
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)

    task_name = args_cli.task.split(":")[-1]
    agent_cfg = cli_args.parse_rsl_rl_cfg(task_name, args_cli)
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    obs, extras = env.get_observations()
    num_obs = obs.shape[1]
    num_actions = env.num_actions
    device = env.device

    obs_space = spaces.Box(-np.inf, np.inf, shape=(num_obs,), dtype=np.float32)
    act_space = spaces.Box(-1.0, 1.0, shape=(num_actions,), dtype=np.float32)  # matches the
    # RslRlVecEnvWrapper clip_actions bound, same convention as train_ppo_plpomdp.py.

    agent = td3_plpomdp.TD3(
        obs_space, act_space, hidden_sizes=(256, 256),
        gamma=0.99, polyak=0.995, pi_lr=1e-3, q_lr=1e-3,
        start_steps=args_cli.start_steps, act_noise=0.1, target_noise=0.2, noise_clip=0.5,
        update_after=args_cli.update_after, update_every=args_cli.update_every, batch_size=64,
        replay_size=int(1e6), policy_delay=2,
        recompute_reward_in_backup=args_cli.recompute_reward_in_backup, device=device,
        strip_last_action_from_actor_obs=args_cli.strip_last_action_from_actor_obs,
        strip_last_action_from_critic_obs=args_cli.strip_last_action_from_critic_obs,
    )

    if args_cli.train_on_gt_reward:
        learned_reward = None
        print("Training directly on GT reward (--train_on_gt_reward) -- no reward model loaded.")
    else:
        lr_cfg = LearnedRewardCfg(rm_checkpoint=args_cli.rm_checkpoint, reward_weight=args_cli.reward_weight)
        learned_reward = LearnedRewardWrapper(num_obs, num_actions, device=device, config=lr_cfg)

    print(f"num_obs={num_obs} num_actions={num_actions} device={device}")
    print(f"total_steps={args_cli.total_steps} start_steps={args_cli.start_steps} "
          f"update_after={args_cli.update_after} update_every={args_cli.update_every}")

    gt_reward_window = []
    learned_reward_window = []
    # Per-EPISODE cumulative return, matching on_policy_runner.py's gt_rewbuffer/rewbuffer
    # convention (not the same quantity as gt_reward_window's raw per-step average below) --
    # added 2026-08-28 after discovering the same mismatch in train_ppo_plpomdp.py (see that
    # file's identical comment for the full story: the established ~4.94 baseline is a per-
    # episode SUM, not a per-step mean).
    gt_ep_rewbuffer = deque(maxlen=100)
    learned_ep_rewbuffer = deque(maxlen=100)
    cur_gt_ep_reward = 0.0
    cur_learned_ep_reward = 0.0
    stats = {}
    window_start_time = time.time()

    for t in range(args_cli.total_steps):
        obs_np = obs[0].detach().cpu().numpy()
        action_np = agent.get_action(obs_np, t)
        action_t = torch.as_tensor(action_np, dtype=torch.float32, device=device).unsqueeze(0)

        obs_before_step = obs
        next_obs, gt_rewards, dones, infos = env.step(action_t)
        next_obs, gt_rewards, dones = next_obs.to(device), gt_rewards.to(device), dones.to(device)

        if learned_reward is not None:
            learned_r = learned_reward.predict(obs_before_step, action_t, next_obs=next_obs)
            rew = float(learned_r.item())
        else:
            rew = float(gt_rewards.item())  # --train_on_gt_reward: train directly on GT reward

        gt_reward_window.append(float(gt_rewards.item()))
        learned_reward_window.append(rew)
        cur_gt_ep_reward += float(gt_rewards.item())
        cur_learned_ep_reward += rew

        time_outs = infos.get("time_outs", torch.zeros_like(dones))
        terminal = bool(dones[0].item())
        done_true_termination = terminal and not bool(time_outs[0].item())

        next_obs_np = next_obs[0].detach().cpu().numpy()
        lr_for_recompute = learned_reward if args_cli.recompute_reward_in_backup else None
        agent.store_and_update(obs_np, action_np, rew, next_obs_np,
                                float(done_true_termination), t, stats, learned_reward=lr_for_recompute)

        if terminal:  # episode boundary (true termination OR timeout) -- matches
            # on_policy_runner.py's `dones > 0` gate.
            gt_ep_rewbuffer.append(cur_gt_ep_reward)
            learned_ep_rewbuffer.append(cur_learned_ep_reward)
            cur_gt_ep_reward = 0.0
            cur_learned_ep_reward = 0.0

        obs = next_obs

        if (t + 1) % args_cli.log_interval == 0:
            window_time = time.time() - window_start_time
            print(f"--- t={t + 1}/{args_cli.total_steps} ---")
            print(f"  Mean reward (per-step): {np.mean(learned_reward_window):.2f}")
            print(f"  Mean GT reward (per-step): {np.mean(gt_reward_window):.2f}")
            if gt_ep_rewbuffer:
                print(f"  Mean episode return (learned): {np.mean(learned_ep_rewbuffer):.2f}  "
                      f"Mean episode return (GT): {np.mean(gt_ep_rewbuffer):.2f}  "
                      f"(n={len(gt_ep_rewbuffer)} episodes)")
            if "LossQ" in stats:
                print(f"  LossQ: {stats['LossQ'][-1]:.4f}  LossPi: {stats['LossPi'][-1]:.4f}")
            print(f"  Window time: {window_time:.2f}s")
            gt_reward_window = []
            learned_reward_window = []
            window_start_time = time.time()

    print("Training complete.")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
