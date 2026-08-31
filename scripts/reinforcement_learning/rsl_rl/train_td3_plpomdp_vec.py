# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Vectorized (num_envs > 1) counterpart to train_td3_plpomdp.py, added 2026-08-29 to test the
"Phase 3" deferral that script's own docstring flags ("vectorizing TD3 is deferred to Phase 3").
Does NOT modify train_td3_plpomdp.py, td3_plpomdp.py, or replay_buffer_plpomdp.py -- those stay
exactly as validated; this is a separate script built on td3_plpomdp_vec.TD3Vec (a subclass that
reuses the original's update math unchanged, see that file's module docstring) and
replay_buffer_plpomdp_vec.VecReplayBuffer.

Structural differences from train_td3_plpomdp.py, all downstream of obs/actions/rewards/dones now
carrying a leading N (num_envs) dimension instead of being squeezed to scalars via obs[0]/.item():
- Per-env episode-return bookkeeping (a (N,) running-sum array, flushed to the reward deques
  per-env exactly when that env's own `dones` fires) instead of two scalar accumulators.
- `global_step` is TOTAL transitions collected across all N envs (incremented by N each
  iteration), not iteration count -- see td3_plpomdp_vec.py's module docstring for why: this
  preserves start_steps/update_after/update_every/--total_steps's real-world meaning unchanged
  from the single-env script, so the same CLI defaults mean the same thing regardless of
  num_envs. --log_interval uses the same block-counter technique update_every does internally
  (global_step increments by N and would otherwise skip most exact multiples of log_interval).

Usage:
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train_td3_plpomdp_vec.py \\
        --task Isaac-Lift-Cube-Franka-Absolute-DR-TableFix-v0 --num_envs 2000 --headless \\
        --rm_checkpoint /path/to/plpomdp_mlp_.../reward_model.pt \\
        --total_steps 300000
"""

"""Launch Isaac Sim Simulator first."""

import argparse
from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Train PL-POMDP's own TD3 (vectorized) against a learned reward, no KL anchor.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of parallel environments. "
                     "Unlike train_td3_plpomdp.py, values > 1 are supported and expected here.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--rm_checkpoint", type=str, default="", help="PL-POMDP-trained reward model "
                     "checkpoint. Required unless --train_on_gt_reward is set.")
parser.add_argument("--reward_weight", type=float, default=0.1, help="Matches every other run this "
                     "session's agent.learned_reward.reward_weight.")
parser.add_argument("--train_on_gt_reward", action="store_true", default=False,
                     help="Train directly on the environment's own ground-truth reward instead of "
                     "a learned reward model -- bypasses --rm_checkpoint/LearnedRewardWrapper "
                     "entirely. Same correctness-check rationale as train_td3_plpomdp.py's flag.")
parser.add_argument("--total_steps", type=int, required=True, help="Total environment transitions "
                     "across ALL envs combined (same real-world meaning as train_td3_plpomdp.py's "
                     "--total_steps at num_envs=1) -- runs total_steps // num_envs iterations.")
parser.add_argument("--start_steps", type=int, default=10000, help="PL-POMDP's own default, in "
                     "TOTAL transitions -- unchanged meaning from train_td3_plpomdp.py.")
parser.add_argument("--update_after", type=int, default=1000, help="PL-POMDP's own default, in "
                     "TOTAL transitions.")
parser.add_argument("--update_every", type=int, default=50, help="PL-POMDP's own default, in "
                     "TOTAL transitions.")
parser.add_argument("--batch_size", type=int, default=64, help="PL-POMDP's own default; FastTD3 "
                     "(arXiv:2505.22642) ablates much larger batch sizes as a top-two contributor "
                     "to performance at high parallelism -- kept overridable for the "
                     "hyperparameter search.")
parser.add_argument("--log_interval", type=int, default=1000, help="Print Mean reward/Mean GT "
                     "reward every this many TOTAL transitions.")
parser.add_argument("--recompute_reward_in_backup", action="store_true", default=False,
                     help="See td3_plpomdp.py's module docstring -- off by default, numerically "
                          "identical to on for this investigation's fixed-reward-checkpoint scope.")
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

import td3_plpomdp_vec
from runners.learned_reward import LearnedRewardCfg, LearnedRewardWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import parse_env_cfg


def main():
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
    num_envs = args_cli.num_envs
    device = env.device

    obs_space = spaces.Box(-np.inf, np.inf, shape=(num_obs,), dtype=np.float32)
    act_space = spaces.Box(-1.0, 1.0, shape=(num_actions,), dtype=np.float32)  # matches the
    # RslRlVecEnvWrapper clip_actions bound, same convention as train_td3_plpomdp.py.

    agent = td3_plpomdp_vec.TD3Vec(
        obs_space, act_space, num_envs=num_envs, hidden_sizes=(256, 256),
        gamma=0.99, polyak=0.995, pi_lr=1e-3, q_lr=1e-3,
        start_steps=args_cli.start_steps, act_noise=0.1, target_noise=0.2, noise_clip=0.5,
        update_after=args_cli.update_after, update_every=args_cli.update_every, batch_size=args_cli.batch_size,
        replay_size=int(1e6), policy_delay=2,
        recompute_reward_in_backup=args_cli.recompute_reward_in_backup, device=device,
    )

    if args_cli.train_on_gt_reward:
        learned_reward = None
        print("Training directly on GT reward (--train_on_gt_reward) -- no reward model loaded.")
    else:
        lr_cfg = LearnedRewardCfg(rm_checkpoint=args_cli.rm_checkpoint, reward_weight=args_cli.reward_weight)
        learned_reward = LearnedRewardWrapper(num_obs, num_actions, device=device, config=lr_cfg)

    print(f"num_obs={num_obs} num_actions={num_actions} num_envs={num_envs} device={device}")
    print(f"total_steps={args_cli.total_steps} start_steps={args_cli.start_steps} "
          f"update_after={args_cli.update_after} update_every={args_cli.update_every} "
          f"(all in TOTAL transitions across {num_envs} envs)")
    sys.stdout.flush()

    gt_reward_window = []
    learned_reward_window = []
    # Per-EPISODE cumulative return, matching on_policy_runner.py's gt_rewbuffer/rewbuffer
    # convention -- one running sum PER ENV now (shape (num_envs,)), flushed to the shared deque
    # individually for whichever envs' `dones` fires this step, matching how a vectorized PPO
    # rollout does the same bookkeeping.
    gt_ep_rewbuffer = deque(maxlen=100)
    learned_ep_rewbuffer = deque(maxlen=100)
    cur_gt_ep_reward = np.zeros(num_envs, dtype=np.float32)
    cur_learned_ep_reward = np.zeros(num_envs, dtype=np.float32)
    stats = {}
    window_start_time = time.time()
    global_step = 0
    last_log_block = -1
    num_iterations = args_cli.total_steps // num_envs

    for it in range(num_iterations):
        obs_np = obs.detach().cpu().numpy()
        action_np = agent.get_action_batch(obs_np, global_step)
        action_t = torch.as_tensor(action_np, dtype=torch.float32, device=device)

        obs_before_step = obs
        next_obs, gt_rewards, dones, infos = env.step(action_t)
        next_obs, gt_rewards, dones = next_obs.to(device), gt_rewards.to(device), dones.to(device)

        if learned_reward is not None:
            learned_r = learned_reward.predict(obs_before_step, action_t, next_obs=next_obs)
            rew_np = learned_r.detach().cpu().numpy()
        else:
            rew_np = gt_rewards.detach().cpu().numpy()  # --train_on_gt_reward

        gt_rewards_np = gt_rewards.detach().cpu().numpy()
        gt_reward_window.extend(gt_rewards_np.tolist())
        learned_reward_window.extend(rew_np.tolist())
        cur_gt_ep_reward += gt_rewards_np
        cur_learned_ep_reward += rew_np

        time_outs = infos.get("time_outs", torch.zeros_like(dones))
        terminal_np = dones.detach().cpu().numpy().astype(bool)
        time_outs_np = time_outs.detach().cpu().numpy().astype(bool)
        done_true_termination_np = terminal_np & ~time_outs_np

        next_obs_np = next_obs.detach().cpu().numpy()
        lr_for_recompute = learned_reward if args_cli.recompute_reward_in_backup else None
        agent.store_and_update_batch(obs_np, action_np, rew_np, next_obs_np,
                                      done_true_termination_np.astype(np.float32),
                                      global_step, stats, learned_reward=lr_for_recompute)

        if terminal_np.any():  # episode boundary (true termination OR timeout) per env --
            # matches on_policy_runner.py's `dones > 0` gate, applied per-env instead of globally.
            for env_idx in np.nonzero(terminal_np)[0]:
                gt_ep_rewbuffer.append(float(cur_gt_ep_reward[env_idx]))
                learned_ep_rewbuffer.append(float(cur_learned_ep_reward[env_idx]))
                cur_gt_ep_reward[env_idx] = 0.0
                cur_learned_ep_reward[env_idx] = 0.0

        obs = next_obs
        global_step += num_envs

        log_block = global_step // args_cli.log_interval
        if log_block > last_log_block:
            last_log_block = log_block
            window_time = time.time() - window_start_time
            print(f"--- t={global_step}/{args_cli.total_steps} ---")
            print(f"  Mean reward (per-step): {np.mean(learned_reward_window):.2f}")
            print(f"  Mean GT reward (per-step): {np.mean(gt_reward_window):.2f}")
            if gt_ep_rewbuffer:
                print(f"  Mean episode return (learned): {np.mean(learned_ep_rewbuffer):.2f}  "
                      f"Mean episode return (GT): {np.mean(gt_ep_rewbuffer):.2f}  "
                      f"(n={len(gt_ep_rewbuffer)} episodes)")
            if "LossQ" in stats:
                print(f"  LossQ: {stats['LossQ'][-1]:.4f}  LossPi: {stats['LossPi'][-1]:.4f}")
            print(f"  Window time: {window_time:.2f}s")
            sys.stdout.flush()  # 2026-08-29: SLURM's file-redirected stdout is block-buffered,
            # unlike Isaac Sim's own Carbon/omni logger (which flushes independently) -- without
            # this, print() output can sit in an in-process buffer and never reach the log file
            # if the process exits abruptly, making a real crash look like silent, instant
            # completion. Diagnostic addition after exactly that happened on the first smoke test.
            gt_reward_window = []
            learned_reward_window = []
            window_start_time = time.time()

    print("Training complete.")
    sys.stdout.flush()
    env.close()


if __name__ == "__main__":
    import traceback
    try:
        main()
    except BaseException:
        # 2026-08-29: explicit traceback + flush + nonzero exit, added after a first smoke-test
        # run exited 0 with zero training output and no visible error -- see the flush comment
        # above for why output can otherwise go missing. Ensures a real failure is never silent.
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        simulation_app.close()
        sys.exit(1)
    simulation_app.close()
