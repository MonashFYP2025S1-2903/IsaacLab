# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Trains PL-POMDP's own (ported) PPO agent against a learned reward model on an IsaacLab task,
run as-is with no KL-anchor-to-reference mechanism -- a second, independent no-reference cross-
check of the RL side, following the reward-model-training-core cross-check (which held the RL side
fixed and only varied the reward-model trainer). See the approved plan
(pref_learning-adjacent: C:\\Users\\MEN119\\.claude\\plans\\vivid-fluttering-hare.md, 2026-08-28)
for full rationale.

Deliberately does NOT go through rsl_rl's OnPolicyRunner/on_policy_runner.py -- that's the thing
being cross-checked against, an independent implementation. Does reuse:
- RslRlVecEnvWrapper (isaaclab_rl.rsl_rl) purely as a convenient, already-tested tensor-interface
  adapter around the underlying Gymnasium ManagerBasedRLEnv (obs flattening, action clipping) --
  orthogonal to which PPO algorithm consumes the data, not "using rsl_rl's PPO".
- LearnedRewardWrapper (rsl_rl.rsl_rl.runners.learned_reward) UNCHANGED, to load the already
  production-scale-trained PL-POMDP Cartpole reward-model checkpoint -- already proven working,
  env-agnostic.
- pref_learning/ppo_plpomdp.py's ported PPO class for the actual policy-training algorithm.

Faithful-port note: no observation normalization is applied (PL-POMDP's own ppo.py/core.py has no
normalizer -- it feeds raw observations directly into the actor-critic), unlike
on_policy_runner.py's EmpiricalNormalization. A deliberate divergence from this pipeline's own
rsl_rl stack, consistent with testing PL-POMDP's code as its author actually wrote it.

Usage:
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train_ppo_plpomdp.py \\
        --task Isaac-Cartpole-PrefLearning-v0 --headless \\
        --rm_checkpoint /path/to/plpomdp_mlp_.../reward_model.pt \\
        --epochs 3 --steps_per_epoch 4000
"""

"""Launch Isaac Sim Simulator first."""

import argparse
from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Train PL-POMDP's own PPO against a learned reward, no KL anchor.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=1, help="num_envs=1 uses the original, "
                     "already-validated single-env PPO/PPOBuffer path unchanged. num_envs>1 uses "
                     "the vectorized VecPPO/VecPPOBuffer path (added 2026-08-28, see "
                     "pref_learning/ppo_plpomdp.py) instead.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--rm_checkpoint", type=str, default="", help="PL-POMDP-trained reward model "
                     "checkpoint. Required unless --train_on_gt_reward is set.")
parser.add_argument("--reward_weight", type=float, default=0.1, help="Matches every other run this "
                     "session's agent.learned_reward.reward_weight.")
parser.add_argument("--train_on_gt_reward", action="store_true", default=False,
                     help="Train directly on the environment's own ground-truth reward instead of "
                     "a learned reward model -- bypasses --rm_checkpoint/LearnedRewardWrapper "
                     "entirely. Added 2026-08-28 as a correctness check for VecPPOBuffer/VecPPO "
                     "(and PPOBuffer/PPO): a subtly wrong GAE/episode-boundary implementation "
                     "would very likely fail to solve even Cartpole on the real reward, a much "
                     "sharper test than \"didn't crash against the learned reward's near-zero "
                     "signal\". Expect convergence toward the established GT-oracle ceiling "
                     "(~4.9-4.95 final GT reward on Cartpole) if the implementation is correct.")
parser.add_argument("--strip_last_action_from_actor_obs", action="store_true", default=False,
                     help="Arm P1, ported from rsl_rl (added 2026-08-31). See "
                     "pref_learning/ppo_plpomdp.py's MLPActorCritic docstring comment for the "
                     "full rationale -- IsaacLab's PolicyCfg obs group embeds mdp.last_action as "
                     "its trailing field for every task, present in the SAME obs this script "
                     "receives.")
parser.add_argument("--strip_last_action_from_critic_obs", action="store_true", default=False,
                     help="Arm C1, ported from rsl_rl (added 2026-08-31). Independent of "
                     "--strip_last_action_from_actor_obs -- combine both for the P1+C1 test.")
parser.add_argument("--epochs", type=int, required=True)
parser.add_argument("--steps_per_epoch", type=int, default=4000, help="PL-POMDP's own default. "
                     "Only used at num_envs=1.")
parser.add_argument("--steps_per_env", type=int, default=16, help="Vectorized rollout length per "
                     "env before each update -- the num_envs>1 analogue of --steps_per_epoch, kept "
                     "separate since steps_per_epoch=4000 at e.g. num_envs=4096 would mean ~16.4M "
                     "samples per update, far more than intended. Default 16 matches rsl_rl's own "
                     "num_steps_per_env for Cartpole, for the most direct comparability. Only used "
                     "at num_envs>1.")
parser.add_argument("--seed", type=int, default=42)
cli_args.add_rsl_rl_args(parser)  # only used here to obtain agent_cfg.clip_actions via the
# registered CartpolePrefLearningPPORunnerCfg -- the rest of agent_cfg (its own PPO
# hyperparameters, use_learned_reward fields, etc.) is intentionally unused; ppo_plpomdp.PPO is
# configured directly from this script's own --epochs/--steps_per_epoch/PL-POMDP-default args.
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
sys.path.insert(0, os.path.join(_REPO_ROOT, "rsl_rl"))  # outer dir, for `rsl_rl.modules` below

import ppo_plpomdp
from runners.learned_reward import LearnedRewardCfg, LearnedRewardWrapper
from rsl_rl.modules import EmpiricalNormalization  # added 2026-08-28: PL-POMDP's own ppo.py/
# core.py has no observation normalizer -- it feeds raw observations straight into the
# actor-critic. Confirmed via a direct GT-reward test (bypassing the learned reward model
# entirely) that this was NOT a benign "faithful port" divergence: without normalization, VecPPO/
# PPO fail to solve even Cartpole on the real reward (~0.02 final GT reward vs. the established
# ~4.94 ceiling every OTHER baseline in this investigation reaches -- all of which use this same
# EmpiricalNormalization via on_policy_runner.py). Reused here as infrastructure/data-plumbing,
# the same category of reuse as RslRlVecEnvWrapper -- not "using rsl_rl's PPO algorithm", just its
# (algorithm-agnostic) running-mean/std observation preprocessor.

import isaaclab_tasks  # noqa: F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import parse_env_cfg


def run_single_env(env, learned_reward, num_obs, num_actions, device, obs_normalizer):
    """num_envs=1 path -- the original single-env PPO/PPOBuffer training loop, now with
    obs_normalizer applied to whatever the POLICY sees (added 2026-08-28, see the import comment
    above for why). The reward net still gets RAW, unnormalized obs -- it was trained on raw obs
    throughout this whole investigation (generate_all_comparisons()'s frag.obs[:-1], no
    normalizer anywhere in the trajectory-collection pipeline), so normalizing what's fed to
    learned_reward.predict() would be a real train/inference mismatch, not a fix. obs_normalizer
    is called exactly once per newly-observed state (cached and reused the next iteration as
    `obs_norm`) to avoid double-counting the same observation into its running statistics."""
    obs_space = spaces.Box(-np.inf, np.inf, shape=(num_obs,), dtype=np.float32)
    act_space = spaces.Box(-1.0, 1.0, shape=(num_actions,), dtype=np.float32)  # bounds unused by
    # ppo_plpomdp (MLPGaussianActor only reads .shape[0]); real clipping is handled by
    # RslRlVecEnvWrapper's clip_actions, matching every other run this session.

    agent = ppo_plpomdp.PPO(
        obs_space, act_space, hidden_sizes=(64, 64),
        gamma=0.99, clip_ratio=0.2, steps_per_epoch=args_cli.steps_per_epoch,
        train_pi_iters=80, train_v_iters=80, target_kl=0.01,
        pi_lr=3e-4, vf_lr=1e-3, device=device,
        strip_last_action_from_actor_obs=args_cli.strip_last_action_from_actor_obs,
        strip_last_action_from_critic_obs=args_cli.strip_last_action_from_critic_obs,
    )

    print(f"epochs={args_cli.epochs} steps_per_epoch={args_cli.steps_per_epoch} "
          f"total_steps={args_cli.epochs * args_cli.steps_per_epoch}")

    obs, extras = env.get_observations()  # raw
    obs_norm = obs_normalizer(obs)
    total_steps = args_cli.epochs * args_cli.steps_per_epoch
    gt_reward_window = []
    learned_reward_window = []
    # Per-EPISODE cumulative return, matching on_policy_runner.py's gt_rewbuffer/rewbuffer
    # convention (cur_*_reward_sum accumulates every step, pushed to a maxlen=100 deque and reset
    # only when the episode ends) -- NOT the same quantity as gt_reward_window's raw per-step
    # average above. Added 2026-08-28 after discovering the two were being compared as if
    # equivalent (established baselines' ~4.94 figure is a per-episode SUM, not a per-step mean).
    gt_ep_rewbuffer = deque(maxlen=100)
    learned_ep_rewbuffer = deque(maxlen=100)
    cur_gt_ep_reward = 0.0
    cur_learned_ep_reward = 0.0
    stats = {}
    epoch_start_time = time.time()

    for t in range(total_steps):
        obs_np = obs_norm[0].detach().cpu().numpy()
        action_np = agent.select_action(obs_np)
        action_t = torch.as_tensor(action_np, dtype=torch.float32, device=device).unsqueeze(0)

        obs_before_step = obs  # raw, for the reward net
        next_obs, gt_rewards, dones, infos = env.step(action_t)
        next_obs, gt_rewards, dones = next_obs.to(device), gt_rewards.to(device), dones.to(device)
        next_obs_norm = obs_normalizer(next_obs)  # one call per new obs -- see docstring

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

        next_obs_np = next_obs_norm[0].detach().cpu().numpy()
        agent.finish_step(rew, terminal, done_true_termination, next_obs_np, stats)

        if terminal:  # episode boundary (true termination OR timeout) -- push cumulative
            # return, reset accumulator. Matches on_policy_runner.py's `dones > 0` gate, which
            # does not distinguish termination from timeout for this bookkeeping.
            gt_ep_rewbuffer.append(cur_gt_ep_reward)
            learned_ep_rewbuffer.append(cur_learned_ep_reward)
            cur_gt_ep_reward = 0.0
            cur_learned_ep_reward = 0.0

        obs = next_obs
        obs_norm = next_obs_norm  # reuse -- already computed above, no redundant normalizer call

        if (t + 1) % args_cli.steps_per_epoch == 0:
            epoch = (t + 1) // args_cli.steps_per_epoch
            epoch_time = time.time() - epoch_start_time
            print(f"--- Epoch {epoch}/{args_cli.epochs} (t={t + 1}) ---")
            print(f"  Mean reward (per-step): {np.mean(learned_reward_window):.2f}")
            print(f"  Mean GT reward (per-step): {np.mean(gt_reward_window):.2f}")
            if gt_ep_rewbuffer:
                print(f"  Mean episode return (learned): {np.mean(learned_ep_rewbuffer):.2f}  "
                      f"Mean episode return (GT): {np.mean(gt_ep_rewbuffer):.2f}  "
                      f"(n={len(gt_ep_rewbuffer)} episodes)")
            if "LossPi" in stats:
                print(f"  LossPi: {stats['LossPi'][-1]:.4f}  LossV: {stats['LossV'][-1]:.4f}  "
                      f"KL: {stats['KL'][-1]:.4f}  Entropy: {stats['Entropy'][-1]:.4f}  "
                      f"ClipFrac: {stats['ClipFrac'][-1]:.4f}  StopIter: {stats['StopIter'][-1]}")
            print(f"  Epoch time: {epoch_time:.2f}s")
            gt_reward_window = []
            learned_reward_window = []
            epoch_start_time = time.time()


def run_vectorized(env, learned_reward, num_obs, num_actions, device, num_envs, obs_normalizer):
    """num_envs>1 path -- added 2026-08-28. Uses VecPPO/VecPPOBuffer (pref_learning/ppo_plpomdp.py)
    instead of PPO/PPOBuffer; batches action selection and transition storage across all envs per
    step instead of one env at a time. See the approved plan for the full design rationale.
    obs_normalizer applied the same way as run_single_env (see that function's docstring) -- fed
    to the policy only, raw obs kept for the reward net, one normalizer call per new observation.
    """
    obs_space = spaces.Box(-np.inf, np.inf, shape=(num_obs,), dtype=np.float32)
    act_space = spaces.Box(-1.0, 1.0, shape=(num_actions,), dtype=np.float32)

    agent = ppo_plpomdp.VecPPO(
        obs_space, act_space, num_envs, hidden_sizes=(64, 64),
        gamma=0.99, clip_ratio=0.2, steps_per_env=args_cli.steps_per_env,
        train_pi_iters=80, train_v_iters=80, target_kl=0.01,
        pi_lr=3e-4, vf_lr=1e-3, device=device,
        strip_last_action_from_actor_obs=args_cli.strip_last_action_from_actor_obs,
        strip_last_action_from_critic_obs=args_cli.strip_last_action_from_critic_obs,
    )

    total_steps_per_env = args_cli.epochs * args_cli.steps_per_env
    total_env_steps = total_steps_per_env * num_envs
    print(f"num_envs={num_envs} epochs={args_cli.epochs} steps_per_env={args_cli.steps_per_env} "
          f"total_steps_per_env={total_steps_per_env} total_env_steps={total_env_steps}")

    obs, extras = env.get_observations()  # raw
    obs_norm = obs_normalizer(obs)
    gt_reward_window = []
    learned_reward_window = []
    # Per-EPISODE cumulative return across all envs, matching on_policy_runner.py's
    # gt_rewbuffer/rewbuffer convention -- see run_single_env's identical comment for the full
    # rationale. Per-env accumulators, pushed/reset independently as each env's episode ends.
    gt_ep_rewbuffer = deque(maxlen=100)
    learned_ep_rewbuffer = deque(maxlen=100)
    cur_gt_ep_reward = np.zeros(num_envs, dtype=np.float64)
    cur_learned_ep_reward = np.zeros(num_envs, dtype=np.float64)
    stats = {}
    epoch_start_time = time.time()

    for t in range(total_steps_per_env):
        obs_np = obs_norm.detach().cpu().numpy()
        action_np = agent.select_actions(obs_np)
        action_t = torch.as_tensor(action_np, dtype=torch.float32, device=device)

        obs_before_step = obs  # raw, for the reward net
        next_obs, gt_rewards, dones, infos = env.step(action_t)
        next_obs, gt_rewards, dones = next_obs.to(device), gt_rewards.to(device), dones.to(device)
        next_obs_norm = obs_normalizer(next_obs)  # one call per new obs -- see docstring

        if learned_reward is not None:
            learned_r = learned_reward.predict(obs_before_step, action_t, next_obs=next_obs)
            rewards_np = learned_r.detach().cpu().numpy()
        else:
            rewards_np = gt_rewards.detach().cpu().numpy()  # --train_on_gt_reward

        gt_reward_window.append(gt_rewards.mean().item())
        learned_reward_window.append(float(rewards_np.mean()))
        gt_rewards_np = gt_rewards.detach().cpu().numpy()
        cur_gt_ep_reward += gt_rewards_np
        cur_learned_ep_reward += rewards_np

        time_outs = infos.get("time_outs", torch.zeros_like(dones))
        dones_np = dones.detach().cpu().numpy()
        time_outs_np = time_outs.detach().cpu().numpy()

        next_obs_np = next_obs_norm.detach().cpu().numpy()
        agent.finish_step_batch(rewards_np, dones_np, time_outs_np, next_obs_np, stats)

        done_env_ids = np.nonzero(dones_np > 0)[0]  # matches on_policy_runner.py's
        # `(dones > 0).nonzero()` -- any episode end (termination or timeout) pushes+resets.
        for i in done_env_ids:
            gt_ep_rewbuffer.append(float(cur_gt_ep_reward[i]))
            learned_ep_rewbuffer.append(float(cur_learned_ep_reward[i]))
        if len(done_env_ids) > 0:
            cur_gt_ep_reward[done_env_ids] = 0.0
            cur_learned_ep_reward[done_env_ids] = 0.0

        obs = next_obs
        obs_norm = next_obs_norm  # reuse -- already computed above, no redundant normalizer call

        if (t + 1) % args_cli.steps_per_env == 0:
            epoch = (t + 1) // args_cli.steps_per_env
            epoch_time = time.time() - epoch_start_time
            print(f"--- Epoch {epoch}/{args_cli.epochs} (t={(t + 1) * num_envs}) ---")
            print(f"  Mean reward (per-step): {np.mean(learned_reward_window):.2f}")
            print(f"  Mean GT reward (per-step): {np.mean(gt_reward_window):.2f}")
            if gt_ep_rewbuffer:
                print(f"  Mean episode return (learned): {np.mean(learned_ep_rewbuffer):.2f}  "
                      f"Mean episode return (GT): {np.mean(gt_ep_rewbuffer):.2f}  "
                      f"(n={len(gt_ep_rewbuffer)} episodes)")
            if "LossPi" in stats:
                print(f"  LossPi: {stats['LossPi'][-1]:.4f}  LossV: {stats['LossV'][-1]:.4f}  "
                      f"KL: {stats['KL'][-1]:.4f}  Entropy: {stats['Entropy'][-1]:.4f}  "
                      f"ClipFrac: {stats['ClipFrac'][-1]:.4f}  StopIter: {stats['StopIter'][-1]}")
            print(f"  Epoch time: {epoch_time:.2f}s")
            gt_reward_window = []
            learned_reward_window = []
            epoch_start_time = time.time()


def main():
    if not args_cli.train_on_gt_reward:
        assert args_cli.rm_checkpoint, "--rm_checkpoint is required unless --train_on_gt_reward is set."

    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)

    task_name = args_cli.task.split(":")[-1]
    agent_cfg = cli_args.parse_rsl_rl_cfg(task_name, args_cli)  # only agent_cfg.clip_actions used
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

    if args_cli.train_on_gt_reward:
        learned_reward = None
        print("Training directly on GT reward (--train_on_gt_reward) -- no reward model loaded.")
    else:
        lr_cfg = LearnedRewardCfg(rm_checkpoint=args_cli.rm_checkpoint, reward_weight=args_cli.reward_weight)
        learned_reward = LearnedRewardWrapper(num_obs, num_actions, device=device, config=lr_cfg)

    print(f"num_obs={num_obs} num_actions={num_actions} device={device} num_envs={args_cli.num_envs}")

    obs_normalizer = EmpiricalNormalization(shape=[num_obs], until=1.0e8).to(device)

    if args_cli.num_envs == 1:
        run_single_env(env, learned_reward, num_obs, num_actions, device, obs_normalizer)
    else:
        run_vectorized(env, learned_reward, num_obs, num_actions, device, args_cli.num_envs, obs_normalizer)

    print("Training complete.")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
