# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""train_paws_policy.py -- PAWS: Preference Learning with Advantage-Weighted Segments
(Taranovic, Celik, Freymuth, Li, Thilges, Le, Hoang, Rayyes, Neumann; ICML 2026,
arXiv:2606.11982), added 2026-09-06 as the direct follow-up to the CPL test (Lingheng: "ok, then
try PAWS"). PAWS's own diagnosis: existing preference-based RL methods train a utility/reward
function on SEGMENT-level preferences but then use its PER-STEP output during policy optimization
(standard PPO against a learned reward net, exactly this investigation's own established pattern)
-- a genuine train/inference granularity mismatch, since the net was only ever supervised to get
SEGMENT SUMS right, never individual per-step values. PAWS's fix: never decompose the segment-level
advantage into per-step values at all -- weight the ENTIRE segment's summed policy log-likelihood
by that segment's own single scalar advantage score, via an advantage-weighted-regression-style
objective with an ADAPTIVE temperature (chosen per batch to hit a target "effective sample size",
not a fixed hyperparameter).

NEW, STANDALONE SCRIPT -- does not import, modify, or monkey-patch train.py, on_policy_runner.py,
ppo.py, learned_reward.py, online_reward_update.py, or train_cpl_policy.py. Shares train_cpl_
policy.py's proven boilerplate pattern (AppLauncher/env setup for architecture-correct ActorCritic
construction via OnPolicyRunner, inline checkpoint-dict construction instead of runner.save() --
see that script's own comment for why) but is a fully independent file; the two scripts share no
imports of each other.

Verified against the authors' own public config (github.com/ataranovic/PAWS):
  - config/reward_cfg/paws_advantage_mlp.yaml -- the advantage function A_phi is trained via the
    EXACT SAME Bradley-Terry segment-preference loss this investigation's train_reward_model.py
    already implements, just semantically relabeled "advantage" instead of "reward". This script
    does NOT retrain it -- see job_lift_paws_advantage_net.sh, which reuses train_reward_model.py
    completely unchanged (architecture=baseline i.e. unbounded MLPRewardNet, matching their
    final_activation="").
  - config/agent_cfg/paws_policy.yaml -- policy loss: loss_fn.name="eff_sample",
    target_eff=0.1. Interpreted (the paper's own text gives the loss FORM, the config gives this
    ONE numeric knob) as: per training batch of segments, adaptively choose a softmax temperature
    lambda such that the resulting importance weights' effective sample size (1/sum(w^2)) equals
    target_eff * batch_size, then use those (already-normalized, "norm_weights: True") weights to
    scale each segment's total policy log-likelihood in a weighted-MLE loss:
        L(theta) = - sum_segments w_segment * sum_t log pi_theta(a_t | s_t)
    "normalize_reward: True" is interpreted as z-score normalizing each batch's raw advantage
    scores before the temperature search (a reasonable, standard choice for exponential-tilting
    numerical stability; the config does not specify batch-level vs. dataset-level, so this is a
    documented interpretation, not a verified-exact replication of an internal implementation
    detail the config alone doesn't pin down).

Usage:
    ./isaaclab.sh -p train_paws_policy.py --task Isaac-Lift-Cube-Franka-Absolute-DR-TableFix-v0 \\
        --num_envs 2 --meta_train /path/train_meta.json --meta_test /path/test_meta.json \\
        --advantage_experiment_dir /path/paws_advantage_net/bounded_..._privileged \\
        --output_dir /path/paws_policy_out --target_eff 0.1 --fragment_length 64 \\
        --total_segments 40000 --epochs 100 --seed 42
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "pref_learning"))

parser = argparse.ArgumentParser(description="Train a policy via PAWS (segment-level advantage-weighted regression).")
parser.add_argument("--num_envs", type=int, default=2, help="Envs to instantiate (architecture construction only -- no rollouts are run).")
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--meta_train", type=str, required=True)
parser.add_argument("--advantage_experiment_dir", type=str, required=True, help="Frozen advantage net (config.json + reward_model.pt), trained via job_lift_paws_advantage_net.sh.")
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--target_eff", type=float, default=0.1, help="Target effective-sample-size fraction -- matches the reference config exactly (config/agent_cfg/paws_policy.yaml).")
parser.add_argument("--fragment_length", type=int, default=64, help="Matches the reference config's pref_segment_size=64.")
parser.add_argument("--total_segments", type=int, default=40000)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=256, help="Smaller than the reference's sub_batch_size=4096 -- this investigation's offline pool (256 trajectories) is far smaller than PAWS's own MetaWorld-scale datasets.")
parser.add_argument("--lr", type=float, default=3e-4, help="Matches the reference config's actor_lr.")
parser.add_argument("--val_frac", type=float, default=0.1)
parser.add_argument("--patience", type=int, default=8)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import json

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from get_trajectories import load_from_meta_dataset, precompute_features  # noqa: E402
from train_reward_model import ExperimentConfig, create_reward_net, generate_all_comparisons  # noqa: E402

from isaaclab.envs import DirectMARLEnv, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import isaaclab_tasks  # noqa: E402,F401
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402


def _load_advantage_net(experiment_dir: str, device: str):
    with open(os.path.join(experiment_dir, "config.json")) as f:
        config = ExperimentConfig.from_dict(json.load(f))
    checkpoint = torch.load(os.path.join(experiment_dir, "reward_model.pt"), map_location="cpu")
    net = create_reward_net(config, checkpoint["obs_dim"], checkpoint["action_dim"], device)
    net.load_state_dict(checkpoint["state_dict"])
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def _segment_advantage(advantage_net, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
    """Sum of the FROZEN advantage net's per-step output over the segment -- A_phi(tau) =
    sum_t A_phi(s_t, a_t), exactly the paper's Eq. for the segment-level advantage. No grad --
    advantage_net is never updated in this script."""
    batch, T = obs.shape[0], obs.shape[1]
    with torch.no_grad():
        per_step = advantage_net(obs.reshape(batch * T, -1), act.reshape(batch * T, -1), None, None)
    return per_step.reshape(batch, T).sum(dim=1)


def _policy_segment_log_prob(policy, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
    """sum_t log pi_theta(a_t|s_t) over the segment, via the SAME ActorCritic.act()/
    get_actions_log_prob() pattern verified working in train_cpl_policy.py."""
    batch, T = obs.shape[0], obs.shape[1]
    obs_flat = obs.reshape(batch * T, -1)
    act_flat = act.reshape(batch * T, -1)
    policy.act(obs_flat)
    log_prob_flat = policy.get_actions_log_prob(act_flat)
    return log_prob_flat.reshape(batch, T).sum(dim=1)


def _solve_temperature_and_weights(advantage: torch.Tensor, target_eff: float, iters: int = 40):
    """Bisects for a temperature lambda such that softmax(advantage/lambda) has effective sample
    size (1/sum(w^2)) approximately target_eff * batch_size -- matches config/agent_cfg/
    paws_policy.yaml's loss_fn.params.target_eff exactly. ESS/N is monotonically increasing in
    lambda (lambda->0 collapses weight onto the single highest-advantage segment, ESS->1;
    lambda->inf makes weights uniform, ESS->N), so bisection over lambda is well posed. Subtracts
    the max advantage before exponentiating for numerical stability -- a scale shift that leaves
    the ESS ratio unchanged (softmax is shift-invariant).
    """
    n = advantage.shape[0]
    target_ess = target_eff * n
    a_shift = advantage - advantage.max()

    def ess_at(log_lam: torch.Tensor) -> torch.Tensor:
        lam = torch.exp(log_lam)
        w = F.softmax(a_shift / lam, dim=0)
        return 1.0 / (w.pow(2).sum() + 1e-12)

    log_lo = torch.tensor(-10.0, device=advantage.device)
    log_hi = torch.tensor(10.0, device=advantage.device)
    for _ in range(iters):
        log_mid = (log_lo + log_hi) / 2
        if ess_at(log_mid) < target_ess:
            log_lo = log_mid
        else:
            log_hi = log_mid
    lam = torch.exp((log_lo + log_hi) / 2)
    weights = F.softmax(a_shift / lam, dim=0)  # already sums to 1 -- "norm_weights: True"
    return weights, lam


def _paws_loss(policy, advantage_net, obs: torch.Tensor, act: torch.Tensor, target_eff: float):
    advantage = _segment_advantage(advantage_net, obs, act)
    advantage_norm = (advantage - advantage.mean()) / (advantage.std() + 1e-8)  # "normalize_reward: True"
    weights, lam = _solve_temperature_and_weights(advantage_norm, target_eff)
    segment_log_prob = _policy_segment_log_prob(policy, obs, act)
    loss = -(weights * segment_log_prob).sum()
    with torch.no_grad():
        ess_frac = (1.0 / weights.pow(2).sum()).item() / obs.shape[0]
    return loss, ess_frac, lam.item()


class _SegmentDataset(torch.utils.data.Dataset):
    def __init__(self, obs_list, act_list):
        self.obs_list, self.act_list = obs_list, act_list

    def __len__(self):
        return len(self.obs_list)

    def __getitem__(self, idx):
        return torch.from_numpy(self.obs_list[idx]), torch.from_numpy(self.act_list[idx])


def _make_segment_pool(trajectories, n_segments: int, fragment_length: int, seed: int):
    """Reuses generate_all_comparisons() purely as a fragment sampler -- PAWS's actor-training
    stage scores individual segments, not comparison pairs, so both sides of each generated
    "comparison" are kept as independent segments and the label is discarded. Avoids writing a new
    fragment-sampling function; RandomFragmenter (already validated elsewhere in this codebase) is
    what actually does the sampling underneath generate_all_comparisons().
    """
    obs_a, act_a, obs_b, act_b, _labels, _na, _nb = generate_all_comparisons(
        trajectories, n_comparisons=(n_segments + 1) // 2, fragment_length=fragment_length, seed=seed,
    )
    return obs_a + obs_b, act_a + act_b


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    agent_cfg.seed = args_cli.seed
    agent_cfg.device = args_cli.device if args_cli.device is not None else agent_cfg.device
    agent_cfg.use_learned_reward = False
    # Same load-bearing assumption as train_cpl_policy.py -- see that script's comment: for this
    # DR-TableFix Lift task variant, the 27-dim "privileged" observation is BY DESIGN identical to
    # the real actor's own live policy observation.

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    device = agent_cfg.device
    os.makedirs(args_cli.output_dir, exist_ok=True)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=args_cli.output_dir, device=device)
    policy = runner.alg.policy
    print(f"[PAWS] Constructed policy via OnPolicyRunner (task={args_cli.task}, device={device}).")

    advantage_net = _load_advantage_net(args_cli.advantage_experiment_dir, device)
    print(f"[PAWS] Loaded frozen advantage net from {args_cli.advantage_experiment_dir}")

    print(f"[PAWS] Loading offline trajectories from {args_cli.meta_train}")
    raw_train = load_from_meta_dataset(args_cli.meta_train, obs_mode="privileged", task_family="lift")
    train_trajs, _ = precompute_features(raw_train, obs_mode="privileged")
    print(f"[PAWS] {len(train_trajs)} trajectories loaded")

    print(f"[PAWS] Sampling {args_cli.total_segments} segments at fragment_length={args_cli.fragment_length}")
    obs_list, act_list = _make_segment_pool(train_trajs, args_cli.total_segments, args_cli.fragment_length, args_cli.seed)
    n = len(obs_list)
    rng = np.random.default_rng(args_cli.seed)
    perm = rng.permutation(n)
    n_val = int(n * args_cli.val_frac)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    train_ds = _SegmentDataset([obs_list[i] for i in train_idx], [act_list[i] for i in train_idx])
    val_ds = _SegmentDataset([obs_list[i] for i in val_idx], [act_list[i] for i in val_idx])
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args_cli.batch_size, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args_cli.batch_size, shuffle=False, drop_last=True)
    print(f"[PAWS] {len(train_ds)} train segments, {len(val_ds)} val segments")

    optimizer = torch.optim.Adam(policy.parameters(), lr=args_cli.lr, betas=(0.9, 0.99))
    best_val_loss, best_state, epochs_no_improve = float("inf"), None, 0

    for epoch in range(args_cli.epochs):
        policy.train()
        train_losses, train_ess = [], []
        for obs, act in train_loader:
            obs, act = obs.to(device).float(), act.to(device).float()
            loss, ess_frac, _lam = _paws_loss(policy, advantage_net, obs, act, args_cli.target_eff)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())
            train_ess.append(ess_frac)

        policy.eval()
        val_losses, val_ess = [], []
        with torch.no_grad():
            for obs, act in val_loader:
                obs, act = obs.to(device).float(), act.to(device).float()
                loss, ess_frac, _lam = _paws_loss(policy, advantage_net, obs, act, args_cli.target_eff)
                val_losses.append(loss.item())
                val_ess.append(ess_frac)
        val_loss = float(np.mean(val_losses))
        print(f"[PAWS] epoch {epoch+1}/{args_cli.epochs} | train_loss={np.mean(train_losses):.4f} "
              f"train_ess_frac={np.mean(train_ess):.3f} | val_loss={val_loss:.4f} val_ess_frac={np.mean(val_ess):.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in policy.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args_cli.patience:
                print(f"[PAWS] early stopping at epoch {epoch+1} (best val_loss={best_val_loss:.4f})")
                break

    if best_state is not None:
        policy.load_state_dict(best_state)
    policy.eval()
    print(f"[PAWS] training complete. best val_loss: {best_val_loss:.4f}")

    # Not calling runner.save() -- same reason as train_cpl_policy.py: it references
    # self.logger_type, only ever set inside learn(), which this script never calls.
    ckpt_path = os.path.join(args_cli.output_dir, "model_0.pt")
    saved_dict = {
        "model_state_dict": runner.alg.policy.state_dict(),
        "optimizer_state_dict": runner.alg.optimizer.state_dict(),
        "iter": 0,
        "infos": None,
    }
    if runner.empirical_normalization:
        saved_dict["obs_norm_state_dict"] = runner.obs_normalizer.state_dict()
        saved_dict["privileged_obs_norm_state_dict"] = runner.privileged_obs_normalizer.state_dict()
    torch.save(saved_dict, ckpt_path)
    with open(os.path.join(args_cli.output_dir, "paws_config.json"), "w") as f:
        json.dump(vars(args_cli), f, indent=2, default=str)
    print(f"[PAWS] saved checkpoint (rsl_rl-compatible, loadable via the standard "
          f"agent.resume=true mechanism) to {ckpt_path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
