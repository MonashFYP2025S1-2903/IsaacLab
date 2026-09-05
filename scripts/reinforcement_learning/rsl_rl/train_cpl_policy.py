# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""train_cpl_policy.py -- Contrastive Preference Learning (Hejna, Rafailov, Sikchi, Finn, Niekum,
Knox, Sadigh; ICLR 2024, arXiv:2310.13639), added 2026-09-06 in response to Lingheng's question:
"do you think we should also consider using preference directly update the policy? if the reward
does not generalize well, probably just use it to update policy." Trains a policy DIRECTLY from
offline preference comparisons -- no reward model is ever fit, sidestepping the reward-net-
generalization axis of this whole investigation's failure mode entirely (though NOT the underlying
preference-data-coverage requirement -- see the honest assessment in
FYP2025S1-2903_ablation_experiment_index.md's "Literature: direct-preference-to-policy methods"
section before reading this as a magic fix).

NEW, STANDALONE SCRIPT -- does not import, modify, or monkey-patch train.py, on_policy_runner.py,
ppo.py, learned_reward.py, or online_reward_update.py. Every piece of existing training/RL-loop
code this investigation depends on is completely untouched by this file's existence. Only reuses
already-existing, unmodified functions: rsl_rl.runners.OnPolicyRunner (for constructing an
architecture-correct, checkpoint-compatible ActorCritic via the SAME construction path train.py/
play.py use, and for reward_calibration-format-safe final checkpoint saving via runner.save()),
get_trajectories.load_from_meta_dataset/precompute_features, and
train_reward_model.generate_all_comparisons/PreferencePairDataset (all already validated
elsewhere in this codebase, untouched here).

Algorithm (verified against the paper's Eq. 5/6 and the authors' own reference implementation,
github.com/jhejna/cpl, configs/mw_state_dense/cpl.yaml -- a MetaWorld state-based manipulation
config, the closest match to this investigation's own privileged/state-based Lift setup):
    For each labeled fragment pair (frag_A, frag_B) of length T (fragment_length, CPL's own
    default segment_length=64, NOT this investigation's usual fragment_length=1 -- CPL's
    "advantage" substitution fundamentally needs a real multi-step segment, a single-timestep
    fragment collapses its own core idea):
        score(frag) = alpha * sum_{t=0}^{T-1} gamma^t * log pi_theta(a_t | s_t)
    which substitutes for a fitted reward net's segment-summed score entirely -- log pi_theta
    comes directly from ActorCritic.act(obs) + get_actions_log_prob(act), already-existing,
    unmodified rsl_rl methods.
    Conservative bias (Eq. 6, contrastive_bias=0.5 in the reference config): with pos/neg
    resolved from the known preference label (pos = preferred fragment, neg = dispreferred),
        loss = -log sigmoid( score(pos) - contrastive_bias * score(neg) )
    Ties (label==0.5, ~1-in-200000 in this investigation's own comparison sets) are dropped
    rather than specially handled -- their frequency is negligible and the paper's own setup
    does not define a tie convention.

Usage:
    ./isaaclab.sh -p train_cpl_policy.py --task Isaac-Lift-Cube-Franka-Absolute-DR-TableFix-v0 \\
        --num_envs 2 --meta_train /path/train_meta.json --meta_test /path/test_meta.json \\
        --output_dir /path/cpl_policy_out --alpha 0.1 --contrastive_bias 0.5 \\
        --fragment_length 64 --total_comparisons 20000 --epochs 20 --seed 42
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "pref_learning"))

parser = argparse.ArgumentParser(description="Train a policy directly from preferences via CPL (no reward model).")
parser.add_argument("--num_envs", type=int, default=2, help="Envs to instantiate (architecture construction only -- no rollouts are run).")
parser.add_argument("--task", type=str, default=None, help="Name of the task (for architecture/obs-space matching only).")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--meta_train", type=str, required=True)
parser.add_argument("--meta_test", type=str, default="", help="Optional held-out set for a final CPL-logit preference-accuracy readout.")
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--alpha", type=float, default=0.1, help="Temperature -- matches the CPL paper's own reference config (configs/mw_state_dense/cpl.yaml).")
parser.add_argument("--contrastive_bias", type=float, default=0.5, help="Conservative bias lambda on the dispreferred segment's score (paper Eq. 6) -- matches the reference config.")
parser.add_argument("--gamma", type=float, default=0.99, help="Discount applied within each fragment's score sum.")
parser.add_argument("--fragment_length", type=int, default=64, help="CPL's own default (segment_length=64 in the reference config) -- deliberately NOT this investigation's usual fragment_length=1, since CPL's advantage substitution needs a real multi-step segment.")
parser.add_argument("--total_comparisons", type=int, default=20000)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--lr", type=float, default=1e-4, help="Matches the CPL reference config's optim_kwargs.lr.")
parser.add_argument("--val_frac", type=float, default=0.1)
parser.add_argument("--patience", type=int, default=5)
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
from train_reward_model import generate_all_comparisons, PreferencePairDataset  # noqa: E402

from isaaclab.envs import DirectMARLEnv, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import isaaclab_tasks  # noqa: E402,F401
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402


def _fragment_score(policy, obs: torch.Tensor, act: torch.Tensor, alpha: float, gamma: float) -> torch.Tensor:
    """alpha * sum_t gamma^t * log pi_theta(a_t|s_t), batched over (batch, T, ...) tensors.
    Uses ActorCritic.act() to set the internal distribution from the CURRENT policy parameters,
    then get_actions_log_prob() for the actual (dataset-logged) action -- both pre-existing,
    unmodified rsl_rl.modules.ActorCritic methods (act() sets self.distribution; get_actions_
    log_prob() returns log pi(a|s) for an arbitrary action, not necessarily freshly sampled).
    """
    batch, T = obs.shape[0], obs.shape[1]
    obs_flat = obs.reshape(batch * T, -1)
    act_flat = act.reshape(batch * T, -1)
    policy.act(obs_flat)  # sets policy.distribution over the flattened (batch*T) rows
    log_prob_flat = policy.get_actions_log_prob(act_flat)  # (batch*T,)
    log_prob = log_prob_flat.reshape(batch, T)
    gamma_powers = (gamma ** torch.arange(T, device=obs.device, dtype=obs.dtype)).unsqueeze(0)  # (1, T)
    return alpha * (gamma_powers * log_prob).sum(dim=1)  # (batch,)


def _cpl_loss(policy, batch, device: str, alpha: float, bias: float, gamma: float):
    obs_a, act_a, _next_a, obs_b, act_b, _next_b, labels, _reason = batch
    obs_a, act_a = obs_a.to(device).float(), act_a.to(device).float()
    obs_b, act_b = obs_b.to(device).float(), act_b.to(device).float()
    labels = labels.to(device).float()

    keep = labels != 0.5  # drop ties -- see module docstring
    if keep.sum() == 0:
        return None, 0
    obs_a, act_a, obs_b, act_b, labels = obs_a[keep], act_a[keep], obs_b[keep], act_b[keep], labels[keep]

    score_a = _fragment_score(policy, obs_a, act_a, alpha, gamma)
    score_b = _fragment_score(policy, obs_b, act_b, alpha, gamma)

    a_preferred = labels > 0.5
    pos = torch.where(a_preferred, score_a, score_b)
    neg = torch.where(a_preferred, score_b, score_a)
    logits = pos - bias * neg
    loss = -F.logsigmoid(logits).mean()
    with torch.no_grad():
        acc = (logits > 0).float().mean().item()
    return loss, acc


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    agent_cfg.seed = args_cli.seed
    agent_cfg.device = args_cli.device if args_cli.device is not None else agent_cfg.device
    agent_cfg.use_learned_reward = False  # explicit: this script never touches the learned-
    # reward/KL-anchor/online-update code paths in on_policy_runner.py at all.
    # Load-bearing assumption, confirmed against this codebase's own history (get_trajectories.py's
    # LIFT_OBS_PRIVILEGED_FIELDS comment, 2026-08-18): for THIS specific DR-TableFix Lift task
    # variant, the 27-dim "privileged" observation this investigation's offline comparisons are
    # built from (joint_pos+object_position+target_object_position+actions, joint_vel deliberately
    # excluded) is BY DESIGN identical to the real actor's own live policy observation, not a
    # separate/richer critic-only space -- so feeding it directly to policy.act() below is correct
    # for this task, not a category error. If this script is ever pointed at a different task
    # variant, re-verify this before trusting the result -- a genuine actor/critic obs mismatch
    # would surface as an immediate shape-mismatch crash in policy.act(), not silent corruption.

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    device = agent_cfg.device
    os.makedirs(args_cli.output_dir, exist_ok=True)
    # A real log_dir (not None) is required -- OnPolicyRunner.__init__ only sets self.logger_type
    # (needed by save()) inside its "if self.log_dir is not None" branch; found via the first
    # smoke test (job 31857266) crashing at runner.save() with "no attribute 'logger_type'".
    # This also means a standard tensorboard SummaryWriter gets created here, same as every other
    # training script in this codebase -- harmless, not something this script otherwise needs.
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=args_cli.output_dir, device=device)
    policy = runner.alg.policy
    print(f"[CPL] Constructed policy via OnPolicyRunner (task={args_cli.task}, device={device}); "
          f"no reward model, no PPO rollout -- training directly on offline preference comparisons.")

    print(f"[CPL] Loading offline trajectories from {args_cli.meta_train}")
    raw_train = load_from_meta_dataset(args_cli.meta_train, obs_mode="privileged", task_family="lift")
    train_trajs, _ = precompute_features(raw_train, obs_mode="privileged")
    print(f"[CPL] {len(train_trajs)} trajectories loaded")

    print(f"[CPL] Generating {args_cli.total_comparisons} comparisons at fragment_length="
          f"{args_cli.fragment_length} (seed={args_cli.seed})")
    obs_a, act_a, obs_b, act_b, labels, _na, _nb = generate_all_comparisons(
        train_trajs, n_comparisons=args_cli.total_comparisons,
        fragment_length=args_cli.fragment_length, seed=args_cli.seed,
    )
    n = len(labels)
    rng = np.random.default_rng(args_cli.seed)
    perm = rng.permutation(n)
    n_val = int(n * args_cli.val_frac)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    def _subset(idx):
        return PreferencePairDataset(
            [obs_a[i] for i in idx], [act_a[i] for i in idx],
            [obs_b[i] for i in idx], [act_b[i] for i in idx], labels[idx],
        )

    train_ds, val_ds = _subset(train_idx), _subset(val_idx)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args_cli.batch_size, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args_cli.batch_size, shuffle=False)
    print(f"[CPL] {len(train_ds)} train pairs, {len(val_ds)} val pairs")

    optimizer = torch.optim.Adam(policy.parameters(), lr=args_cli.lr)
    best_val_acc, best_state, epochs_no_improve = -1.0, None, 0

    for epoch in range(args_cli.epochs):
        policy.train()
        train_losses, train_accs = [], []
        for batch in train_loader:
            loss, acc = _cpl_loss(policy, batch, device, args_cli.alpha, args_cli.contrastive_bias, args_cli.gamma)
            if loss is None:
                continue
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())
            train_accs.append(acc)

        policy.eval()
        val_losses, val_accs = [], []
        with torch.no_grad():
            for batch in val_loader:
                loss, acc = _cpl_loss(policy, batch, device, args_cli.alpha, args_cli.contrastive_bias, args_cli.gamma)
                if loss is None:
                    continue
                val_losses.append(loss.item())
                val_accs.append(acc)
        val_acc = float(np.mean(val_accs)) if val_accs else float("nan")
        print(f"[CPL] epoch {epoch+1}/{args_cli.epochs} | train_loss={np.mean(train_losses):.4f} "
              f"train_acc={np.mean(train_accs):.4f} | val_loss={np.mean(val_losses):.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().clone() for k, v in policy.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args_cli.patience:
                print(f"[CPL] early stopping at epoch {epoch+1} (best val_acc={best_val_acc:.4f})")
                break

    if best_state is not None:
        policy.load_state_dict(best_state)
    policy.eval()
    print(f"[CPL] training complete. best val preference-accuracy (CPL logit, not GT-reward): {best_val_acc:.4f}")

    if args_cli.meta_test:
        print(f"[CPL] scoring held-out {args_cli.meta_test}")
        raw_test = load_from_meta_dataset(args_cli.meta_test, obs_mode="privileged", task_family="lift")
        test_trajs, _ = precompute_features(raw_test, obs_mode="privileged")
        toa, taa, tob, tab, tlabels, _tna, _tnb = generate_all_comparisons(
            test_trajs, n_comparisons=2000, fragment_length=args_cli.fragment_length, seed=args_cli.seed + 1000,
        )
        test_ds = PreferencePairDataset(toa, taa, tob, tab, tlabels)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args_cli.batch_size, shuffle=False)
        test_accs = []
        with torch.no_grad():
            for batch in test_loader:
                _loss, acc = _cpl_loss(policy, batch, device, args_cli.alpha, args_cli.contrastive_bias, args_cli.gamma)
                if _loss is not None:
                    test_accs.append(acc)
        print(f"[CPL] held-out test preference-accuracy (CPL logit): {np.mean(test_accs):.4f}")

    # Not calling runner.save() directly: it references self.logger_type, which OnPolicyRunner
    # only ever sets inside learn() (found via job 31857266/31858858 both crashing here with
    # "no attribute 'logger_type'" regardless of log_dir) -- this script deliberately never calls
    # learn() (no PPO rollout is wanted), so that attribute is never created. Building the dict
    # inline instead, mirroring OnPolicyRunner.save()'s own format exactly (verified against its
    # source, rsl_rl/runners/on_policy_runner.py) rather than depending on a side effect of a
    # method this script doesn't use.
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
    with open(os.path.join(args_cli.output_dir, "cpl_config.json"), "w") as f:
        json.dump(vars(args_cli), f, indent=2, default=str)
    print(f"[CPL] saved checkpoint (rsl_rl-compatible, loadable via the standard "
          f"agent.resume=true/--checkpoint mechanism) to {ckpt_path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
