"""diagnose_no_termination.py -- standalone diagnostic to confirm whether apply_no_termination()'s
injected reward term (cart_out_of_bounds_penalty / torso_height_penalty) is actually registered
and firing inside the live RewardManager, not just present in env_cfg.rewards.__dict__.
"""
import argparse
import sys
import os
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--task_family", type=str, required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def p(*a):
    print(*a, flush=True)


def main():
    import gymnasium as gym
    import torch

    _REPO_ROOT = "/datasets/work/hri-fyp2025s1-2903/work/lingheng_stack_clean/hri-pl-frm-mvvd"
    sys.path.insert(0, os.path.join(_REPO_ROOT, "pref_learning"))
    from env_cfg_utils import apply_no_termination

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    p("STEP1: parsing env_cfg for", args_cli.task)
    env_cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=4, use_fabric=True)
    p("STEP2: BEFORE mutation, rewards.__dict__ keys:", list(vars(env_cfg.rewards).keys()))
    p("STEP2: BEFORE mutation, terminations.__dict__ keys:", list(vars(env_cfg.terminations).keys()))

    env_cfg = apply_no_termination(env_cfg, args_cli.task_family)
    p("STEP3: AFTER mutation, rewards.__dict__ keys:", list(vars(env_cfg.rewards).keys()))
    p("STEP3: AFTER mutation, terminations.__dict__ keys:", list(vars(env_cfg.terminations).keys()))

    p("STEP4: constructing env via gym.make...")
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = env.unwrapped
    p("STEP4: env constructed OK")

    rm = env.reward_manager
    p("\n=== RewardManager active terms ===")
    p(str(rm))
    p("active_terms:", rm.active_terms)

    obs, _ = env.reset()
    act_dim = env.action_space.shape[-1]
    action = torch.ones((env.num_envs, act_dim), device=env.device) * 10.0

    candidates = [n for n in rm.active_terms if "out_of_bounds" in n or "height_penalty" in n]
    if not candidates:
        p("ERROR: no matching term found in active_terms! Cannot track.")
        return
    target_term = candidates[0]
    term_idx = rm.active_terms.index(target_term)
    p(f"\nTracking term '{target_term}' at index {term_idx}")

    for i in range(200):
        obs, rew, terminated, truncated, info = env.step(action)
        if i % 20 == 0 or i > 180:
            step_reward_term = rm._step_reward[:, term_idx]
            p(f"step {i}: total_reward={rew[0].item():.4f}  {target_term}={step_reward_term[0].item():.4f}  "
              f"terminated={terminated[0].item()}  truncated={truncated[0].item()}")

    env.close()
    p("DONE_OK")


try:
    main()
except Exception:
    print("EXCEPTION IN MAIN:", flush=True)
    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
finally:
    simulation_app.close()
