# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play a trained RSL-RL checkpoint and collect rollout trajectories for preference learning.

Always logs privileged state (object/ee/goal position, joint velocities, actions, rewards -- the
`preflog` observation group) to one .npz file per trajectory. Pass --collect_images to additionally
save per-step rgb/depth/semantic_segmentation frames from the task's cameras -- only usable against a
camera-enabled task (e.g. Isaac-Lift-Cube-Franka-Camera-v0). Without the flag, no camera sensors are
looked up or touched at all, so this also works against camera-less tasks (e.g. the TableFix tasks).

Renamed 2026-08-13 from play_oneil.py. One script instead of two so the privileged-only and
full-image collection paths can't drift apart -- the collection control flow (obs bookkeeping,
per-env trajectory rollover) is unchanged from the original; only the camera/image-saving statements
are gated behind --collect_images, plus dropped the dead commented-out alternate versions of the same
logic that had accumulated in play_oneil.py, and replaced the hardcoded, other-machine data_dir with
a required --out_dir.

Usage:
    # Privileged-only, against the already-calibrated TableFix task -- no camera needed.
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_collect_pref_data.py \
        --task Isaac-Lift-Cube-Franka-Delta-DR-TableFix-v0 --num_envs 4 \
        --checkpoint <path/to/model_*.pt> --out_dir <path> --headless

    # Full collection (images + privileged), against the camera task.
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_collect_pref_data.py \
        --task Isaac-Lift-Cube-Franka-Camera-v0 --num_envs 2 --collect_images \
        --checkpoint <path/to/model_*.pt> --out_dir <path> --headless
"""
"""Launch Isaac Sim Simulator first."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Collect preference-learning trajectories from a trained checkpoint.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--collect_images",
    action="store_true",
    default=False,
    help="Also save per-step rgb/depth/semantic_segmentation frames (front/side/bird/hand, plus "
    "robot-only and object-only semantic masks). Requires a camera-enabled task.",
)
parser.add_argument("--out_dir", type=str, required=True, help="Root directory to write collected trajectories to.")
parser.add_argument(
    "--random_policy",
    action="store_true",
    default=False,
    help="Collect with i.i.d. uniform random actions instead of a trained checkpoint -- no "
    "--checkpoint needed. Added 2026-08-25 to give the reward model exposure to jerky/violent "
    "motion (uniformly sampled actions produce high action-rate variance by construction), "
    "since every checkpoint-sourced collection so far only ever showed it smooth, converging-"
    "toward-good behaviour -- the reward net has no calibrated opinion on what jerky motion "
    "looks like, which the PPO-against-learned-reward exploit repeatedly found and exploited.",
)
parser.add_argument(
    "--random_policy_tag", type=str, default="",
    help="Unique tag for this --random_policy run's output subfolder (e.g. $SLURM_JOB_ID), so "
    "multiple parallel random-policy collection jobs don't overwrite each other's env_i/traj_NNN "
    "files -- unlike checkpoint-sourced collections, random rollouts have no natural per-run "
    "uniqueness source. Required when --random_policy is set.",
)
parser.add_argument(
    "--total_traj",
    type=int,
    default=52,
    help="Total trajectories to collect across all envs combined (the first trajectory per env is "
    "discarded as warm-up, matching the original play_oneil.py behaviour).",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video or collect image frames
if args_cli.video or args_cli.collect_images:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import json
import os
import time
import torch
import sys
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'rsl_rl', 'rsl_rl'))
from runners.on_policy_runner import OnPolicyRunner
import shutil
import omni.usd
from pxr import Sdf, Usd, UsdGeom
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab.utils.math import subtract_frame_transforms
# PLACEHOLDER: Extension template (do not remove this comment)
import numpy as np
from isaaclab.sensors import save_images_to_file, depth_to_rgba
from torchvision.utils import make_grid, save_image

def main():
    """Play with RSL-RL agent."""
    task_name = args_cli.task.split(":")[-1]
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )

    # Force the cube-spawn curriculum (if this task has one) to its fully-ramped end state for
    # collection. A freshly-created env's common_step_counter starts at 0 regardless of which policy
    # checkpoint is loaded, so a curriculum term that only widens the cube-spawn range after N steps of
    # *this* env's own lifetime would otherwise leave every collection run sampling only the narrow
    # "easy" range, never the wide range the policy was actually trained/evaluated on (found 2026-08-14:
    # a full 9-checkpoint collection stayed under the ramp-start step count throughout). Read the wide
    # end-range values from the curriculum term's own config, not hardcoded, so this keeps working if
    # the task's curriculum parameters change.
    collection_metadata = {"task": args_cli.task}
    widen_term = getattr(env_cfg.curriculum, "widen_cube_range", None)
    if widen_term is not None:
        x_end = widen_term.params["x_end"]
        y_end = widen_term.params["y_end"]
        event_term = getattr(env_cfg.events, widen_term.params["term_name"])
        event_term.params["pose_range"]["x"] = x_end
        event_term.params["pose_range"]["y"] = y_end
        env_cfg.curriculum.widen_cube_range = None
        collection_metadata["cube_spawn_range"] = {"x": list(x_end), "y": list(y_end)}
        print(f"[INFO] Forced cube-spawn range to curriculum's fully-ramped end state: x={x_end}, y={y_end}")
    else:
        pr = env_cfg.events.reset_object_position.params["pose_range"]
        collection_metadata["cube_spawn_range"] = {"x": list(pr["x"]), "y": list(pr["y"])}

    # Force any reward-WEIGHT curricula (e.g. action_rate/joint_vel modify_reward_weight, which ramps
    # -1e-4 -> -1e-1 after common_step_counter > 10000, i.e. ~83% of a 2500-iteration training run) to
    # their fully-ramped target weight before snapshotting, for the exact same reason as the
    # widen_cube_range handling above: a freshly-created collection env's common_step_counter starts at
    # 0 regardless of which checkpoint is loaded, so env_cfg.rewards.<term>.weight is always the
    # pre-curriculum startup value, never the mature one the checkpoint was actually shaped by for most
    # of training. Using the startup weight to reconstruct GT-based preference labels for ALL
    # checkpoints under-penalizes jerky/high-action-rate motion by 1000x -- traced 2026-08-25 as the
    # likely cause of the violent-motion reward-hacking exploit seen across every learned-reward PPO
    # run. Generic over any CurrTerm using mdp.modify_reward_weight, not hardcoded to these two names.
    for curr_name in vars(env_cfg.curriculum).keys():
        if curr_name.startswith("_"):
            continue
        curr_term = getattr(env_cfg.curriculum, curr_name, None)
        if curr_term is None or curr_term.func.__name__ != "modify_reward_weight":
            continue
        target_term_name = curr_term.params["term_name"]
        target_weight = curr_term.params["weight"]
        reward_term = getattr(env_cfg.rewards, target_term_name)
        old_weight = reward_term.weight
        reward_term.weight = target_weight
        print(f"[INFO] Forced reward term '{target_term_name}' to curriculum's fully-ramped weight: "
              f"{old_weight} -> {target_weight}")

    # Snapshot the actual reward-term parameters this env is using, so downstream reward reconstruction
    # (pref_learning/get_trajectories.py's calculate_reward) can read the real values instead of hardcoding
    # its own copy that can silently drift out of sync with the task -- exactly how the TABLE_RAISE
    # is_lifted-threshold bug happened (found 2026-08-14).
    def _reward_term_snapshot(name):
        term = getattr(env_cfg.rewards, name, None)
        if term is None:
            return None
        snap = {"weight": term.weight}
        for key, value in (term.params or {}).items():
            if isinstance(value, (int, float, str, bool)):
                snap[key] = value
        return snap

    collection_metadata["reward_terms"] = {
        name: snap
        for name in (
            "reaching_object", "lifting_object", "object_goal_tracking",
            "object_goal_tracking_fine_grained", "action_rate", "joint_vel",
        )
        if (snap := _reward_term_snapshot(name)) is not None
    }
    collection_metadata["decimation"] = env_cfg.decimation
    collection_metadata["sim_dt"] = env_cfg.sim.dt

    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(task_name, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    if args_cli.random_policy:
        resume_path = None
        log_dir = None
    else:
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        if args_cli.use_pretrained_checkpoint:
            resume_path = get_published_pretrained_checkpoint("rsl_rl", task_name)
            if not resume_path:
                print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
                return
        elif args_cli.checkpoint:
            resume_path = retrieve_file_path(args_cli.checkpoint)
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

        log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir or args_cli.out_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    if args_cli.random_policy:
        # i.i.d. uniform random actions in [-1, 1] -- matches the normalized action-space bounds
        # PPO's own policy output is clipped to (RslRlVecEnvWrapper(..., clip_actions=...) above),
        # so this samples from the same range a trained policy could ever produce, just without any
        # temporal correlation/smoothness -- exactly the jerky-motion coverage gap being targeted.
        action_shape = env.unwrapped.action_space.shape
        action_device = env.unwrapped.device
        def policy(obs):
            return torch.rand(action_shape, device=action_device) * 2 - 1
        print("[INFO]: Using i.i.d. uniform random actions (--random_policy), no checkpoint loaded.")
    else:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        ppo_runner.load(resume_path)

        # obtain the trained policy for inference
        policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

        # extract the neural network module
        # we do this in a try-except to maintain backwards compatibility.
        try:
            # version 2.3 onwards
            policy_nn = ppo_runner.alg.policy
        except AttributeError:
            # version 2.2 and below
            policy_nn = ppo_runner.alg.actor_critic

        # export policy to onnx/jit
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
        export_policy_as_jit(policy_nn, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(
            policy_nn, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
        )

    dt = env.unwrapped.step_dt

    #file locations
    data_dir = args_cli.out_dir

    #make checkpoint directory
    if args_cli.random_policy:
        if not args_cli.random_policy_tag:
            raise ValueError("--random_policy_tag is required when --random_policy is set, so "
                              "parallel random-policy collection jobs don't overwrite each other.")
        checkpoint_dir = os.path.join(data_dir, f"random_policy_{args_cli.random_policy_tag}")
    else:
        checkpoint_location = resume_path[resume_path.rfind('/') + 1:]
        checkpoint_location = checkpoint_location.split('.')[0]
        log_location = log_dir[log_dir.rfind('/') + 1:]
        checkpoint_dir = os.path.join(data_dir, f"{log_location}_{checkpoint_location}")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    #env directories
    for i in range(env_cfg.scene.num_envs):
        if not os.path.exists(os.path.join(checkpoint_dir, f"env_{i}")):
            os.makedirs(os.path.join(checkpoint_dir, f"env_{i}"))
    if args_cli.random_policy:
        collection_metadata["checkpoint"] = "random_policy (no trained checkpoint -- i.i.d. uniform random actions)"
    else:
        shutil.copyfile(resume_path, os.path.join(checkpoint_dir, f"{checkpoint_location}.pt"))
        collection_metadata["checkpoint"] = resume_path
    with open(os.path.join(checkpoint_dir, "collection_metadata.json"), "w") as f:
        json.dump(collection_metadata, f, indent=2)
    frame_idx =0
    total_traj = args_cli.total_traj #total trajectories to collect - throw away first trajectories
    num_envs = env_cfg.scene.num_envs
    pref_log = [{'actions': [], 'rewards': []} for _ in range(num_envs)]
    traj = np.zeros(num_envs, dtype=int)
    steps_per_traj = np.zeros(num_envs, dtype=int)

    env_ids = torch.arange(env_cfg.scene.num_envs)
    save_data = [{} for _ in range(num_envs)]
    for env_idx in range(env_cfg.scene.num_envs):
            for type_cam in ['rgb','depth','semantic_all','semantic_robot','semantic_object']:
                save_data[env_idx][type_cam] = {}
                for cam in ['front','side','bird','hand']:
                    save_data[env_idx][type_cam][cam] = {}

    for i in range(env_cfg.scene.num_envs):
        if not os.path.exists(os.path.join(checkpoint_dir, f"env_{i}",f"traj_{traj[i]:03d}")):
            #trajectory directory
            os.makedirs(os.path.join(checkpoint_dir, f"env_{i}",f"traj_{traj[i]:03d}"))
            if args_cli.collect_images:
                #data types
                os.makedirs(os.path.join(checkpoint_dir, f"env_{i}",f"traj_{traj[i]:03d}","rgb"))
                os.makedirs(os.path.join(checkpoint_dir, f"env_{i}",f"traj_{traj[i]:03d}","depth"))
                #camera types
                os.makedirs(os.path.join(checkpoint_dir, f"env_{i}",f"traj_{traj[i]:03d}","rgb","front"))
                os.makedirs(os.path.join(checkpoint_dir, f"env_{i}",f"traj_{traj[i]:03d}","rgb","side"))
                os.makedirs(os.path.join(checkpoint_dir, f"env_{i}",f"traj_{traj[i]:03d}","rgb","bird"))
                os.makedirs(os.path.join(checkpoint_dir, f"env_{i}",f"traj_{traj[i]:03d}","rgb","hand"))

                os.makedirs(os.path.join(checkpoint_dir, f"env_{i}",f"traj_{traj[i]:03d}","depth","front"))
                os.makedirs(os.path.join(checkpoint_dir, f"env_{i}",f"traj_{traj[i]:03d}","depth","side"))
                os.makedirs(os.path.join(checkpoint_dir, f"env_{i}",f"traj_{traj[i]:03d}","depth","bird"))
                os.makedirs(os.path.join(checkpoint_dir, f"env_{i}",f"traj_{traj[i]:03d}","depth","hand"))

                os.makedirs(os.path.join(checkpoint_dir, f"env_{i}",f"traj_{traj[i]:03d}","semantic_all","front"))
                os.makedirs(os.path.join(checkpoint_dir, f"env_{i}",f"traj_{traj[i]:03d}","semantic_all","side"))
                os.makedirs(os.path.join(checkpoint_dir, f"env_{i}",f"traj_{traj[i]:03d}","semantic_all","bird"))
                os.makedirs(os.path.join(checkpoint_dir, f"env_{i}",f"traj_{traj[i]:03d}","semantic_all","hand"))

                os.makedirs(os.path.join(checkpoint_dir, f"env_{i}",f"traj_{traj[i]:03d}","semantic_robot","front"))
                os.makedirs(os.path.join(checkpoint_dir, f"env_{i}",f"traj_{traj[i]:03d}","semantic_robot","side"))
                os.makedirs(os.path.join(checkpoint_dir, f"env_{i}",f"traj_{traj[i]:03d}","semantic_robot","bird"))


                os.makedirs(os.path.join(checkpoint_dir, f"env_{i}",f"traj_{traj[i]:03d}","semantic_object","front"))
                os.makedirs(os.path.join(checkpoint_dir, f"env_{i}",f"traj_{traj[i]:03d}","semantic_object","side"))
                os.makedirs(os.path.join(checkpoint_dir, f"env_{i}",f"traj_{traj[i]:03d}","semantic_object","bird"))


    stage = omni.usd.get_context().get_stage()
    which_obs = ['pobserve' for _ in range(env_cfg.scene.num_envs)]
    if args_cli.collect_images:
        sensor = env.unwrapped.scene["camera_ext1"] #front
        sensor1 = env.unwrapped.scene["camera_ext2"] #side
        sensor2 = env.unwrapped.scene["camera_bird"]
        sensor3 = env.unwrapped.scene["camera"]
    else:
        sensor = sensor1 = sensor2 = sensor3 = None
    asset = env.unwrapped.scene["object"]
    robot = env.unwrapped.scene["robot"]
    dummy_obs = ['observations' for _ in range(env_cfg.scene.num_envs)]
    # reset environment
    obs, infos = env.get_observations()
    dummy_dones = [0 for _ in range(env_cfg.scene.num_envs)]
    dummy_ones = [1 for _ in range(env_cfg.scene.num_envs)]
    for i in range(env_cfg.scene.num_envs):
        save_step_data(dummy_obs,traj, steps_per_traj, pref_log, checkpoint_dir, sensor, sensor1, sensor2,sensor3,infos, env,save_data,stage,dummy_ones,i,args_cli.collect_images)
    frame_idx+=1
    for i in range(len(steps_per_traj)):
        steps_per_traj[i] += 1
    print(f"Frame: {frame_idx} completed.")
    timestep = 0

    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():

            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, rewards, dones, infos = env.step(actions)

            for i in range (env_cfg.scene.num_envs):
                if dones[i]:
                    print(f"Environment {i} done at frame {frame_idx}")

                    # see the note below on 'pobserve' not existing for camera-less tasks
                    which_obs[i] = 'observations' if not args_cli.collect_images else 'pobserve'

                    save_step_data(which_obs,traj, steps_per_traj, pref_log, checkpoint_dir, sensor, sensor1, sensor2,sensor3,infos, env,save_data,stage,dummy_dones,i,args_cli.collect_images)
                    np.savez(os.path.join(checkpoint_dir, f"env_{i}",f"traj_{traj[i]:03d}",f"env{i}_traj{traj[i]:03d}.npz"), **pref_log[i])

                    total_traj -= 1

                    pref_log[i] = {'actions': [], 'rewards': []}
                    traj[i] +=1
                    steps_per_traj[i] = 0
                elif frame_idx > 0: # skip the very first step as there is no action/reward yet
                    pref_log[i]['actions'].append(actions[i].cpu().numpy())
                    pref_log[i]['rewards'].append(rewards[i].cpu().numpy())
                if not os.path.exists(os.path.join(checkpoint_dir, f"env_{i}",f"traj_{traj[i]:03d}")):
                    #trajectory directory
                    os.makedirs(os.path.join(checkpoint_dir, f"env_{i}",f"traj_{traj[i]:03d}"))
                    if args_cli.collect_images:
                        #data types
                        os.makedirs(os.path.join(checkpoint_dir, f"env_{i}",f"traj_{traj[i]:03d}","rgb"))
                        os.makedirs(os.path.join(checkpoint_dir, f"env_{i}",f"traj_{traj[i]:03d}","depth"))
                        #camera types
                        os.makedirs(os.path.join(checkpoint_dir, f"env_{i}",f"traj_{traj[i]:03d}","rgb","front"))
                        os.makedirs(os.path.join(checkpoint_dir, f"env_{i}",f"traj_{traj[i]:03d}","rgb","side"))
                        os.makedirs(os.path.join(checkpoint_dir, f"env_{i}",f"traj_{traj[i]:03d}","rgb","bird"))
                        os.makedirs(os.path.join(checkpoint_dir, f"env_{i}",f"traj_{traj[i]:03d}","rgb","hand"))

                        os.makedirs(os.path.join(checkpoint_dir, f"env_{i}",f"traj_{traj[i]:03d}","depth","front"))
                        os.makedirs(os.path.join(checkpoint_dir, f"env_{i}",f"traj_{traj[i]:03d}","depth","side"))
                        os.makedirs(os.path.join(checkpoint_dir, f"env_{i}",f"traj_{traj[i]:03d}","depth","bird"))
                        os.makedirs(os.path.join(checkpoint_dir, f"env_{i}",f"traj_{traj[i]:03d}","depth","hand"))

                        os.makedirs(os.path.join(checkpoint_dir, f"env_{i}",f"traj_{traj[i]:03d}","semantic_all","front"))
                        os.makedirs(os.path.join(checkpoint_dir, f"env_{i}",f"traj_{traj[i]:03d}","semantic_all","side"))
                        os.makedirs(os.path.join(checkpoint_dir, f"env_{i}",f"traj_{traj[i]:03d}","semantic_all","bird"))
                        os.makedirs(os.path.join(checkpoint_dir, f"env_{i}",f"traj_{traj[i]:03d}","semantic_all","hand"))

                        os.makedirs(os.path.join(checkpoint_dir, f"env_{i}",f"traj_{traj[i]:03d}","semantic_robot","front"))
                        os.makedirs(os.path.join(checkpoint_dir, f"env_{i}",f"traj_{traj[i]:03d}","semantic_robot","side"))
                        os.makedirs(os.path.join(checkpoint_dir, f"env_{i}",f"traj_{traj[i]:03d}","semantic_robot","bird"))

                        os.makedirs(os.path.join(checkpoint_dir, f"env_{i}",f"traj_{traj[i]:03d}","semantic_object","front"))
                        os.makedirs(os.path.join(checkpoint_dir, f"env_{i}",f"traj_{traj[i]:03d}","semantic_object","side"))
                        os.makedirs(os.path.join(checkpoint_dir, f"env_{i}",f"traj_{traj[i]:03d}","semantic_object","bird"))

            for i in range(env_cfg.scene.num_envs):
                if not args_cli.collect_images:
                    # 'pobserve' is populated only for camera-instrumented tasks (confirmed empirically
                    # 2026-08-13: infos['pobserve'] does not exist at all against a camera-less task,
                    # even on a non-done step) -- privileged `preflog` data is reliably available under
                    # 'observations' every step regardless of done-state, so skip the toggle entirely.
                    which_obs[i] = 'observations'
                elif dones[i]:
                    which_obs[i] = 'observations'
                else:
                    which_obs[i] = 'pobserve'
            if total_traj <= 0:
                break

            for i in range(env_cfg.scene.num_envs):
                save_step_data(which_obs,traj, steps_per_traj, pref_log, checkpoint_dir, sensor, sensor1, sensor2,sensor3,infos, env,save_data,stage,dones,i,args_cli.collect_images)

            
            frame_idx+=1
            for i in range(len(steps_per_traj)):
                steps_per_traj[i] += 1
            print(f"Frame: {frame_idx} completed.")

        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()

def hard_reset_prim_visibility(prim_path, stage, visible=True):
    # stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        print(f"[WARN] Prim not found at {prim_path}")
        return
    
    imageable = UsdGeom.Imageable(prim)
    visibility_attr = imageable.GetVisibilityAttr()
    
    # # Remove the visibility attribute entirely (clears any overrides)
    # if visibility_attr.HasAuthoredValueOpinion():
    #     prim.RemoveProperty('visibility')

    # # Recreate the attribute to force a fresh value
    # visibility_attr = imageable.CreateVisibilityAttr()
    visibility_attr.Set('visible' if visible else 'invisible')


def save_step_data(which_obs,traj, steps_per_traj, pref_log, checkpoint_dir, sensor, sensor1, sensor2,sensor3,infos, env,save_data,stage,dones,i,collect_images):

    # save data function -- always logs privileged state; everything below (unchanged from
    # play_oneil.py) only runs when collecting images, and is unreachable/untouched otherwise.
    for key, step_pref_log in infos[which_obs[i]]['preflog'].items():
        if key not in pref_log[i]:
            pref_log[i][key] = []
        pref_log[i][key].append(step_pref_log[i].cpu().numpy())

    if not collect_images:
        return

    images = infos[which_obs[i]]['rgb']['image']
    images1 = infos[which_obs[i]]['rgb']['image1']
    images2 = infos[which_obs[i]]['rgb']['image2']
    images3 = infos[which_obs[i]]['rgb']['image3']
    depth = infos[which_obs[i]]['depth']['image']
    depth1 = infos[which_obs[i]]['depth']['image1']
    depth2 = infos[which_obs[i]]['depth']['image2']
    depth3 = infos[which_obs[i]]['depth']['image3']
    semantic = infos[which_obs[i]]['semantic']['image']
    semantic1 = infos[which_obs[i]]['semantic']['image1']
    semantic2 = infos[which_obs[i]]['semantic']['image2']
    semantic3 = infos[which_obs[i]]['semantic']['image3']
    semantic_rgb = semantic[:, :, :, :3]
    semantic1_rgb = semantic1[:, :, :, :3]
    semantic2_rgb = semantic2[:, :, :, :3]
    semantic3_rgb = semantic3[:, :, :, :3]
    
    save_data[i]['rgb']['hand'][os.path.join(checkpoint_dir, f"env_{i}",f"traj_{traj[i]:03d}","rgb","hand",f"{steps_per_traj[i]:03d}.jpg")] = images[i]
    save_data[i]['rgb']['front'][os.path.join(checkpoint_dir, f"env_{i}",f"traj_{traj[i]:03d}","rgb","front",f"{steps_per_traj[i]:03d}.jpg")] = images1[i]
    save_data[i]['rgb']['side'][os.path.join(checkpoint_dir, f"env_{i}",f"traj_{traj[i]:03d}","rgb","side",f"{steps_per_traj[i]:03d}.jpg")] = images2[i]
    save_data[i]['rgb']['bird'][os.path.join(checkpoint_dir, f"env_{i}",f"traj_{traj[i]:03d}","rgb","bird",f"{steps_per_traj[i]:03d}.jpg")] = images3[i]

    save_data[i]['depth']['hand'][os.path.join(checkpoint_dir, f"env_{i}",f"traj_{traj[i]:03d}","depth","hand",f"{steps_per_traj[i]:03d}.npz")] = depth[i].cpu()
    save_data[i]['depth']['front'][os.path.join(checkpoint_dir, f"env_{i}",f"traj_{traj[i]:03d}","depth","front",f"{steps_per_traj[i]:03d}.npz")] = depth1[i].cpu()
    save_data[i]['depth']['side'][os.path.join(checkpoint_dir, f"env_{i}",f"traj_{traj[i]:03d}","depth","side",f"{steps_per_traj[i]:03d}.npz")] = depth2[i].cpu()
    save_data[i]['depth']['bird'][os.path.join(checkpoint_dir, f"env_{i}",f"traj_{traj[i]:03d}","depth","bird",f"{steps_per_traj[i]:03d}.npz")] = depth3[i].cpu()

    save_data[i]['semantic_all']['hand'][os.path.join(checkpoint_dir, f"env_{i}",f"traj_{traj[i]:03d}","semantic_all","hand",f"{steps_per_traj[i]:03d}.jpg")] = semantic_rgb[i]/255.0
    save_data[i]['semantic_all']['front'][os.path.join(checkpoint_dir, f"env_{i}",f"traj_{traj[i]:03d}","semantic_all","front",f"{steps_per_traj[i]:03d}.jpg")] = semantic1_rgb[i]/255.0
    save_data[i]['semantic_all']['side'][os.path.join(checkpoint_dir, f"env_{i}",f"traj_{traj[i]:03d}","semantic_all","side",f"{steps_per_traj[i]:03d}.jpg")] = semantic2_rgb[i]/255.0
    save_data[i]['semantic_all']['bird'][os.path.join(checkpoint_dir, f"env_{i}",f"traj_{traj[i]:03d}","semantic_all","bird",f"{steps_per_traj[i]:03d}.jpg")] = semantic3_rgb[i]/255.0
    if dones[i]:
        hard_reset_prim_visibility(f"/World/envs/env_{i}/Object",stage,False) 
        env.unwrapped.sim.render()
        # sensor.reset()
        sensor.update(dt=0, force_recompute=True) 
        # sensor1.reset()
        sensor1.update(dt=0, force_recompute=True) 
        # sensor2.reset()
        sensor2.update(dt=0, force_recompute=True)
        # sensor3.reset()
        sensor3.update(dt=0, force_recompute=True)
        
        obs,infos = env.get_observations()
        semantic1_robot = infos['observations']['semantic']['image1']
        semantic2_robot = infos['observations']['semantic']['image2']
        semantic3_robot = infos['observations']['semantic']['image3']
        semantic1_rgb_robot = semantic1_robot[:, :, :, :3]
        semantic2_rgb_robot = semantic2_robot[:, :, :, :3]
        semantic3_rgb_robot = semantic3_robot[:, :, :, :3]
        
        save_data[i]['semantic_robot']['front'][os.path.join(checkpoint_dir, f"env_{i}",f"traj_{traj[i]:03d}","semantic_robot","front",f"{steps_per_traj[i]:03d}.jpg")] = semantic1_rgb_robot[i]/255.0
        save_data[i]['semantic_robot']['side'][os.path.join(checkpoint_dir, f"env_{i}",f"traj_{traj[i]:03d}","semantic_robot","side",f"{steps_per_traj[i]:03d}.jpg")] = semantic2_rgb_robot[i]/255.0
        save_data[i]['semantic_robot']['bird'][os.path.join(checkpoint_dir, f"env_{i}",f"traj_{traj[i]:03d}","semantic_robot","bird",f"{steps_per_traj[i]:03d}.jpg")] = semantic3_rgb_robot[i]/255.0
    
        hard_reset_prim_visibility(f"/World/envs/env_{i}/Robot",stage,False)     
        hard_reset_prim_visibility(f"/World/envs/env_{i}/Object",stage,True)
        # import pdb; pdb.set_trace()
        # env.unwrapped.sim.step()
        env.unwrapped.sim.render()
        # pdb.set_trace()
        # env.unwrapped.render()
        # sensor.reset()
        sensor.update(dt=0, force_recompute=True) 
        # sensor1.reset()
        sensor1.update(dt=0, force_recompute=True) 
        # sensor2.reset()
        sensor2.update(dt=0, force_recompute=True)
        # sensor3.reset()
        sensor3.update(dt=0, force_recompute=True)
        obs,infos = env.get_observations()
        # semantic_object = infos['observations']['semantic']['image']
        semantic1_object = infos['observations']['semantic']['image1']
        semantic2_object = infos['observations']['semantic']['image2']
        semantic3_object = infos['observations']['semantic']['image3']
        # semantic_rgb_object = semantic_object[:, :, :, :3]
        semantic1_rgb_object = semantic1_object[:, :, :, :3]
        semantic2_rgb_object = semantic2_object[:, :, :, :3]
        semantic3_rgb_object = semantic3_object[:, :, :, :3]
    
        save_data[i]['semantic_object']['front'][os.path.join(checkpoint_dir, f"env_{i}",f"traj_{traj[i]:03d}","semantic_object","front",f"{steps_per_traj[i]:03d}.jpg")] = semantic1_rgb_object[i]/255.0
        save_data[i]['semantic_object']['side'][os.path.join(checkpoint_dir, f"env_{i}",f"traj_{traj[i]:03d}","semantic_object","side",f"{steps_per_traj[i]:03d}.jpg")] = semantic2_rgb_object[i]/255.0
        save_data[i]['semantic_object']['bird'][os.path.join(checkpoint_dir, f"env_{i}",f"traj_{traj[i]:03d}","semantic_object","bird",f"{steps_per_traj[i]:03d}.jpg")] = semantic3_rgb_object[i]/255.0
        hard_reset_prim_visibility(f"/World/envs/env_{i}/Robot",stage,True)     
        hard_reset_prim_visibility(f"/World/envs/env_{i}/Object",stage,True)
    else:
        semantic1_robot = infos['semantic_robot']['image1']
        semantic2_robot = infos['semantic_robot']['image2']
        semantic3_robot = infos['semantic_robot']['image3']
        semantic1_rgb_robot = semantic1_robot[:, :, :, :3]
        semantic2_rgb_robot = semantic2_robot[:, :, :, :3]
        semantic3_rgb_robot = semantic3_robot[:, :, :, :3]
    
        save_data[i]['semantic_robot']['front'][os.path.join(checkpoint_dir, f"env_{i}",f"traj_{traj[i]:03d}","semantic_robot","front",f"{steps_per_traj[i]:03d}.jpg")] = semantic1_rgb_robot[i]/255.0
        save_data[i]['semantic_robot']['side'][os.path.join(checkpoint_dir, f"env_{i}",f"traj_{traj[i]:03d}","semantic_robot","side",f"{steps_per_traj[i]:03d}.jpg")] = semantic2_rgb_robot[i]/255.0
        save_data[i]['semantic_robot']['bird'][os.path.join(checkpoint_dir, f"env_{i}",f"traj_{traj[i]:03d}","semantic_robot","bird",f"{steps_per_traj[i]:03d}.jpg")] = semantic3_rgb_robot[i]/255.0

        semantic1_object = infos['semantic_object']['image1']
        semantic2_object = infos['semantic_object']['image2']
        semantic3_object = infos['semantic_object']['image3']
        # semantic_rgb_object = semantic_object[:, :, :, :3]
        semantic1_rgb_object = semantic1_object[:, :, :, :3]
        semantic2_rgb_object = semantic2_object[:, :, :, :3]
        semantic3_rgb_object = semantic3_object[:, :, :, :3]
    
        save_data[i]['semantic_object']['front'][os.path.join(checkpoint_dir, f"env_{i}",f"traj_{traj[i]:03d}","semantic_object","front",f"{steps_per_traj[i]:03d}.jpg")] = semantic1_rgb_object[i]/255.0
        save_data[i]['semantic_object']['side'][os.path.join(checkpoint_dir, f"env_{i}",f"traj_{traj[i]:03d}","semantic_object","side",f"{steps_per_traj[i]:03d}.jpg")] = semantic2_rgb_object[i]/255.0
        save_data[i]['semantic_object']['bird'][os.path.join(checkpoint_dir, f"env_{i}",f"traj_{traj[i]:03d}","semantic_object","bird",f"{steps_per_traj[i]:03d}.jpg")] = semantic3_rgb_object[i]/255.0



        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for type_cam in ['rgb','depth','semantic_all','semantic_robot','semantic_object']:
                for cam in ['front','side','bird','hand']:
                    if cam == 'hand' and type_cam == 'semantic_object':
                            continue
                    if cam == 'hand' and type_cam == 'semantic_robot':
                            continue
                    for file_path in save_data[i][type_cam][cam]:
                        if type_cam == 'depth':
                            futures.append(executor.submit(
                                np.savez_compressed,
                                file_path,
                                save_data[i][type_cam][cam][file_path])
                            )
                        else:
                            # import pdb; pdb.set_trace()
                            futures.append(executor.submit(
                                save_images_to_file,
                                save_data[i][type_cam][cam][file_path],
                                file_path
                            ))
            # Wait for all threads to complete
            for future in as_completed(futures):
                future.result()  # This will also raise exceptions if any occurred

        save_data[i] = {'rgb': {'front': {}, 'side': {}, 'bird': {}, 'hand': {}},
                        'depth': {'front': {}, 'side': {}, 'bird': {}, 'hand': {}},
                        'semantic_all': {'front': {}, 'side': {}, 'bird': {}, 'hand': {}},
                        'semantic_robot': {'front': {}, 'side': {}, 'bird': {}},
                        'semantic_object': {'front': {}, 'side': {}, 'bird': {}}}



if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
