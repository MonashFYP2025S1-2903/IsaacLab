# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""
"""./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_oneil.py --task=Isaac-Lift-Cube-Franka-Camera-v0 --num_envs=2 --enable_cameras --load_run /home/sh-d61-cps-hri/hri-pl-frm-mvvd/isaaclab/logs/rsl_rl/franka_lift/2025-09-12_13-48-36 --checkpoint /home/sh-d61-cps-hri/hri-pl-frm-mvvd/isaaclab/logs/rsl_rl/franka_lift/2025-09-12_13-48-36/model_1050.pt
"""
"""Launch Isaac Sim Simulator first."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

import shutil

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
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
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch
import sys
sys.path.insert(0, '/home/sh-d61-cps-hri/hri-pl-frm-mvvd/rsl_rl/rsl_rl')
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
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(task_name, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
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
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

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
    # data_dir = "/media/sh-d61-cps-hri/franka_panda_portable_disk_2T/FYP2025S1_2903/oneil_test"
    # data_dir = "/media/sh-d61-cps-hri/franka_panda_portable_disk_2T/FYP2025S1_2903/mohammad_test"
    data_dir = "/media/sh-d61-cps-hri/franka_panda_portable_disk_2T/FYP2025S1_2903/thread_data"

    #make checkpoint directory
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
    shutil.copyfile(resume_path, os.path.join(checkpoint_dir, f"{checkpoint_location}.pt"))

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0
    import csv
    sensor = env.unwrapped.scene["camera_ext1"] #front
    sensor1 = env.unwrapped.scene["camera_ext2"] #side
    sensor2 = env.unwrapped.scene["camera_bird"]
    # sensor3 = env.unwrapped.scene["camera"]
    asset = env.unwrapped.scene["object"]
    robot = env.unwrapped.scene["robot"]
    robots = env.unwrapped.scene.articulations['robot']          # get robot

    rigid_objects = env.unwrapped.scene.rigid_objects['object']                 # get rigid objects

    # File to write to
    # csv_file = "data.csv"
    # counter = 0

    # # If file doesn’t exist yet, create and write header
    # file_exists = os.path.isfile(csv_file)
    # with open(csv_file, mode="a", newline="") as f:
    #     writer = csv.writer(f)
    #     if not file_exists:
    #         writer.writerow(["frame", "env_id", "cube_pos", "cube_ore","robot_pos","robot_ore","cube_changed_pos","cube_changed_ore","front_img_rgb","side_img_rgb","bird_img_rgb",
    #                          "front_img_dp","side_img_dp","bird_img_dp","env_origin","front_sem","side_sem","bird_sem","front_sem_robot","side_sem_robot","bird_sem_robot",
    #                          "front_sem_object","side_sem_object","bird_sem_object"])  # header row
    frame_idx =0
    total_traj = 1000  #total trajectories to collect
    num_envs = env_cfg.scene.num_envs
    pref_log = [{'actions': [], 'rewards': []} for _ in range(num_envs)]
    traj = np.zeros(num_envs, dtype=int)
    steps_per_traj = np.zeros(num_envs, dtype=int)

    env_ids = torch.arange(env_cfg.scene.num_envs)

    save_data = [{} for _ in range(num_envs)]

    for env_idx in range(env_cfg.scene.num_envs):
        for type_cam in ['rgb','depth','semantic_all']:
            save_data[env_idx][type_cam] = {}
            for cam in ['front','side','bird']:
                save_data[env_idx][type_cam][cam] = {}

    for i in range(env_cfg.scene.num_envs):
        if not os.path.exists(os.path.join(checkpoint_dir, f"env_{i}",f"traj_{traj[i]:03d}")):
            #trajectory directory
            os.makedirs(os.path.join(checkpoint_dir, f"env_{i}",f"traj_{traj[i]:03d}"))
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
            os.makedirs(os.path.join(checkpoint_dir, f"env_{i}",f"traj_{traj[i]:03d}","semantic_robot","hand"))

            os.makedirs(os.path.join(checkpoint_dir, f"env_{i}",f"traj_{traj[i]:03d}","semantic_object","front"))
            os.makedirs(os.path.join(checkpoint_dir, f"env_{i}",f"traj_{traj[i]:03d}","semantic_object","side"))
            os.makedirs(os.path.join(checkpoint_dir, f"env_{i}",f"traj_{traj[i]:03d}","semantic_object","bird"))
            os.makedirs(os.path.join(checkpoint_dir, f"env_{i}",f"traj_{traj[i]:03d}","semantic_object","hand"))

                
    
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, rewards, dones, infos = env.step(actions)
            # robot_pos = robot._data.root_state_w[:,:3]
            # robot_ore = robot._data.root_state_w[:,3:7]
            


            # env.step(actions)
            # env.step(actions)

            # cube_data = asset._data.root_link_pose_w

            # cube_changed = subtract_frame_transforms(robot_pos,robot_ore,cube_data[...,:3],cube_data[...,3:7]) #cube wrt robot

            # cube_changed_pos = cube_changed[0]
            # cube_changed_ore = cube_changed[1]
            # import pdb; pdb.set_trace()
            # for _ in range(50):
            #     env.render()
            # import pdb; pdb.set_trace()
            for i in range (env_cfg.scene.num_envs):
                if dones[i]:
                    # import pdb; pdb.set_trace()
                    print(f"Environment {i} done at frame {frame_idx}")
                    pref_log[i]['actions'].pop()  # remove last action corresponding to terminal state that has no observation
                    pref_log[i]['rewards'].pop() # remove last reward as with action
                    np.savez(os.path.join(checkpoint_dir, f"env_{i}",f"traj_{traj[i]:03d}",f"env{i}_traj{traj[i]:03d}.npz"), **pref_log[i])

                    for j in range(env_cfg.scene.num_envs):
                        with ThreadPoolExecutor(max_workers=16) as executor:
                            futures = []
                            for type_cam in ['rgb','depth','semantic_all']:
                                for cam in ['front','side','bird']:
                                    for file_path in save_data[j][type_cam][cam]:
                                        if type_cam == 'depth':
                                            futures.append(executor.submit(
                                                np.savez_compressed,
                                                file_path,
                                                save_data[j][type_cam][cam][file_path])
                                            )
                                        else:
                                            futures.append(executor.submit(
                                                save_images_to_file,
                                                save_data[j][type_cam][cam][file_path],
                                                file_path
                                            ))
                            # Wait for all threads to complete
                            for future in as_completed(futures):
                                future.result()  # This will also raise exceptions if any occurred

                        save_data[j] = {'rgb': {'front': {}, 'side': {}, 'bird': {}},
                                        'depth': {'front': {}, 'side': {}, 'bird': {}},
                                        'semantic_all': {'front': {}, 'side': {}, 'bird': {}}}

                    total_traj -= 1
                        
                    pref_log[i] = {'actions': [], 'rewards': []}
                    traj[i] +=1
                    steps_per_traj[i] = 0
                    # import pdb; pdb.set_trace()
                elif frame_idx > 0: # skip the very first step as there is no action/reward yet
                    pref_log[i]['actions'].append(actions[i].cpu().numpy())
                    pref_log[i]['rewards'].append(rewards[i].cpu().numpy())
                if not os.path.exists(os.path.join(checkpoint_dir, f"env_{i}",f"traj_{traj[i]:03d}")):
                    #trajectory directory
                    os.makedirs(os.path.join(checkpoint_dir, f"env_{i}",f"traj_{traj[i]:03d}"))
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
                    os.makedirs(os.path.join(checkpoint_dir, f"env_{i}",f"traj_{traj[i]:03d}","semantic_robot","hand"))

                    os.makedirs(os.path.join(checkpoint_dir, f"env_{i}",f"traj_{traj[i]:03d}","semantic_object","front"))
                    os.makedirs(os.path.join(checkpoint_dir, f"env_{i}",f"traj_{traj[i]:03d}","semantic_object","side"))
                    os.makedirs(os.path.join(checkpoint_dir, f"env_{i}",f"traj_{traj[i]:03d}","semantic_object","bird"))
                    os.makedirs(os.path.join(checkpoint_dir, f"env_{i}",f"traj_{traj[i]:03d}","semantic_object","hand"))
            
            if total_traj < 0:
                break

            for key, step_pref_log in infos['observations']['preflog'].items():
                for i in range(num_envs):
                    if key not in pref_log[i]:
                        pref_log[i][key] = []
                    pref_log[i][key].append(step_pref_log[i].cpu().numpy())
            # for i in range(env_cfg.scene.num_envs):
            #     if 'front_rgb' not in pref_log[i]:
            #         pref_log[i]['front_rgb'] = []
            #     front = infos['observations']['rgb']['image1'][i].cpu().numpy()
            #     front_image = (np.clip(front, 0, 1) * 255).astype(np.uint8)
            #     pref_log[i]['front_rgb'].append(front_image)

            #trajectory directory
            


            # import pdb; pdb.set_trace() 
            # images = infos['observations']['rgb']['image']
            images1 = infos['observations']['rgb']['image1']
            images2 = infos['observations']['rgb']['image2']
            images3 = infos['observations']['rgb']['image3']
            depth1 = infos['observations']['depth']['image1']
            depth2 = infos['observations']['depth']['image2']
            depth3 = infos['observations']['depth']['image3']
            semantic1 = infos['observations']['semantic']['image1']
            semantic2 = infos['observations']['semantic']['image2']
            semantic3 = infos['observations']['semantic']['image3']
            # semantic_rgb = semantic[:, :, :, :3]
            semantic1_rgb = semantic1[:, :, :, :3]
            semantic2_rgb = semantic2[:, :, :, :3]
            semantic3_rgb = semantic3[:, :, :, :3]
            for i in range(env_cfg.scene.num_envs):
                save_data[i]['rgb']['front'][os.path.join(checkpoint_dir, f"env_{i}",f"traj_{traj[i]:03d}","rgb","front",f"{steps_per_traj[i]:03d}.jpg")] = images1[i]
                save_data[i]['rgb']['side'][os.path.join(checkpoint_dir, f"env_{i}",f"traj_{traj[i]:03d}","rgb","side",f"{steps_per_traj[i]:03d}.jpg")] = images2[i]
                save_data[i]['rgb']['bird'][os.path.join(checkpoint_dir, f"env_{i}",f"traj_{traj[i]:03d}","rgb","bird",f"{steps_per_traj[i]:03d}.jpg")] = images3[i]

                save_data[i]['depth']['front'][os.path.join(checkpoint_dir, f"env_{i}",f"traj_{traj[i]:03d}","depth","front",f"{steps_per_traj[i]:03d}.npz")] = depth1[i]
                save_data[i]['depth']['side'][os.path.join(checkpoint_dir, f"env_{i}",f"traj_{traj[i]:03d}","depth","side",f"{steps_per_traj[i]:03d}.npz")] = depth2[i]
                save_data[i]['depth']['bird'][os.path.join(checkpoint_dir, f"env_{i}",f"traj_{traj[i]:03d}","depth","bird",f"{steps_per_traj[i]:03d}.npz")] = depth3[i]

                save_data[i]['semantic_all']['front'][os.path.join(checkpoint_dir, f"env_{i}",f"traj_{traj[i]:03d}","semantic_all","front",f"{steps_per_traj[i]:03d}.jpg")] = semantic1_rgb[i]/255.0
                save_data[i]['semantic_all']['side'][os.path.join(checkpoint_dir, f"env_{i}",f"traj_{traj[i]:03d}","semantic_all","side",f"{steps_per_traj[i]:03d}.jpg")] = semantic2_rgb[i]/255.0
                save_data[i]['semantic_all']['bird'][os.path.join(checkpoint_dir, f"env_{i}",f"traj_{traj[i]:03d}","semantic_all","bird",f"{steps_per_traj[i]:03d}.jpg")] = semantic3_rgb[i]/255.0
            if frame_idx % 10 == 0 and frame_idx > 0:
                for i in range(env_cfg.scene.num_envs):
                        with ThreadPoolExecutor(max_workers=16) as executor:
                            futures = []
                            for type_cam in ['rgb','depth','semantic_all']:
                                for cam in ['front','side','bird']:
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

                        save_data[i] = {'rgb': {'front': {}, 'side': {}, 'bird': {}},
                                        'depth': {'front': {}, 'side': {}, 'bird': {}},
                                        'semantic_all': {'front': {}, 'side': {}, 'bird': {}}}
    
            #      hard_reset_prim_visibility(f"/World/envs/env_{i}/Object",False) 
            # env.unwrapped.sim.render()
            # # sensor.reset()
            # sensor.update(dt=0, force_recompute=True) 
            # # sensor1.reset()
            # sensor1.update(dt=0, force_recompute=True) 
            # # sensor2.reset()
            # sensor2.update(dt=0, force_recompute=True)
            # # sensor3.reset()
            # sensor3.update(dt=0, force_recompute=True)
            # # import time
            # # time.sleep(0.1)
            # obs,infos = env.get_observations()
            # semantic_robot = infos['observations']['semantic']['image']
            # semantic1_robot = infos['observations']['semantic']['image1']
            # semantic2_robot = infos['observations']['semantic']['image2']
            # semantic3_robot = infos['observations']['semantic']['image3']
            # semantic_rgb_robot = semantic_robot[:, :, :, :3]
            # semantic1_rgb_robot = semantic1_robot[:, :, :, :3]
            # semantic2_rgb_robot = semantic2_robot[:, :, :, :3]
            # semantic3_rgb_robot = semantic3_robot[:, :, :, :3]
            # for i in range(env_cfg.scene.num_envs):
            #         save_images_to_file(semantic_rgb_robot[i].cpu()/255.0,os.path.join(checkpoint_dir, f"env_{i}",f"traj_{traj[i]:03d}","semantic_robot","hand",f"{steps_per_traj[i]:03d}.jpg"))
            #         save_images_to_file(semantic1_rgb_robot[i].cpu()/255.0,os.path.join(checkpoint_dir, f"env_{i}",f"traj_{traj[i]:03d}","semantic_robot","front",f"{steps_per_traj[i]:03d}.jpg"))
            #         save_images_to_file(semantic2_rgb_robot[i].cpu()/255.0,os.path.join(checkpoint_dir, f"env_{i}",f"traj_{traj[i]:03d}","semantic_robot","side",f"{steps_per_traj[i]:03d}.jpg"))
            #         save_images_to_file(semantic3_rgb_robot[i].cpu()/255.0,os.path.join(checkpoint_dir, f"env_{i}",f"traj_{traj[i]:03d}","semantic_robot","bird",f"{steps_per_traj[i]:03d}.jpg"))
            # for i in range(env_cfg.scene.num_envs):
            #      hard_reset_prim_visibility(f"/World/envs/env_{i}/Robot",False)     
            #      hard_reset_prim_visibility(f"/World/envs/env_{i}/Object",True)
            # # import pdb; pdb.set_trace()
            # # env.unwrapped.sim.step()
            # env.unwrapped.sim.render()
            # # pdb.set_trace()
            # # env.unwrapped.render()
            # # sensor.reset()
            # sensor.update(dt=0, force_recompute=True) 
            # # sensor1.reset()
            # sensor1.update(dt=0, force_recompute=True) 
            # # sensor2.reset()
            # sensor2.update(dt=0, force_recompute=True)
            # # sensor3.reset()
            # sensor3.update(dt=0, force_recompute=True)
            # obs,infos = env.get_observations()
            # semantic_object = infos['observations']['semantic']['image']
            # semantic1_object = infos['observations']['semantic']['image1']
            # semantic2_object = infos['observations']['semantic']['image2']
            # semantic3_object = infos['observations']['semantic']['image3']
            # semantic_rgb_object = semantic_object[:, :, :, :3]
            # semantic1_rgb_object = semantic1_object[:, :, :, :3]
            # semantic2_rgb_object = semantic2_object[:, :, :, :3]
            # semantic3_rgb_object = semantic3_object[:, :, :, :3]
            # for i in range(env_cfg.scene.num_envs):
            #         save_images_to_file(semantic_rgb_object[i].cpu()/255.0,os.path.join(checkpoint_dir, f"env_{i}",f"traj_{traj[i]:03d}","semantic_object","hand",f"{steps_per_traj[i]:03d}.jpg"))
            #         save_images_to_file(semantic1_rgb_object[i].cpu()/255.0,os.path.join(checkpoint_dir, f"env_{i}",f"traj_{traj[i]:03d}","semantic_object","front",f"{steps_per_traj[i]:03d}.jpg"))
            #         save_images_to_file(semantic2_rgb_object[i].cpu()/255.0,os.path.join(checkpoint_dir, f"env_{i}",f"traj_{traj[i]:03d}","semantic_object","side",f"{steps_per_traj[i]:03d}.jpg"))
            #         save_images_to_file(semantic3_rgb_object[i].cpu()/255.0,os.path.join(checkpoint_dir, f"env_{i}",f"traj_{traj[i]:03d}","semantic_object","bird",f"{steps_per_traj[i]:03d}.jpg"))

            # # # Append row to CSV
            # # with open(csv_file, mode="a", newline="") as f:
            # #     writer = csv.writer(f)
            # #     for i in range(env_cfg.scene.num_envs):
            # #         save_images_to_file(images[i].cpu()/255.0,f"frames/front/rgb_env{i}_{frame_idx}.jpg")
            # #         save_images_to_file(images1[i].cpu()/255.0,f"frames/side/rgb_env{i}_{frame_idx}.jpg")
            # #         save_images_to_file(images2[i].cpu()/255.0,f"frames/bird/rgb_env{i}_{frame_idx}.jpg")
            # #         # Save depth maps as compressed .npz
            # #         writer.writerow([frame_idx, i, cube_data[i][:3].cpu().numpy(), cube_data[i][3:7].cpu().numpy(),robot._data.root_state_w[i][:3].cpu().numpy(),robot._data.root_state_w[i][3:7].cpu().numpy(),cube_changed_pos[i].cpu().numpy(),cube_changed_ore[i].cpu().numpy(),
            # #                          f"frames/front/rgb_env{i}_{frame_idx}.jpg",f"frames/side/rgb_env{i}_{frame_idx}.jpg",f"frames/bird/rgb_env{i}_{frame_idx}.jpg",f"frames/front/depth_{frame_idx}.npz",f"frames/side/depth_{frame_idx}.npz",f"frames/bird/depth_{frame_idx}.npz",
            # #                          env.unwrapped.scene.env_origins[i].cpu().numpy(),f"frames/front/semantic_env{i}_{frame_idx}.jpg",f"frames/side/semantic_env{i}_{frame_idx}.jpg",f"frames/bird/semantic_env{i}_{frame_idx}.jpg",f"frames/front/semantic_object_only_env{i}_{frame_idx}.jpg",
            # #                          f"frames/side/semantic_object_only_env{i}_{frame_idx}.jpg",f"frames/bird/semantic_object_only_env{i}_{frame_idx}.jpg",f"frames/front/semantic_object_only_env{i}_{frame_idx}.jpg",f"frames/side/semantic_object_only_env{i}_{frame_idx}.jpg",f"frames/bird/semantic_object_only_env{i}_{frame_idx}.jpg"])
            # for i in range(env_cfg.scene.num_envs):
            #      hard_reset_prim_visibility(f"/World/envs/env_{i}/Robot",True)     
            #      hard_reset_prim_visibility(f"/World/envs/env_{i}/Object",True) 
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

def hard_reset_prim_visibility(prim_path, visible=True):
    stage = omni.usd.get_context().get_stage()
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



if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
