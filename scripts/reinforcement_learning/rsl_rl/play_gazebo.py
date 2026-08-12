# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""
"""
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_gazebo.py --task=Isaac-Lift-Cube-Franka-Camera-v0 --num_envs=1 --enable_cameras --load_run /home/sh-d61-cps-hri/hri-pl-frm-mvvd/isaaclab/logs/rsl_rl/franka_lift/2025-09-12_13-48-36 --checkpoint /home/sh-d61-cps-hri/hri-pl-frm-mvvd/isaaclab/logs/rsl_rl/franka_lift/2025-09-12_13-48-36/model_1050.pt --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import csv
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

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

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

# PLACEHOLDER: Extension template (do not remove this comment)
import zmq
import numpy as np
import pickle

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
    from isaaclab.sensors import save_images_to_file, depth_to_rgba
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
        policy_nn, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="1050.onnx"
    )
    # sensor = env.unwrapped.scene["camera_ext1"] #front
    robot = env.unwrapped.scene["robot"]
    dt = env.unwrapped.step_dt
    cube = env.unwrapped.scene["object"]
    cube.write_root_pose_to_sim(torch.tensor([0.5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]))
    # reset environment
    action = np.zeros(8)  
    
    context = zmq.Context()
    socket = context.socket(zmq.REP)  # Request socket (sending requests)
    socket.bind("tcp://localhost:5555")  # Connect to the server
    timestep = 0
    # simulate environment
    frame =0
    obs, _ = env.get_observations()
    csv_file = "/home/sh-d61-cps-hri/hri-pl-frm-mvvd/obs_data_isaac_real.csv"
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["joint_pos_1","joint_pos_2","joint_pos_3","joint_pos_4","joint_pos_5","joint_pos_6","joint_pos_7","joint_pos_8","joint_pos_9","joint_vel_1","joint_vel_2","joint_vel_3","joint_vel_4","joint_vel_5","joint_vel_6","joint_vel_7","joint_vel_8","joint_vel_9"])  # header row
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            processed_data = socket.recv()  # Blocking until response is received
            processed_obs = pickle.loads(processed_data)

            # convert received numpy observation to a torch tensor on the env device and add batch dim
            if isinstance(processed_obs, np.ndarray):
                processed_action = torch.from_numpy(processed_obs).to(device=env.unwrapped.device, dtype=torch.float32).unsqueeze(0)
            else:
                processed_action = torch.tensor(processed_obs, device=env.unwrapped.device, dtype=torch.float32).unsqueeze(0)
            
            # obs in shape of tensor in tensor [[]]
            obs[0][0:9] = processed_action[0][0:9]
            # obs[0][18:] = processed_action[0][18:]
            # obs[0] = processed_action[0]
            # scale = torch.tensor([60, 55, 25, 41, 115, 103, 88, 4.5, 9], 
            #          device=obs.device, dtype=torch.float32)
            # obs[0][9:18] = obs[0][9:18] * scale
            # # obs[0][10] = jv
            # # obs[0][8:10] = grip


            # record observations to csv file
            # with open(csv_file, mode="a", newline="") as f:
            #     writer = csv.writer(f)
            #     writer.writerow([obs[0][0].item(), obs[0][1].item(), obs[0][2].item(), obs[0][3].item(), obs[0][4].item(), obs[0][5].item(), obs[0][6].item(), obs[0][7].item(), obs[0][8].item(),
            #                      obs[0][9].item(), obs[0][10].item(), obs[0][11].item(), obs[0][12].item(), obs[0][13].item(), obs[0][14].item(), obs[0][15].item(), obs[0][16].item(), obs[0][17].item()])  # header row
            
            
            actions = policy(obs)
            
            # env stepping
            obs, _, dones, infos = env.step(actions)
            joint_pos = robot._data.joint_pos.cpu().numpy()
            action_to_send = actions.cpu().numpy()

            serialized_data = pickle.dumps((joint_pos, action_to_send))
            # print(f"Joint positions: {joint_pos}")
            # Send data to the server
            socket.send(serialized_data)
            frame +=1


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


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
