# Copyright (c) 2026, Dual Arm Imitation Learning Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Replay Demonstrations Script (Kinematic / Visuomotor).

Replays recorded demonstration trajectories in Isaac Sim kinematically
to visually inspect and verify motion quality and task success.
"""

import argparse
import os
import sys
import time
import h5py
import torch
import cv2
import numpy as np

from isaaclab.app import AppLauncher

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT_ROOT = os.path.dirname(PROJECT_ROOT)
for path in [PROJECT_ROOT, PARENT_ROOT]:
    if path not in sys.path:
        sys.path.insert(0, path)

parser = argparse.ArgumentParser(description="Kinematic Replay of multiple demonstrations.")
parser.add_argument(
    "--dataset",
    type=str,
    default=os.path.join(PROJECT_ROOT, "data", "demos.hdf5"),
    help="Path to HDF5 dataset file.",
)
parser.add_argument("--demo_idx", type=int, default=0, help="Starting index of demo to replay. Negative indices supported.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of demos to play simultaneously.")
parser.add_argument("--delay", type=float, default=0.033, help="Delay between frames in seconds.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True  # Force enable cameras for rendering

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from isaaclab.envs import ManagerBasedRLEnv
try:
    from configs.env_cfg import VisuomotorObsCfg
except ImportError:
    from dual_arm_il.configs.env_cfg import VisuomotorObsCfg
try:
    from configs.env_cfg import DualArmILEnvCfg
except ImportError:
    from dual_arm_il.configs.env_cfg import DualArmILEnvCfg

def main():
    if not os.path.exists(args_cli.dataset):
        raise FileNotFoundError(f"Dataset not found: {args_cli.dataset}")

    with h5py.File(args_cli.dataset, "r") as f:
        data_grp = f["data"]
        demo_keys = sorted([k for k in data_grp.keys() if k.startswith("demo_")], key=lambda x: int(x.split("_")[1]))

        if len(demo_keys) == 0:
            print("[Replay] No demonstrations found in dataset!")
            simulation_app.close()
            return
            
        num_avail = len(demo_keys)
        num_parallel = min(args_cli.num_envs, num_avail)
        
        if args_cli.demo_idx < 0:
            # If negative (e.g. -1), pick the last num_parallel demos
            start_idx = max(0, num_avail + args_cli.demo_idx - num_parallel + 1)
        else:
            start_idx = min(args_cli.demo_idx, num_avail - 1)
            
        end_idx = min(num_avail, start_idx + num_parallel)
        target_demos = demo_keys[start_idx : end_idx]
        num_parallel = len(target_demos)  # update to actual
        
        print(f"[Kinematic Replay] Preparing to play {num_parallel} demos in parallel: {target_demos}")

        # Check if kinematic data is available
        if "object_poses" not in data_grp[target_demos[0]]:
            print("ERROR: Dataset does not contain 'object_poses' and 'robot_joint_poses'.")
            print("Please regenerate the dataset using the updated generate_scripted_demos.py script!")
            simulation_app.close()
            return

        all_obj_traj = []
        all_joint_traj = []
        all_init_robot_pos = []
        all_init_robot_quat = []
        all_init_target_pos = []
        all_init_target_quat = []
        max_length = 0
        
        for key in target_demos:
            obj_traj = data_grp[key]["object_poses"][:]
            joint_traj = data_grp[key]["robot_joint_poses"][:]
            all_obj_traj.append(obj_traj)
            all_joint_traj.append(joint_traj)
            if len(obj_traj) > max_length:
                max_length = len(obj_traj)
                
            if "init_robot_pos" in data_grp[key]:
                all_init_robot_pos.append(data_grp[key]["init_robot_pos"][:])
                all_init_robot_quat.append(data_grp[key]["init_robot_quat"][:])
            else:
                all_init_robot_pos.append(None)
                all_init_robot_quat.append(None)
                
            if "init_target_pos" in data_grp[key]:
                all_init_target_pos.append(data_grp[key]["init_target_pos"][:])
                all_init_target_quat.append(data_grp[key]["init_target_quat"][:])
            else:
                all_init_target_pos.append(None)
                all_init_target_quat.append(None)

    env_cfg = DualArmILEnvCfg()
    env_cfg.sim.device = args_cli.device
    
    # -------------------------------------------------------------
    # Viewer & Recording Setup (Fixing Cropped FOV)
    # -------------------------------------------------------------
    env_cfg.viewer.resolution = (1920, 1080)
    if num_parallel > 1:
        # Pull the camera back and up to capture all parallel environments
        env_cfg.viewer.eye = (2.5 + num_parallel * 0.2, 0.0, 2.0 + num_parallel * 0.1)
        env_cfg.viewer.lookat = (0.0, 0.0, 0.0)
    
    # FOR KINEMATIC REPLAY: Disable physics on the object so it doesn't fall or get pushed
    if hasattr(env_cfg.scene, "object") and hasattr(env_cfg.scene.object, "spawn"):
        from isaaclab.sim import RigidBodyPropertiesCfg
        env_cfg.scene.object.spawn.rigid_props = RigidBodyPropertiesCfg(
            kinematic_enabled=True,
            disable_gravity=True,
        )
    env_cfg.scene.num_envs = num_parallel
    env_cfg.observations = VisuomotorObsCfg()

    try:
        gym.register(
            id="Isaac-Dual-Arm-IL-v0",
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={"env_cfg_entry_point": env_cfg.__class__},
        )
    except Exception:
        pass
    env: ManagerBasedRLEnv = gym.make("Isaac-Dual-Arm-IL-v0", cfg=env_cfg).unwrapped
    env.reset()
    
    # Setup Video Writer
    video_out = None
    
    print(f"\n--- Starting KINEMATIC parallel replay (Max steps: {max_length}) ---")
    
    obj_state = env.scene["object"].data.default_root_state.clone()
    tgt_state = env.scene["target"].data.default_root_state.clone()
    rob_state = env.scene["robot"].data.default_root_state.clone()
    j_pos = env.scene["robot"].data.default_joint_pos.clone()
    j_vel = env.scene["robot"].data.default_joint_vel.clone() * 0.0
    
    for step_idx in range(max_length):
        if not simulation_app.is_running():
            break

        for i in range(num_parallel):
            o_traj = all_obj_traj[i]
            j_traj = all_joint_traj[i]
            idx = min(step_idx, len(o_traj) - 1)
            
            obj_state[i, :3] = torch.tensor(o_traj[idx, :3], device=env.device) + env.scene.env_origins[i]
            obj_state[i, 3:7] = torch.tensor(o_traj[idx, 3:7], device=env.device)
            j_pos[i] = torch.tensor(j_traj[idx], device=env.device)
            
            if all_init_target_pos[i] is not None:
                tgt_state[i, :3] = torch.tensor(all_init_target_pos[i], device=env.device) + env.scene.env_origins[i]
                tgt_state[i, 3:7] = torch.tensor(all_init_target_quat[i], device=env.device)
                
            if all_init_robot_pos[i] is not None:
                rob_state[i, :3] = torch.tensor(all_init_robot_pos[i], device=env.device) + env.scene.env_origins[i]
                rob_state[i, 3:7] = torch.tensor(all_init_robot_quat[i], device=env.device)
            
        env.scene["object"].write_root_state_to_sim(obj_state)
        env.scene["target"].write_root_state_to_sim(tgt_state)
        env.scene["robot"].write_root_state_to_sim(rob_state)
        env.scene["robot"].write_joint_state_to_sim(j_pos, j_vel)
        
        env.sim.step()
        
        # Update scene and compute observations to render camera
        env.scene.update(dt=env.physics_dt)
        obs_dict = env.observation_manager.compute()
        
        # Display camera view
        if "image" not in obs_dict:
            print(f"DEBUG: 'image' not in obs_dict. Keys are: {list(obs_dict.keys())}")
        if "image" in obs_dict and "rgb" in obs_dict["image"]:
            rgb_data = obs_dict["image"]["rgb"]
            if rgb_data is not None:

                rgb_np = rgb_data.clone().detach().cpu().numpy()
                if rgb_np.dtype != np.uint8:
                    if rgb_np.max() <= 1.0:
                        rgb_np = (rgb_np * 255.0)
                    rgb_np = np.clip(rgb_np, 0, 255).astype(np.uint8)
                
                # Stack images horizontally
                grid_img = np.concatenate(rgb_np[:num_parallel], axis=1)
                
                # Add camera position text to the first image
                cam_pos = env.scene["front_camera"].data.pos_w[0].cpu().numpy()
                cam_text = f"Cam Pos: [{cam_pos[0]:.2f}, {cam_pos[1]:.2f}, {cam_pos[2]:.2f}]"
                
                # Convert to BGR for OpenCV
                if grid_img.shape[-1] == 3:
                    grid_img = cv2.cvtColor(grid_img, cv2.COLOR_RGB2BGR)
                elif grid_img.shape[-1] == 4:
                    grid_img = cv2.cvtColor(grid_img, cv2.COLOR_RGBA2BGR)
                    
                cv2.putText(grid_img, cam_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Write to video
                if video_out is None:
                    h, w = grid_img.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_out = cv2.VideoWriter('replay_video.mp4', fourcc, 30.0, (w, h))
                video_out.write(grid_img)
        
        if args_cli.delay > 0:
            time.sleep(args_cli.delay)

    print("Finished kinematic replay.")
    time.sleep(2.0)
    if video_out is not None:
        video_out.release()
        print("Saved replay video to replay_video.mp4")
    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()
