# Copyright (c) 2026, Dual Arm Imitation Learning Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Replay Demonstrations Script.

Replays recorded demonstration trajectories in Isaac Sim to visually
inspect and verify motion quality and task success.

Usage:
    python scripts/replay_demos.py --dataset=data/demos.hdf5 --demo_idx=0
"""

import argparse
import os
import sys
import time
import h5py
import torch

from isaaclab.app import AppLauncher

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT_ROOT = os.path.dirname(PROJECT_ROOT)
for path in [PROJECT_ROOT, PARENT_ROOT]:
    if path not in sys.path:
        sys.path.insert(0, path)

parser = argparse.ArgumentParser(description="Replay recorded demonstrations.")
parser.add_argument(
    "--dataset",
    type=str,
    default=os.path.join(PROJECT_ROOT, "data", "demos.hdf5"),
    help="Path to HDF5 dataset file.",
)
parser.add_argument("--demo_idx", type=int, default=0, help="Demo index to replay (or -1 for all).")
parser.add_argument("--num_envs", type=int, default=1, help="Number of demos to replay simultaneously.")
parser.add_argument("--delay", type=float, default=0.02, help="Delay between steps in seconds.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from isaaclab.envs import ManagerBasedRLEnv
try:
    from dual_arm_il.configs.env_cfg import DualArmILEnvCfg
except ModuleNotFoundError:
    from configs.env_cfg import DualArmILEnvCfg

def replay_batch_demos(env: ManagerBasedRLEnv, actions_list: list, demo_names: list, delay: float):
    print(f"\n--- Replaying {len(demo_names)} demos simultaneously: {demo_names[0]} ~ {demo_names[-1]} ---")
    env.reset()

    max_steps = max(len(acts) for acts in actions_list)
    num_envs = len(actions_list)

    for step_idx in range(max_steps):
        if not simulation_app.is_running():
            break
        
        # Build batch action tensor
        batch_acts = []
        for env_idx in range(num_envs):
            acts = actions_list[env_idx]
            if step_idx < len(acts):
                batch_acts.append(acts[step_idx])
            else:
                # Repeat last action if this demo finished early
                batch_acts.append(acts[-1])
                
        act_t = torch.tensor(batch_acts, dtype=torch.float32, device=args_cli.device)
        env.step(act_t)

        if delay > 0:
            time.sleep(delay)

    print(f"Finished replaying {len(demo_names)} demos.")


def main():
    if not os.path.exists(args_cli.dataset):
        raise FileNotFoundError(f"Dataset not found: {args_cli.dataset}")

    env_cfg = DualArmILEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    env: ManagerBasedRLEnv = gym.make("Isaac-Dual-Arm-v0", cfg=env_cfg).unwrapped

    with h5py.File(args_cli.dataset, "r") as f:
        data_grp = f["data"]
        demo_keys = sorted([k for k in data_grp.keys() if k.startswith("demo_")], key=lambda x: int(x.split("_")[1]))

        if len(demo_keys) == 0:
            print("[Replay] No demonstrations found in dataset!")
            env.close()
            simulation_app.close()
            return

        if args_cli.demo_idx >= 0:
            target_demos = [f"demo_{args_cli.demo_idx}"]
        else:
            target_demos = demo_keys
            
        # Group demos into batches of size num_envs
        for i in range(0, len(target_demos), args_cli.num_envs):
            batch_keys = target_demos[i : i + args_cli.num_envs]
            
            # If the last batch is smaller than num_envs, we must stop, 
            # because the environment was instantiated with a fixed num_envs.
            if len(batch_keys) < args_cli.num_envs:
                # Pad with the first demo in the batch to match num_envs
                print(f"Padding last batch to match num_envs={args_cli.num_envs}")
                while len(batch_keys) < args_cli.num_envs:
                    batch_keys.append(batch_keys[0])

            actions_list = [data_grp[key]["actions"][:] for key in batch_keys]
            replay_batch_demos(env, actions_list, batch_keys, args_cli.delay)
            time.sleep(1.0)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
