import os
import re

script_path = "scripts/generate_scripted_demos.py"
with open(script_path, "r") as f:
    code = f.read()

# 1. Update save_episode_to_hdf5 signature
code = code.replace(
    "def save_episode_to_hdf5(hdf5_path: str, ep_idx: int, observations: list, images: list, actions: list, rewards: list):",
    "def save_episode_to_hdf5(hdf5_path: str, ep_idx: int, observations: list, images: list, actions: list, rewards: list, object_poses: list, robot_joint_poses: list, init_robot_pos: list, init_robot_quat: list, init_target_pos: list, init_target_quat: list):"
)

# 2. Add saving logic to save_episode_to_hdf5
save_logic = """        demo_group.create_dataset("rewards", data=rew_array, compression="gzip")
        demo_group.create_dataset("object_poses", data=np.array(object_poses, dtype=np.float32), compression="gzip")
        demo_group.create_dataset("robot_joint_poses", data=np.array(robot_joint_poses, dtype=np.float32), compression="gzip")
        demo_group.create_dataset("init_robot_pos", data=np.array(init_robot_pos, dtype=np.float32))
        demo_group.create_dataset("init_robot_quat", data=np.array(init_robot_quat, dtype=np.float32))
        demo_group.create_dataset("init_target_pos", data=np.array(init_target_pos, dtype=np.float32))
        demo_group.create_dataset("init_target_quat", data=np.array(init_target_quat, dtype=np.float32))
        demo_group.attrs["num_samples"] = len(act_array)"""
code = code.replace(
    '        demo_group.create_dataset("rewards", data=rew_array, compression="gzip")\n        demo_group.attrs["num_samples"] = len(act_array)',
    save_logic
)

# 3. Initialize tracking arrays in main loop
init_arrays = """        observations = []
        images = []
        actions = []
        rewards = []
        object_poses = []
        robot_joint_poses = []
        init_robot_pos = robot.data.root_pos_w.cpu().numpy()[0].copy()
        init_robot_quat = robot.data.root_quat_w.cpu().numpy()[0].copy()
        init_target_pos = target.data.root_pos_w.cpu().numpy()[0].copy()
        init_target_quat = target.data.root_quat_w.cpu().numpy()[0].copy()"""
code = code.replace(
    "        observations = []\n        images = []\n        actions = []\n        rewards = []",
    init_arrays
)

# 4. Append to tracking arrays in step loop
append_logic = """            actions.append(joint_actions[0].cpu().numpy())
            object_poses.append(obj.data.root_state_w.cpu().numpy()[0])
            robot_joint_poses.append(robot.data.joint_pos.cpu().numpy()[0])"""
code = code.replace(
    "            actions.append(joint_actions[0].cpu().numpy())",
    append_logic
)

# 5. Call save_episode_to_hdf5 with new arguments
code = code.replace(
    "save_episode_to_hdf5(args_cli.dataset, ep_idx, observations, images, actions, rewards)",
    "save_episode_to_hdf5(args_cli.dataset, ep_idx, observations, images, actions, rewards, object_poses, robot_joint_poses, init_robot_pos, init_robot_quat, init_target_pos, init_target_quat)"
)

with open(script_path, "w") as f:
    f.write(code)
print("generate_scripted_demos.py patched.")
