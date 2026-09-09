import os

script_path = "teleop/collect_demos.py"
with open(script_path, "r") as f:
    code = f.read()

# 1. Update save_episode_to_hdf5 signature
code = code.replace(
    "def save_episode_to_hdf5(hdf5_path: str, ep_idx: int, observations: list, actions: list, rewards: list):",
    "def save_episode_to_hdf5(hdf5_path: str, ep_idx: int, observations: list, actions: list, rewards: list, object_poses: list, robot_joint_poses: list, init_robot_pos: list, init_robot_quat: list, init_target_pos: list, init_target_quat: list):"
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
init_arrays = """        ep_obs = []
        ep_actions = []
        ep_rewards = []
        object_poses = []
        robot_joint_poses = []
        init_robot_pos = robot.data.root_pos_w.cpu().numpy()[0].copy()
        init_robot_quat = robot.data.root_quat_w.cpu().numpy()[0].copy()
        init_target_pos = target.data.root_pos_w.cpu().numpy()[0].copy()
        init_target_quat = target.data.root_quat_w.cpu().numpy()[0].copy()"""
code = code.replace(
    "        ep_obs = []\n        ep_actions = []\n        ep_rewards = []",
    init_arrays
)

# 4. Append to tracking arrays in step loop
append_logic = """            ep_actions.append(action_np)
            object_poses.append(obj.data.root_state_w.cpu().numpy()[0])
            robot_joint_poses.append(robot.data.joint_pos.cpu().numpy()[0])"""
code = code.replace(
    "            ep_actions.append(action_np)",
    append_logic
)

# 5. Call save_episode_to_hdf5 with new arguments
code = code.replace(
    """                save_episode_to_hdf5(
                    args_cli.dataset_file,
                    collected_count,
                    ep_obs,
                    ep_actions,
                    ep_rewards,
                )""",
    """                save_episode_to_hdf5(
                    args_cli.dataset_file,
                    collected_count,
                    ep_obs,
                    ep_actions,
                    ep_rewards,
                    object_poses,
                    robot_joint_poses,
                    init_robot_pos,
                    init_robot_quat,
                    init_target_pos,
                    init_target_quat,
                )"""
)

# Fix missing env.scene accesses
code = code.replace(
    "env.scene[\"robot\"]",
    "robot"
)
# Add obj and target
code = code.replace(
    "robot = env.scene[\"robot\"]",
    "robot = env.scene[\"robot\"]\n    obj = env.scene[\"object\"]\n    target = env.scene[\"target\"]"
)

with open(script_path, "w") as f:
    f.write(code)
print("collect_demos.py patched.")
