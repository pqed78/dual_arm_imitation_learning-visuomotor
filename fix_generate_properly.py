import os

script_path = "scripts/generate_scripted_demos.py"
with open(script_path, "r") as f:
    code = f.read()

# 1. Init arrays
target_init = """        ep_obs = []
        ep_images = []
        ep_actions = []
        ep_rewards = []"""
replacement_init = """        ep_obs = []
        ep_images = []
        ep_actions = []
        ep_rewards = []
        object_poses = []
        robot_joint_poses = []
        init_robot_pos = robot.data.root_pos_w.cpu().numpy()[0].copy()
        init_robot_quat = robot.data.root_quat_w.cpu().numpy()[0].copy()
        init_target_pos = target.data.root_pos_w.cpu().numpy()[0].copy()
        init_target_quat = target.data.root_quat_w.cpu().numpy()[0].copy()"""
code = code.replace(target_init, replacement_init)

# 2. Append arrays
target_append = """            ep_obs.append(policy_obs)
            ep_images.append(rgb_image)
            ep_actions.append(action_np)"""
replacement_append = """            ep_obs.append(policy_obs)
            ep_images.append(rgb_image)
            ep_actions.append(action_np)
            object_poses.append(obj.data.root_state_w.cpu().numpy()[0])
            robot_joint_poses.append(robot.data.joint_pos.cpu().numpy()[0])"""
code = code.replace(target_append, replacement_append)

with open(script_path, "w") as f:
    f.write(code)
