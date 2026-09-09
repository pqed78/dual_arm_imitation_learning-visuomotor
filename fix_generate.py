import os

script_path = "scripts/generate_scripted_demos.py"
with open(script_path, "r") as f:
    code = f.read()

target = """                    save_episode_to_hdf5(
                        args_cli.dataset_file,
                        collected_count,
                        ep_obs,
                        ep_images,
                        ep_actions,
                        ep_rewards,
                    )"""
replacement = """                    save_episode_to_hdf5(
                        args_cli.dataset_file,
                        collected_count,
                        ep_obs,
                        ep_images,
                        ep_actions,
                        ep_rewards,
                        object_poses,
                        robot_joint_poses,
                        init_robot_pos,
                        init_robot_quat,
                        init_target_pos,
                        init_target_quat,
                    )"""

code = code.replace(target, replacement)
with open(script_path, "w") as f:
    f.write(code)
