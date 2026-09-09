import os

script_path = "scripts/replay_demos.py"
with open(script_path, "r") as f:
    code = f.read()

target1 = """from isaaclab.envs import ManagerBasedRLEnv"""
replacement1 = """from isaaclab.envs import ManagerBasedRLEnv
try:
    from dual_arm_il.configs.env_cfg import VisuomotorObsCfg
except ModuleNotFoundError:
    from configs.env_cfg import VisuomotorObsCfg"""

target2 = """    env_cfg.scene.num_envs = num_parallel

    env: ManagerBasedRLEnv = gym.make("Isaac-Dual-Arm-v0", cfg=env_cfg).unwrapped"""
replacement2 = """    env_cfg.scene.num_envs = num_parallel
    env_cfg.observations = VisuomotorObsCfg()

    gym.register(
        id="Isaac-Dual-Arm-IL-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={"env_cfg_entry_point": env_cfg.__class__},
    )
    env: ManagerBasedRLEnv = gym.make("Isaac-Dual-Arm-IL-v0", cfg=env_cfg).unwrapped"""

code = code.replace(target1, replacement1)
code = code.replace(target2, replacement2)

with open(script_path, "w") as f:
    f.write(code)
print("Env name fixed.")
