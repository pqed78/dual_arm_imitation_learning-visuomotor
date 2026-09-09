import os

script_path = "scripts/replay_demos.py"
with open(script_path, "r") as f:
    code = f.read()

target = """    gym.register(
        id="Isaac-Dual-Arm-IL-v0","""
replacement = """    try:
        gym.register(
            id="Isaac-Dual-Arm-IL-v0",
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={"env_cfg_entry_point": env_cfg.__class__},
        )
    except Exception:
        pass
    # Dummy block to match target replace length if needed
    if True:"""

# Just doing a string replace for the whole register block
target = """    gym.register(
        id="Isaac-Dual-Arm-IL-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={"env_cfg_entry_point": env_cfg.__class__},
    )"""

replacement = """    try:
        gym.register(
            id="Isaac-Dual-Arm-IL-v0",
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={"env_cfg_entry_point": env_cfg.__class__},
        )
    except Exception:
        pass"""

code = code.replace(target, replacement)

with open(script_path, "w") as f:
    f.write(code)
