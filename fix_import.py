import os

script_path = "scripts/replay_demos.py"
with open(script_path, "r") as f:
    code = f.read()

target1 = """try:
    from dual_arm_il.configs.env_cfg import VisuomotorObsCfg
except ModuleNotFoundError:
    from configs.env_cfg import VisuomotorObsCfg"""
replacement1 = """try:
    from configs.env_cfg import VisuomotorObsCfg
except ImportError:
    from dual_arm_il.configs.env_cfg import VisuomotorObsCfg"""

target2 = """try:
    from dual_arm_il.configs.env_cfg import DualArmILEnvCfg
except ModuleNotFoundError:
    from configs.env_cfg import DualArmILEnvCfg"""
replacement2 = """try:
    from configs.env_cfg import DualArmILEnvCfg
except ImportError:
    from dual_arm_il.configs.env_cfg import DualArmILEnvCfg"""

code = code.replace(target1, replacement1)
code = code.replace(target2, replacement2)

with open(script_path, "w") as f:
    f.write(code)
print("Import fixed.")
