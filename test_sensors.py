import os
import sys

script_path = "scripts/replay_demos.py"
with open(script_path, "r") as f:
    code = f.read()

# Replace the condition
target = "if \"front_camera\" in env.scene.sensors:"
replacement = "if \"front_camera\" in env.scene.keys():"
code = code.replace(target, replacement)

with open(script_path, "w") as f:
    f.write(code)
print("Condition patched.")
