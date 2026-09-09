import os
script_path = "scripts/replay_demos.py"
with open(script_path, "r") as f:
    code = f.read()

target = "if rgb_data is not None:"
replacement = """if rgb_data is not None:
                print(f"DEBUG: rgb_data shape: {rgb_data.shape}, dtype: {rgb_data.dtype}, max: {rgb_data.max().item()}")"""

code = code.replace(target, replacement)
with open(script_path, "w") as f:
    f.write(code)
print("Added shape print.")
