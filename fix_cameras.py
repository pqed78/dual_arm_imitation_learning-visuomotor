import os

script_path = "scripts/replay_demos.py"
with open(script_path, "r") as f:
    code = f.read()

target = """args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)"""
replacement = """args_cli = parser.parse_args()
args_cli.enable_cameras = True  # Force enable cameras for rendering

app_launcher = AppLauncher(args_cli)"""

code = code.replace(target, replacement)

with open(script_path, "w") as f:
    f.write(code)
print("Forced --enable_cameras.")
