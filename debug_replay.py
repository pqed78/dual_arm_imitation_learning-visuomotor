import os

script_path = "scripts/replay_demos.py"
with open(script_path, "r") as f:
    code = f.read()

target = """        # Display camera view
        if "image" in obs_dict and "rgb" in obs_dict["image"]:"""
replacement = """        # Display camera view
        if "image" not in obs_dict:
            print(f"DEBUG: 'image' not in obs_dict. Keys are: {list(obs_dict.keys())}")
        if "image" in obs_dict and "rgb" in obs_dict["image"]:"""

code = code.replace(target, replacement)

# Add namedWindow before the loop
target2 = """    env.reset()
    
    print(f"\\n--- Starting KINEMATIC parallel replay (Max steps: {max_length}) ---")"""
replacement2 = """    env.reset()
    
    cv2.namedWindow("Visuomotor Replay - Camera View", cv2.WINDOW_NORMAL)
    
    print(f"\\n--- Starting KINEMATIC parallel replay (Max steps: {max_length}) ---")"""

code = code.replace(target2, replacement2)

with open(script_path, "w") as f:
    f.write(code)
print("Debug statements added.")
