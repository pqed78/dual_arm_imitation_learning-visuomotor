import os
import glob

for script_path in ["scripts/replay_demos.py", "scripts/generate_scripted_demos.py", "teleop/collect_demos.py"]:
    if not os.path.exists(script_path):
        continue
    with open(script_path, "r") as f:
        code = f.read()

    # Replay
    if "replay" in script_path:
        # Remove debug print
        target0 = """                print(f"DEBUG: rgb_data shape: {rgb_data.shape}, dtype: {rgb_data.dtype}, max: {rgb_data.max().item()}")"""
        code = code.replace(target0, "")
        
        target1 = """                if rgb_np.dtype != np.uint8:
                    rgb_np = rgb_np.astype(np.uint8)"""
        replacement1 = """                if rgb_np.dtype != np.uint8:
                    if rgb_np.max() <= 1.0:
                        rgb_np = (rgb_np * 255.0)
                    rgb_np = np.clip(rgb_np, 0, 255).astype(np.uint8)"""
        code = code.replace(target1, replacement1)

    # Generate
    if "generate" in script_path:
        target2 = """            if rgb_image.dtype != np.uint8:
                rgb_image = rgb_image.astype(np.uint8)"""
        replacement2 = """            if rgb_image.dtype != np.uint8:
                if rgb_image.max() <= 1.0:
                    rgb_image = (rgb_image * 255.0)
                rgb_image = np.clip(rgb_image, 0, 255).astype(np.uint8)"""
        code = code.replace(target2, replacement2)

    # Collect
    if "collect" in script_path:
        target3 = """            if rgb_image.dtype != np.uint8:
                rgb_image = rgb_image.astype(np.uint8)"""
        replacement3 = """            if rgb_image.dtype != np.uint8:
                if rgb_image.max() <= 1.0:
                    rgb_image = (rgb_image * 255.0)
                rgb_image = np.clip(rgb_image, 0, 255).astype(np.uint8)"""
        code = code.replace(target3, replacement3)

    with open(script_path, "w") as f:
        f.write(code)
print("RGB fix applied.")
