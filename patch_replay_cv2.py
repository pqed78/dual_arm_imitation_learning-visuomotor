import os

script_path = "scripts/replay_demos.py"
with open(script_path, "r") as f:
    code = f.read()

# Add cv2 and numpy imports
if "import cv2" not in code:
    code = code.replace("import torch", "import torch\nimport cv2\nimport numpy as np")

# Find the simulation step loop
target_step = "env.sim.step()"
replacement = """env.sim.step()
        
        # Update scene to render sensors
        env.scene.update(dt=env.physics_dt)
        
        # Display camera view
        if "front_camera" in env.scene.sensors:
            rgb_data = env.scene["front_camera"].data.output["rgb"]
            if rgb_data is not None:
                rgb_np = rgb_data.clone().detach().cpu().numpy()
                
                # Stack images horizontally
                grid_img = np.concatenate(rgb_np[:num_parallel], axis=1)
                
                # Convert to BGR for OpenCV
                if grid_img.shape[-1] == 3:
                    grid_img = cv2.cvtColor(grid_img, cv2.COLOR_RGB2BGR)
                elif grid_img.shape[-1] == 4:
                    grid_img = cv2.cvtColor(grid_img, cv2.COLOR_RGBA2BGR)
                    
                cv2.imshow("Visuomotor Replay - Camera View", grid_img)
                cv2.waitKey(1)"""

code = code.replace(target_step, replacement)

with open(script_path, "w") as f:
    f.write(code)

print("replay_demos.py patched with cv2.")
