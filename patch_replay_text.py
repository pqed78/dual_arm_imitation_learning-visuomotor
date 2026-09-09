import os

script_path = "scripts/replay_demos.py"
with open(script_path, "r") as f:
    code = f.read()

target = """                # Convert to BGR for OpenCV
                if grid_img.shape[-1] == 3:"""
replacement = """                # Add camera position text to the first image
                cam_pos = env.scene["front_camera"].data.pos_w[0].cpu().numpy()
                cam_text = f"Cam Pos: [{cam_pos[0]:.2f}, {cam_pos[1]:.2f}, {cam_pos[2]:.2f}]"
                
                # Convert to BGR for OpenCV
                if grid_img.shape[-1] == 3:
                    grid_img = cv2.cvtColor(grid_img, cv2.COLOR_RGB2BGR)
                elif grid_img.shape[-1] == 4:
                    grid_img = cv2.cvtColor(grid_img, cv2.COLOR_RGBA2BGR)
                    
                cv2.putText(grid_img, cam_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
"""

code = code.replace(target, replacement)

with open(script_path, "w") as f:
    f.write(code)

print("replay_demos.py patched with text.")
