import os

script_path = "scripts/replay_demos.py"
with open(script_path, "r") as f:
    lines = f.readlines()

# The duplicated lines are around 176-178
# Let's just fix it by matching the exact duplicated string and removing it.
# Actually it's safer to just replace the whole block cleanly.
code = "".join(lines)
broken_block = """                # Convert to BGR for OpenCV
                if grid_img.shape[-1] == 3:
                    grid_img = cv2.cvtColor(grid_img, cv2.COLOR_RGB2BGR)
                elif grid_img.shape[-1] == 4:
                    grid_img = cv2.cvtColor(grid_img, cv2.COLOR_RGBA2BGR)
                    
                cv2.putText(grid_img, cam_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    grid_img = cv2.cvtColor(grid_img, cv2.COLOR_RGB2BGR)
                elif grid_img.shape[-1] == 4:
                    grid_img = cv2.cvtColor(grid_img, cv2.COLOR_RGBA2BGR)
                    
                cv2.imshow("Visuomotor Replay - Camera View", grid_img)
                cv2.waitKey(1)"""

fixed_block = """                # Convert to BGR for OpenCV
                if grid_img.shape[-1] == 3:
                    grid_img = cv2.cvtColor(grid_img, cv2.COLOR_RGB2BGR)
                elif grid_img.shape[-1] == 4:
                    grid_img = cv2.cvtColor(grid_img, cv2.COLOR_RGBA2BGR)
                    
                cv2.putText(grid_img, cam_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow("Visuomotor Replay - Camera View", grid_img)
                cv2.waitKey(1)"""

new_code = code.replace(broken_block, fixed_block)

with open(script_path, "w") as f:
    f.write(new_code)
