import os

script_path = "scripts/replay_demos.py"
with open(script_path, "r") as f:
    code = f.read()

target1 = """    cv2.namedWindow("Visuomotor Replay - Camera View", cv2.WINDOW_NORMAL)
    
    print(f"\\n--- Starting KINEMATIC parallel replay (Max steps: {max_length}) ---")"""
replacement1 = """    cv2.namedWindow("Visuomotor Replay - Camera View", cv2.WINDOW_NORMAL)
    
    # Setup Video Writer
    video_out = None
    
    print(f"\\n--- Starting KINEMATIC parallel replay (Max steps: {max_length}) ---")"""

target2 = """                cv2.putText(grid_img, cam_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow("Visuomotor Replay - Camera View", grid_img)
                cv2.waitKey(1)"""
replacement2 = """                cv2.putText(grid_img, cam_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow("Visuomotor Replay - Camera View", grid_img)
                cv2.waitKey(1)
                
                # Write to video
                if video_out is None:
                    h, w = grid_img.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_out = cv2.VideoWriter('replay_video.mp4', fourcc, 30.0, (w, h))
                video_out.write(grid_img)"""

target3 = """    print("Finished kinematic replay.")
    time.sleep(2.0)
    cv2.destroyAllWindows()"""
replacement3 = """    print("Finished kinematic replay.")
    time.sleep(2.0)
    if video_out is not None:
        video_out.release()
        print("Saved replay video to replay_video.mp4")
    cv2.destroyAllWindows()"""

code = code.replace(target1, replacement1)
code = code.replace(target2, replacement2)
code = code.replace(target3, replacement3)

with open(script_path, "w") as f:
    f.write(code)
print("Video writer added.")
