import os

script_path = "scripts/replay_demos.py"
with open(script_path, "r") as f:
    code = f.read()

target1 = """    cv2.namedWindow("Visuomotor Replay - Camera View", cv2.WINDOW_NORMAL)
    
    # Setup Video Writer"""
replacement1 = """    # Setup Video Writer"""

target2 = """                cv2.putText(grid_img, cam_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow("Visuomotor Replay - Camera View", grid_img)
                cv2.waitKey(1)
                
                # Write to video"""
replacement2 = """                cv2.putText(grid_img, cam_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Write to video"""

target3 = """    if video_out is not None:
        video_out.release()
        print("Saved replay video to replay_video.mp4")
    cv2.destroyAllWindows()"""
replacement3 = """    if video_out is not None:
        video_out.release()
        print("Saved replay video to replay_video.mp4")"""

code = code.replace(target1, replacement1)
code = code.replace(target2, replacement2)
code = code.replace(target3, replacement3)

with open(script_path, "w") as f:
    f.write(code)
print("Removed cv2 GUI calls.")
