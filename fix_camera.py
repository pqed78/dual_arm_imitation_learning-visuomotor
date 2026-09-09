import os

script_path = "scripts/replay_demos.py"
with open(script_path, "r") as f:
    code = f.read()

target = """        # Update scene to render sensors
        env.scene.update(dt=env.physics_dt)
        
        # Display camera view
        if "front_camera" in env.scene.keys():
            rgb_data = env.scene["front_camera"].data.output["rgb"]
            if rgb_data is not None:
                rgb_np = rgb_data.clone().detach().cpu().numpy()"""

replacement = """        # Update scene and compute observations to render camera
        env.scene.update(dt=env.physics_dt)
        obs_dict = env.observation_manager.compute()
        
        # Display camera view
        if "image" in obs_dict and "rgb" in obs_dict["image"]:
            rgb_data = obs_dict["image"]["rgb"]
            if rgb_data is not None:
                rgb_np = rgb_data.clone().detach().cpu().numpy()
                if rgb_np.dtype != np.uint8:
                    rgb_np = rgb_np.astype(np.uint8)"""

code = code.replace(target, replacement)
with open(script_path, "w") as f:
    f.write(code)
print("Camera rendering fixed.")
