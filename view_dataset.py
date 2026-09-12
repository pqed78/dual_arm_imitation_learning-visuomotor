import h5py
import cv2
import os

hdf5_path = "data/demos.hdf5"
out_dir = "sample_images"

if not os.path.exists(hdf5_path):
    print(f"Dataset not found at {hdf5_path}!")
    exit(1)

os.makedirs(out_dir, exist_ok=True)

with h5py.File(hdf5_path, "r") as f:
    grp = f["data"]
    demo_keys = list(grp.keys())
    if not demo_keys:
        print("No demos found in dataset.")
        exit(1)
        
    print(f"Found {len(demo_keys)} demos.")
    
    # Extract first 3 frames from the first 4 demos
    for demo_idx in range(min(4, len(demo_keys))):
        demo_key = demo_keys[demo_idx]
        print(f"Extracting sample images from {demo_key}...")
        
        if "images" in grp[demo_key]:
            images = grp[demo_key]["images"][:]
            for i in range(min(3, len(images))):
                img = images[i]
                if img.max() <= 1.0:
                    img = (img * 255.0).astype('uint8')
                if img.shape[-1] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                out_path = os.path.join(out_dir, f"{demo_key}_frame_{i}.png")
                cv2.imwrite(out_path, img)
                print(f"  Saved: {out_path}")
        else:
            print(f"  Could not find RGB image data inside {demo_key}.")

print(f"\nExtraction complete! Check the '{out_dir}' folder to view the images.")
