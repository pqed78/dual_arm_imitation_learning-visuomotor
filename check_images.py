import h5py
import numpy as np

try:
    with h5py.File("data/demos.hdf5", "r") as f:
        demo = f["data/demo_0"]
        images = demo["obs"]["image"]["rgb"][:]
        print(f"Images shape: {images.shape}")
        print(f"Max value: {np.max(images)}, Min value: {np.min(images)}")
except Exception as e:
    print(f"Error: {e}")
