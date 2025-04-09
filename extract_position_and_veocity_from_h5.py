import h5py
import numpy as np

# Path to your HDF5 demonstration file
hdf5_path = r"C:\Users\Admin\robosuite_demos\1741724444_7722592\demo.hdf5"  # Change this to your actual file path

def explore_hdf5_structure(hdf5_path):
    with h5py.File(hdf5_path, "r") as f:
        print("\nHDF5 File Structure:")
        
        # Recursively print groups and datasets
        def print_structure(name, obj):
            if isinstance(obj, h5py.Group):
                print(f"📂 Group: {name}")
            elif isinstance(obj, h5py.Dataset):
                print(f"📄 Dataset: {name} | Shape: {obj.shape}, Dtype: {obj.dtype}")

        f.visititems(print_structure)

        # Check the first demo's states dataset
        demo_key = list(f["data"].keys())[0]  # First demonstration
        print(f"\nInspecting: {demo_key}")

        states = f[f"data/{demo_key}/states"]
        actions = f[f"data/{demo_key}/actions"]

        print(f"\nStates Shape: {states.shape} | Actions Shape: {actions.shape}")
        print("First State Sample:", states[0])
        print("First Action Sample:", actions[0])

if __name__ == "__main__":
    explore_hdf5_structure(hdf5_path)
