import h5py
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# Load HDF5 file
# hdf5_path = r"C:\Users\Admin\robosuite_demos\1741724444_7722592\demo.hdf5"
hdf5_path = r"C:\Users\Admin\robosuite_demos\1741904602_8723783\demo.hdf5"

EE_POS_IDX = slice(14, 17) 

with h5py.File(hdf5_path, "r") as f:
    states = f["data"]["demo_1"]["states"][:]  # shape: (4896, N)

# STEP 1: Extract end-effector position (assumed to be first 3 columns: [X, Y, Z])
eef_pos = states[:, EE_POS_IDX]
x_pos = eef_pos[:, 0]
y_pos = eef_pos[:, 1]
z_pos = eef_pos[:, 2]

# STEP 2: Invert Z to detect valleys
z_inverted = -z_pos

# STEP 3: Find valleys in Z
peaks, _ = find_peaks(z_inverted, distance=500, prominence=0.005)
print("Potential placement points (valleys in Z):", peaks)

# STEP 4: Compute split index
if len(peaks) >= 2:
    split_idx = int((peaks[0] + peaks[1]) // 2)
    print(f"Estimated split point between nut 1 and nut 2 at timestep: {split_idx}")
else:
    print("Warning: Could not find two clear placement valleys.")
    split_idx = None

# STEP 5: Plot X, Y, Z with split marker
fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

axs[0].plot(x_pos, label='X Position')
# axs[0].axvline(split_idx, color='r', linestyle='--', label='Split Point')
axs[0].set_ylabel('X')
axs[0].legend()
axs[0].grid(True)

axs[1].plot(y_pos, label='Y Position')
# axs[1].axvline(split_idx, color='r', linestyle='--')
axs[1].set_ylabel('Y')
axs[1].legend()
axs[1].grid(True)

axs[2].plot(z_pos, label='Z Position')
# axs[2].axvline(split_idx, color='r', linestyle='--')
axs[2].set_xlabel('Timestep')
axs[2].set_ylabel('Z')
axs[2].legend()
axs[2].grid(True)

fig.suptitle('End-Effector X, Y, Z Trajectories with Split Point')
plt.tight_layout()
plt.show()
