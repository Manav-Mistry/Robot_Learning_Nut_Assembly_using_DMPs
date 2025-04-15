import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load your HDF5 file
hdf5_path = r"C:\Users\Admin\robosuite_demos\friend\demo.hdf5"  # ← Update this filename
with h5py.File(hdf5_path, "r") as f:
    states = f["data"]["demo1"]["states"][:]
    eef_traj = states[:, 52:55]  # x, y, z

# Extract x, y, z
x, y, z = eef_traj[:, 0], eef_traj[:, 1], eef_traj[:, 2]

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
t = np.linspace(0, 1, len(eef_traj))
ax.scatter(x, y, z, c=t, s=3)
# Mark start and end
ax.scatter(x[0], y[0], z[0], c='green', s=80, label='Start', marker='o')
ax.scatter(x[-1], y[-1], z[-1], c='red', s=80, label='End', marker='X')

ax.set_title("End-Effector 3D Trajectory")
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.view_init(elev=30, azim=135)  # Optional: adjust camera angle

plt.tight_layout()
plt.show()
