import h5py
import numpy as np
import matplotlib.pyplot as plt
from movement_primitives.dmp import DMP

# Path to your HDF5 file
hdf5_path = r"C:\Users\Admin\robosuite_demos\1741724444_7722592\demo.hdf5"

# Indices based on your state vector breakdown
EE_POS_IDX = slice(14, 17)  # End-Effector Position (3)
EE_VEL_IDX = slice(21, 24)  # End-Effector Velocity (3)

# Load the HDF5 file
with h5py.File(hdf5_path, "r") as f:
    demo_keys = list(f["data"].keys())
    
    all_trajectories = []
    
    for demo in demo_keys:
        if "demo" in demo:  # Filter out metadata keys
            states = f[f"data/{demo}/states"][:]  # Load full state array
            
            ee_pos = states[:, EE_POS_IDX]  # Extract EE position
            ee_vel = states[:, EE_VEL_IDX]  # Extract EE velocity
            
            ee_traj = np.hstack((ee_pos, ee_vel))  # Combine position and velocity
            all_trajectories.append(ee_traj)


ee_trajectories  = np.array(all_trajectories, dtype=object)

demo_traj = ee_trajectories[0]  # First demonstration

ee_pos = demo_traj[:, :3]
ee_pos = np.array(ee_pos, dtype=np.float64)

timesteps = ee_pos.shape[0]

# print(ee_pos)
# Create time vector
# T = np.linspace(0, 1, timesteps)
T = np.arange(timesteps)

# Define DMP model
dmp = DMP(
    n_dims=3,  # 3D position (x, y, z)
    execution_time=1.0,
    dt=0.01,
    n_weights_per_dim=20,  # More weights for complex motion
    int_dt=0.0001,
    alpha_y=np.array([25.0, 25.0, 25.0]),
    beta_y=np.array([6.25, 6.25, 6.25]),
)

# Learn from the trajectory
dmp.imitate(T, ee_pos)

# Generate new trajectory using DMP
T_gen, generated_pos = dmp.open_loop()

T_gen = np.linspace(0, timesteps-1, len(T_gen))

fig, axs = plt.subplots(3, 1, figsize=(8, 6))
labels = ["X", "Y", "Z"]
for i in range(3):
    axs[i].plot(T, ee_pos[:, i], label="Original", linestyle="dashed")
    axs[i].plot(T_gen, generated_pos[:, i], label="DMP Generated")
    axs[i].set_ylabel(f"Position {labels[i]}")
    axs[i].legend()

plt.xlabel("Timesteps")
plt.suptitle("DMP Reproduction of End-Effector Position")
plt.show()
