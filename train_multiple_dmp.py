import h5py
import numpy as np
import matplotlib.pyplot as plt
from movement_primitives.dmp import DMP

hdf5_files = [
    r"C:\Users\Admin\robosuite_demos\1741724444_7722592\demo.hdf5",
    r"C:\Users\Admin\robosuite_demos\1741904602_8723783\demo.hdf5"
]  # Add more files if needed

# Stored Data
# [ Joint Angles (7) | Joint Velocities (7) | EE Position (3) | EE Orientation (4) | EE Linear Velocity (3) | EE Angular Velocity (3) | Other (26) ]


# Indices for end-effector position
EE_POS_IDX = slice(14, 17) 

all_dmp_models = []  
all_generated_trajectories = []  
all_original_trajectories = []  

for file_path in hdf5_files:
    with h5py.File(file_path, "r") as f:
        demo_keys = list(f["data"].keys())
        
        for demo in demo_keys:
            if "demo" in demo:
                states = f[f"data/{demo}/states"][:]  
                
                # Extract End-Effector Position
                ee_pos = states[:, EE_POS_IDX]
                ee_pos = np.array(ee_pos, dtype=np.float64)
                
                # Timesteps
                timesteps = ee_pos.shape[0]
                T = np.arange(timesteps)  

                # Create and train a separate DMP
                dmp = DMP(
                    n_dims=3,  # 3D position (x, y, z)
                    execution_time=1.0,
                    dt=0.01,
                    n_weights_per_dim=20,  # Adjust weights if needed
                    int_dt=0.0001,
                    alpha_y=np.array([25.0, 25.0, 25.0]),
                    beta_y=np.array([6.25, 6.25, 6.25]),
                )
                
                dmp.imitate(T, ee_pos)  # Train DMP
                T_gen, generated_pos = dmp.open_loop()  # Generate new trajectory
                
                # Adjusting generated time to match original time range
                T_gen = np.linspace(0, timesteps - 1, len(T_gen))

                all_dmp_models.append(dmp)
                all_generated_trajectories.append((T_gen, generated_pos))
                all_original_trajectories.append((T, ee_pos))


fig, axs = plt.subplots(3, 1, figsize=(8, 6))
labels = ["X", "Y", "Z"]

for i in range(3):
    count = 0
    for (T, ee_pos), (T_gen, gen_pos) in zip(all_original_trajectories, all_generated_trajectories):
        axs[i].plot(T, ee_pos[:, i], linestyle="dashed", alpha=0.7, label= f"Original Demonstration{count}")
        axs[i].plot(T_gen, gen_pos[:, i], label= f"DMP Generated for Demo {count}")
        count += 1
    
    axs[i].set_ylabel(f"Position {labels[i]}")
    axs[i].legend()

plt.xlabel("Timesteps")
plt.suptitle("DMP Learning for Multiple Demonstrations")
plt.show()
