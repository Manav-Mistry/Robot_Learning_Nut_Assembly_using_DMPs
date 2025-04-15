import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d  # NEW
from movement_primitives.dmp import DMP

hdf5_files = [
    r"C:\Users\Admin\robosuite_demos\1741724444_7722592\demo.hdf5",
    r"C:\Users\Admin\robosuite_demos\1741904602_8723783\demo.hdf5"
]

EE_POS_IDX = slice(14, 17) 

all_dmp_models = []  
all_generated_trajectories = []  
all_original_trajectories = []  

fixed_timestep_count = 100  # NEW: You want to resample to this

for file_path in hdf5_files:
    with h5py.File(file_path, "r") as f:
        demo_keys = list(f["data"].keys())
        
        for demo in demo_keys:
            if "demo" in demo:
                states = f[f"data/{demo}/states"][:]  
                
                ee_pos = states[:, EE_POS_IDX]
                ee_pos = np.array(ee_pos, dtype=np.float64)

                original_timesteps = ee_pos.shape[0]
                original_time = np.linspace(0, 1, original_timesteps)  # Normalize time
                new_time = np.linspace(0, 1, fixed_timestep_count)  # NEW: Resample target

                # Interpolate each dimension (X, Y, Z)
                ee_pos_resampled = np.zeros((fixed_timestep_count, 3))  # NEW
                for dim in range(3):
                    interp_func = interp1d(original_time, ee_pos[:, dim], kind='linear')
                    ee_pos_resampled[:, dim] = interp_func(new_time)  # NEW

                # DMP training on resampled data
                dmp = DMP(
                    n_dims=3,
                    execution_time=1.0,
                    dt=1.0 / fixed_timestep_count,  # NEW: match resampled rate
                    n_weights_per_dim=20,
                    int_dt=0.0001,
                    alpha_y=np.array([25.0, 25.0, 25.0]),
                    beta_y=np.array([6.25, 6.25, 6.25]),
                )

                dmp.imitate(new_time, ee_pos_resampled)
                T_gen, generated_pos = dmp.open_loop()

                all_dmp_models.append(dmp)
                all_generated_trajectories.append((T_gen, generated_pos))
                all_original_trajectories.append((new_time, ee_pos_resampled))  # NEW

fig, axs = plt.subplots(3, 1, figsize=(8, 6))
labels = ["X", "Y", "Z"]

for i in range(3):
    count = 0
    for (T, ee_pos), (T_gen, gen_pos) in zip(all_original_trajectories, all_generated_trajectories):
        axs[i].plot(T, ee_pos[:, i], linestyle="dashed", alpha=0.7, label= f"Original Demo {count}")
        axs[i].plot(T_gen, gen_pos[:, i], label= f"DMP Generated {count}")
        count += 1

    axs[i].set_ylabel(f"Position {labels[i]}")
    axs[i].legend()

plt.xlabel("Timesteps")
plt.suptitle("DMP Learning for Multiple Demonstrations")
plt.show()
