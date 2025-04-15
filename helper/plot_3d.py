import h5py
import numpy as np
import matplotlib.pyplot as plt
# from scipy.interpolate import interp1d  # NEW
from scipy.interpolate import UnivariateSpline
from movement_primitives.dmp import DMP
from mpl_toolkits.mplot3d import Axes3D

def plot_spline_and_DMP_generated_trajectories_3D(all_spline_trajectories, all_generated_trajectories):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    count = 0
    for (T, ee_pos), (T_gen, gen_pos) in zip(all_spline_trajectories, all_generated_trajectories):
        # Plot full trajectories
        ax.plot(ee_pos[:, 0], ee_pos[:, 1], ee_pos[:, 2], linestyle="dashed", alpha=0.7, label=f"Original Demo {count}")
        ax.plot(gen_pos[:, 0], gen_pos[:, 1], gen_pos[:, 2], label=f"DMP Generated {count}")
        
        # Mark start and end points
        ax.scatter(ee_pos[0, 0], ee_pos[0, 1], ee_pos[0, 2], color='green', marker='o', s=40, label=f"Start Demo {count}")
        ax.scatter(ee_pos[-1, 0], ee_pos[-1, 1], ee_pos[-1, 2], color='red', marker='x', s=40, label=f"End Demo {count}")
        ax.scatter(gen_pos[0, 0], gen_pos[0, 1], gen_pos[0, 2], color='green', marker='o', s=40)
        ax.scatter(gen_pos[-1, 0], gen_pos[-1, 1], gen_pos[-1, 2], color='red', marker='x', s=40)
        
        count += 1

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("DMP Learning for Multiple Demonstrations (3D Trajectory)")

    # Avoid duplicate legends for start/end markers
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys())

    plt.tight_layout()
    plt.show()


def view_original_and_spline_trajectory(original_time, new_time, ee_pos, ee_pos_resampled):
    for dim in range(3):
        plt.figure()
        plt.plot(original_time, ee_pos[:, dim], label='Original', alpha=0.5)
        plt.plot(new_time, ee_pos_resampled[:, dim], label='Smoothed+Resampled')
        plt.title(f'Dimension {dim}')
        plt.legend()
        plt.show()


def plot_spline_and_DMP_generated_trajectories(all_spline_trajectories, all_generated_trajectories):

    fig, axs = plt.subplots(3, 1, figsize=(8, 6))
    labels = ["X", "Y", "Z"]

    for i in range(3):
        count = 0
        for (T, ee_pos), (T_gen, gen_pos) in zip(all_spline_trajectories, all_generated_trajectories):
            axs[i].plot(T, ee_pos[:, i], linestyle="dashed", alpha=0.7, label= f"Original Demo {count}")
            axs[i].plot(T_gen, gen_pos[:, i], label= f"DMP Generated {count}")
            count += 1

        axs[i].set_ylabel(f"Position {labels[i]}")
        axs[i].legend()

    plt.xlabel("Timesteps")
    plt.suptitle("DMP Learning for Multiple Demonstrations")
    plt.show()


def generate_DMP_trajectories(hdf5_files, fixed_timestep_count, EE_POS_IDX):
    all_dmp_models = []  
    all_generated_trajectories = []  
    all_spline_trajectories = []  

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
                    all_spline_trajectories.append((T, ee_pos))
    
    return all_dmp_models, all_generated_trajectories, all_spline_trajectories 


def main():
    hdf5_files = [
        # r"C:\Users\Admin\robosuite_demos\1741724444_7722592\demo.hdf5",
        r"C:\Users\Admin\robosuite_demos\1741904602_8723783\demo.hdf5"
    ]

    EE_POS_IDX = slice(14, 17) 
    fixed_timestep_count = 100  # resample

    all_dmp_models, all_generated_trajectories, all_spline_trajectories = generate_DMP_trajectories(hdf5_files, fixed_timestep_count, EE_POS_IDX)

    # plot_spline_and_DMP_generated_trajectories(all_spline_trajectories, all_generated_trajectories)
    plot_spline_and_DMP_generated_trajectories_3D(all_spline_trajectories, all_generated_trajectories)


if __name__ == "__main__":
    main()






