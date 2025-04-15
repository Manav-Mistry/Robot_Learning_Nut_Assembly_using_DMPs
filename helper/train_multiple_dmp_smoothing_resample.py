import h5py
import numpy as np
import matplotlib.pyplot as plt
# from scipy.interpolate import interp1d  # NEW
from scipy.interpolate import UnivariateSpline, splrep
from movement_primitives.dmp import DMP
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R, Slerp

def plot_spline_and_DMP_generated_trajectories_3D(all_spline_trajectories, all_generated_trajectories):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    count = 0
    for (T, ee_pos), (T_gen, gen_pos) in zip(all_spline_trajectories, all_generated_trajectories):
        ax.plot(ee_pos[:, 0], ee_pos[:, 1], ee_pos[:, 2], linestyle="dashed", alpha=0.7, label=f"Original Demo {count}")
        ax.plot(gen_pos[:, 0], gen_pos[:, 1], gen_pos[:, 2], label=f"DMP Generated {count}")
        count += 1

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("DMP Learning for Multiple Demonstrations (3D Trajectory)")
    ax.legend()
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


def generate_DMP_trajectories(hdf5_files, fixed_timestep_count, EE_POS_IDX, EE_ORI_IDX):
    all_dmp_models = []  
    all_generated_trajectories = []  
    all_spline_trajectories = [] 
    all_ee_ori = []
    all_original_time = []
    
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
                        spline = UnivariateSpline(original_time, ee_pos[:, dim], s=1e-1)
                        ee_pos_resampled[:, dim] = spline(new_time)  # NEW
                        
                    # Same way resampling Quat
                    ee_ori = states[:, EE_ORI_IDX]
                    ee_ori = np.array(ee_ori, dtype=np.float64)
                    
                    # NOTE
                    # rotations = R.from_quat(ee_ori)  # [x, y, z, w]
                    # slerp = Slerp(original_time, rotations)
                    # resampled_rotations = slerp(new_time)
                    # ee_ori_resampled = resampled_rotations.as_quat() 


                    # view_original_and_spline_trajectory(original_time, new_time, ee_pos, ee_pos_resampled)

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

                    # NOTE
                    # resampled_rotations = slerp(new_time)
                    # ee_ori_resampled = resampled_rotations.as_quat() 

                    all_dmp_models.append(dmp)
                    all_generated_trajectories.append((T_gen, generated_pos))
                    all_spline_trajectories.append((new_time, ee_pos_resampled))  # NEW 
                    all_ee_ori.append(ee_ori)
                    all_original_time.append(original_time)

    return all_dmp_models, all_generated_trajectories, all_spline_trajectories, all_ee_ori, all_original_time 


def main():
    hdf5_files = [
        r"C:\Users\Admin\robosuite_demos\1741724444_7722592\demo.hdf5",
        r"C:\Users\Admin\robosuite_demos\1741904602_8723783\demo.hdf5"
    ]

    EE_POS_IDX = slice(14, 17) 
    fixed_timestep_count = 100  # resample

    EE_ORI_IDX = slice(17, 21)

    _, all_generated_trajectories, all_spline_trajectories, _, _ = generate_DMP_trajectories(hdf5_files, fixed_timestep_count, EE_POS_IDX, EE_ORI_IDX)

    plot_spline_and_DMP_generated_trajectories(all_spline_trajectories, all_generated_trajectories)
    # plot_spline_and_DMP_generated_trajectories_3D(all_spline_trajectories, all_generated_trajectories)


if __name__ == "__main__":
    main()






