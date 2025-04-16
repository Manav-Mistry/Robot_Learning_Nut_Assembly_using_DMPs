import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline, splrep
from movement_primitives.dmp import DMP
from scipy.spatial.transform import Rotation as R


def return_eef_pos_from_states(hdf5_path):
    with h5py.File(hdf5_path, "r") as f:
        # List available episodes
        episodes = list(f["data"].keys())
        print("Available episodes:", episodes) # output: ['demo_1']

        # Load first (or only) demonstration
        ep = f["data"][episodes[0]] 
        print(ep.keys())

        print(ep["actions"].shape)
        print(ep["states"].shape)
        
        # Get the full trajectory
        actions = ep["actions"][:]                # [T, action_dim]
        states = ep["states"][:]                  # [T, state_dim]

        end_eff_states = [state[-3:] for state in states]
    
    return np.array(end_eff_states)

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


def print_dmp_and_original_traj(dmp_traj, original_traj):
    for (T_gen, generated_traj) in (dmp_traj):
        print("Lenght of DMP traj:", len(T_gen))
        print("Print T_gen: ", T_gen)
        print("--------------------------------------")
        print("Length of end effector array: ", len(generated_traj))
        print("Generated Traj: ", generated_traj)

    for (T, original_traj) in (dmp_traj):
        print("Lenght of original_traj:", len(T))
        print("Print T: ", T)
        print("--------------------------------------")
        print("Length of end effector array original_traj: ", len(original_traj))
        print("original_traj: ", original_traj)
    

def generate_DMP_trajectories(hdf5_files, fixed_timestep_count):
    all_dmp_models = []  
    all_original_demo_eff_pos = []
    all_spline_trajectories = [] 
    all_generated_trajectories = []
    all_trajectory_types = []  
    
    for file_info in hdf5_files:
        ee_pos = return_eef_pos_from_states(file_info["path"])

        original_timesteps = ee_pos.shape[0]
        original_time = np.linspace(0, 1, original_timesteps)  # Normalize time
        new_time = np.linspace(0, 1, fixed_timestep_count)  # Resample target

        # Interpolate each dimension (X, Y, Z)
        ee_pos_resampled = np.zeros((fixed_timestep_count, 3))

        spline_value = 0.01 if file_info["type"] == "pick" else 0.1
    
        for dim in range(3):
            spline = UnivariateSpline(original_time, ee_pos[:, dim], s=spline_value)
            ee_pos_resampled[:, dim] = spline(new_time) 

        # view_original_and_spline_trajectory(original_time, new_time, ee_pos, ee_pos_resampled)

        # DMP training on resampled data
        dmp = DMP(
            n_dims=3,
            execution_time=1.0,
            dt=1.0 / fixed_timestep_count,  # NEW: match resampled rate
            n_weights_per_dim=100,
            int_dt=0.0001,
            alpha_y=np.array([25.0, 25.0, 25.0]),
            beta_y=np.array([6.25, 6.25, 6.25]),
        )

        dmp.imitate(new_time, ee_pos_resampled)
        T_gen, generated_pos = dmp.open_loop()

        all_dmp_models.append(dmp)
        all_generated_trajectories.append((T_gen, generated_pos))
        all_spline_trajectories.append((new_time, ee_pos_resampled))
        all_original_demo_eff_pos.append(ee_pos)
        all_trajectory_types.append(file_info["type"])
        # view_original_and_spline_trajectory(original_time, new_time, ee_pos, ee_pos_resampled) 

    return all_dmp_models, all_generated_trajectories, all_spline_trajectories, all_original_demo_eff_pos, all_trajectory_types