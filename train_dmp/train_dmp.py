from utils import *

def main():

    hdf5_files = [
        r"C:\Users\Admin\robosuite_demos\friend\demo.hdf5", # pick
    ]  
    fixed_timestep_count = 100  # resample

    _, all_generated_trajectories, all_spline_trajectories = generate_DMP_trajectories(hdf5_files, fixed_timestep_count)

    # plot_spline_and_DMP_generated_trajectories(all_spline_trajectories, all_generated_trajectories)
    plot_spline_and_DMP_generated_trajectories_3D(all_spline_trajectories, all_generated_trajectories)
    # print_dmp_and_original_traj(all_generated_trajectories, all_spline_trajectories)

if __name__ == "__main__":
    main()






