from utils import *

def main():

    # hdf5_files = [
    #     r"C:\Users\Admin\robosuite_demos\splitted_trajectories\demo_split_pick.hdf5", # pick
    #     r"C:\Users\Admin\robosuite_demos\splitted_trajectories\demo_split_place.hdf5", # place

    # ] 
    hdf5_files = [
        # {"path": r"C:\Users\Admin\robosuite_demos\splitted_traj_for_revised_demo\demo_split_pick.hdf5", "type": "pick"},
        {"path": r"C:\Users\Admin\robosuite_demos\splitted_traj_for_revised_demo\demo_split_place.hdf5", "type": "place"}
    ]
 
    fixed_timestep_count = 100  # resample

    _, all_generated_trajectories, all_spline_trajectories, all_original_trajs, _ = generate_DMP_trajectories(hdf5_files, fixed_timestep_count)

    plot_spline_and_DMP_generated_trajectories(all_spline_trajectories, all_generated_trajectories)
    plot_spline_and_DMP_generated_trajectories_3D(all_spline_trajectories, all_generated_trajectories)

if __name__ == "__main__":
    main()






