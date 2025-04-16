import numpy as np
from utils import generate_DMP_trajectories
from reproduce_helper import *

np.random.seed(89)

def main():
    # 1. Setup Environment
    env = create_env()

    # 2. Load your DMP-generated trajectory (from original demo)
    hdf5_files = [
        {"path": r"C:\Users\Admin\robosuite_demos\splitted_traj_for_revised_demo\demo_split_pick.hdf5", "type": "pick"},
        {"path": r"C:\Users\Admin\robosuite_demos\splitted_traj_for_revised_demo\demo_split_place.hdf5", "type": "place"}
    ]

    fixed_timestep_count = 200

    # Only load the model
    all_dmp_models, _, all_spline_trajectory, all_original_demo_eff_pos, all_trajectory_types = generate_DMP_trajectories(hdf5_files, fixed_timestep_count)


    for (dmp_model, spline_traj, original_demo_eff_pos, trajectory_type) in zip(all_dmp_models, all_spline_trajectory, all_original_demo_eff_pos, all_trajectory_types):
        
        if trajectory_type == "pick":
            object_name = "SquareNut_handle_site"
            goal_pos = get_goal_pos(env, object_name)

        if trajectory_type == "place":
            # complete this code
            # goal is the respective peg's location
            peg_name = "peg1"
            goal_pos = get_peg_pos(env, peg_name)

        (x, y, z) = original_demo_eff_pos[-1]

        if trajectory_type == "pick":
            goal_pos[2] = z 
            run_pick_dmp(env, dmp_model, spline_traj, goal_pos)

        if trajectory_type == "place":
            goal_pos[2] = z + 0.015
            run_place_dmp(env, dmp_model, spline_traj, goal_pos)


if __name__ == "__main__":
    main()