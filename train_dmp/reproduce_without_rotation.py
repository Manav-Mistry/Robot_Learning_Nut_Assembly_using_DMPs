import numpy as np
import time
import robosuite as suite
from robosuite import load_composite_controller_config
from robosuite.wrappers import VisualizationWrapper
from scipy.spatial.transform import Rotation as R
from utils import generate_DMP_trajectories
from reproduce_helper import *

np.random.seed(222)

def main():
    # 1. Setup Environment
    env = create_env()

    # 2. Load your DMP-generated trajectory (from original demo)
    hdf5_files = [
        {"path": r"C:\Users\Admin\robosuite_demos\splitted_traj_for_revised_demo\demo_split_pick.hdf5", "type": "pick"},
        {"path": r"C:\Users\Admin\robosuite_demos\splitted_traj_for_revised_demo\demo_split_place.hdf5", "type": "place"}
    ]

    fixed_timestep_count = 100

    # Only load the model
    all_dmp_models, _, all_spline_trajectory, all_original_demo_eff_pos = generate_DMP_trajectories(hdf5_files, fixed_timestep_count)

    spline_traj = all_spline_trajectory[0]
    dmp_model = all_dmp_models[0]  # assuming one trajectory

    # 3. Get current EEF position (start)

    eef_start = get_eff_pos(env)
    print(f"Start position (EEF): {eef_start}")

    # 4. Get SquareNut handle site position (goal)
    goal_pos = get_goal_pos(env, "RoundNut_handle_site")

    # 4.5 hardcode the height so that it goes to ground level
    (x, y, z) = all_original_demo_eff_pos[0][-1]
    goal_pos[2] = z

    run_pick_dmp(env, dmp_model, spline_traj, goal_pos)


if __name__ == "__main__":
    main()