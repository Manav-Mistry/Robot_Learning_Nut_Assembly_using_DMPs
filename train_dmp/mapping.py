import numpy as np
from utils import generate_DMP_trajectories
from reproduce_helper import *

np.random.seed(89)

def get_pick_and_place_dmp(all_dmp_models, all_traj_types):
    
    for (dmp_model, traj_type) in zip(all_dmp_models, all_traj_types):
        if traj_type == "pick":
            dmp_pick = dmp_model
        if traj_type == "place":
            dmp_place = dmp_model

    return (dmp_pick, dmp_place)


def get_original_traj_for_pick_and_place(all_original_demo_eff_pos, all_traj_types):

    for (org_ee_traj, traj_type) in zip(all_original_demo_eff_pos, all_traj_types):
        if traj_type == "pick":
            original_pick_traj = org_ee_traj
        if traj_type == "place":
            original_place_traj = org_ee_traj

    return (original_pick_traj, original_place_traj)

def get_spline_traj_for_pick_and_place(all_spline_trajectory, all_traj_types):

    for (spline_traj, traj_type) in zip(all_spline_trajectory, all_traj_types):
        if traj_type == "pick":
            spline_pick_traj = spline_traj
        if traj_type == "place":
            spline_place_traj = spline_traj

    return (spline_pick_traj, spline_place_traj)


def main():
    # 1. Setup Environment
    env = create_env()

    # 2. Load your DMP-generated trajectories (from original demos)
    hdf5_files = [
        {"path": r"C:\\Users\\Admin\\robosuite_demos\\splitted_traj_for_revised_demo\\demo_split_pick.hdf5", "type": "pick"},
        {"path": r"C:\\Users\\Admin\\robosuite_demos\\splitted_traj_for_revised_demo\\demo_split_place.hdf5", "type": "place"},
    ]

    fixed_timestep_count = 200

    all_dmp_models, _, all_spline_trajectory, all_original_demo_eff_pos, all_trajectory_types = generate_DMP_trajectories(
        hdf5_files, fixed_timestep_count
    )

    dmp_pick, dmp_place = get_pick_and_place_dmp(all_dmp_models, all_trajectory_types)
    original_pick_traj, original_place_traj = get_original_traj_for_pick_and_place(all_original_demo_eff_pos,all_trajectory_types)
    spline_pick_traj, spline_place_traj = get_spline_traj_for_pick_and_place(all_spline_trajectory, all_trajectory_types)

    # 3. Planner Actions
    planner_actions = [
        {'action': 'pick', 'robot': 'robot1', 'nut': 'squarenut', 'peg': None},
        {'action': 'place_on_peg_square', 'robot': 'robot1', 'nut': 'squarenut', 'peg': 'peg1'},
        {'action': 'pick', 'robot': 'robot1', 'nut': 'hexnut', 'peg': None},
        {'action': 'place_on_peg_hex', 'robot': 'robot1', 'nut': 'hexnut', 'peg': 'peg2'}
    ]

    for step in planner_actions:
        act_type = step['action'].split('_')[0]  # pick or place
        nut = step['nut']
        peg = step['peg'] if act_type == 'place' else None

        
        if act_type == "pick":
            object_name = "SquareNut_handle_site" if nut == "squarenut" else "RoundNut_handle_site"
            goal_pos = get_goal_pos(env, object_name)
            (x, y, z) = original_pick_traj[-1] # ee pos at last time step in pick demonstration
            goal_pos[2] = z
            run_pick_dmp(env, dmp_pick, spline_pick_traj, goal_pos)

        elif act_type == "place":
            peg_name = peg
            goal_pos = get_peg_pos(env, peg_name)
            (x, y, z) = original_place_traj[-1]
            goal_pos[2] = z + 0.015
            run_place_dmp(env, dmp_place, spline_place_traj, goal_pos)


if __name__ == "__main__":
    main()
