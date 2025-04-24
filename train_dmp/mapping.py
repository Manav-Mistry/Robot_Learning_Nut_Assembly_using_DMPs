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
    env = create_env("NutAssemblyThreeNuts")

    # 2. Load your DMP-generated trajectories (from original demos)
    hdf5_files = [
        {"path": r"C:\\Users\\Admin\\robosuite_demos\\splitted_traj_for_revised_demo\\demo_split_pick.hdf5", "type": "pick"},
        {"path": r"C:\\Users\\Admin\\robosuite_demos\\splitted_trajectories\\demo_split_place.hdf5", "type": "place"},
    ]

    fixed_timestep_count = 100
    
    all_dmp_models, _, all_spline_trajectory, all_original_demo_eff_pos, all_trajectory_types = generate_DMP_trajectories(
        hdf5_files, fixed_timestep_count
    )

    dmp_pick, dmp_place = get_pick_and_place_dmp(all_dmp_models, all_trajectory_types)
    original_pick_traj, original_place_traj = get_original_traj_for_pick_and_place(all_original_demo_eff_pos,all_trajectory_types)
    spline_pick_traj, spline_place_traj = get_spline_traj_for_pick_and_place(all_spline_trajectory, all_trajectory_types)

    # 3. Planner Actions
    planner_actions = parse_planner_actions(r"C:\Users\Admin\anaconda3\envs\robosuite\project_scripts_2\pddl\new_task_for_three_nuts.pddl.soln")

    for step in planner_actions:
        act_type = step['action']
        nut = step['nut']
        peg = step['peg']

        
        if act_type == "pick":
            object_name = nut
            goal_pos_pick = get_goal_pos(env, object_name)
            (x, y, z) = original_pick_traj[-1] # ee pos at last time step in pick demonstration
            goal_pos_pick[2] = z
            run_pick_dmp(env, dmp_pick, spline_pick_traj, goal_pos_pick)

        elif act_type == "place":
            peg_name = peg
            goal_pos_place = get_peg_pos(env, peg_name)
            print("peg pos goal_pos_place: ", goal_pos_place)
            # (x, y, z) = goal_pos_place[-1]
            goal_pos_place[2] +=  0.12
            print("updated goal_pos_place:", goal_pos_place)
            run_place_dmp(env, dmp_place, spline_place_traj, goal_pos_place)


if __name__ == "__main__":
    main()
