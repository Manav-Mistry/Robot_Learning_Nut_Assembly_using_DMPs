import numpy as np
from scipy.spatial.transform import Rotation as R
import time
import robosuite as suite
from robosuite import load_composite_controller_config
from robosuite.wrappers import VisualizationWrapper
from utils import plot_spline_and_DMP_generated_trajectories_3D

def parse_planner_actions(filepath):
    # Predefined type maps (you can make this dynamic later)
    nut_types = {
        'hexnut': 'RoundNut_handle_site',
        'squarenut': 'SquareNut_handle_site',
        'hexnut1': 'RoundNut1_handle_site',
        'hexnut2': 'RoundNut2_handle_site'

    }
    peg_types = {
        'square': 'peg1',
        'round': 'peg2'
    }

    planner_actions = []

    with open(filepath, 'r') as file:
        for line in file:
            tokens = line.strip("()\n").split()
            if not tokens:
                continue

            action = tokens[0]

            if action == "pick":
                _, robot, nut = tokens
                nut_type = nut_types.get(nut)
                planner_actions.append({
                    'action': 'pick',
                    'nut': nut_type,
                    'peg': None
                })

            elif action == "place":
                _, robot, nut, peg = tokens
                nut_type = nut_types.get(nut)
                peg_type = peg_types.get(peg)

                
                planner_actions.append({
                    'action': "place",
                    'nut': nut_type,
                    'peg': peg_type
                })

    return planner_actions



def rotate_three_nuts_in_env(env, angle_degrees=180, axis="z"):
    r_rot = R.from_euler(axis, angle_degrees, degrees=True)
    quat = np.roll(r_rot.as_quat(), 1)  # Mujoco uses wxyz

    nut_joints = ["SquareNut_joint0", "RoundNut1_joint0", "RoundNut2_joint0"]

    for joint in nut_joints:
        pos = env.sim.data.get_joint_qpos(joint)[:3]
        env.sim.data.set_joint_qpos(joint, np.concatenate([pos, quat]))

    env.sim.forward()


def move_three_nuts_with_random_y_safe(env, peg1_body="peg1", peg2_body="peg2", arm="right", margin=0.02):
    eef_xy = env.sim.data.site_xpos[env.robots[0].eef_site_id[arm]].copy()[:2]
    peg1_xy = env.sim.data.get_body_xpos(peg1_body)[:2]
    peg2_xy = env.sim.data.get_body_xpos(peg2_body)[:2]

    def safe_rand(a, b, m):
        low, high = sorted([a, b])
        return np.random.uniform(low + m, high - m)

    # Calculate 3 positions between eef and peg1/peg2
    square_xy = [(eef_xy[0] + peg1_xy[0]) / 2, safe_rand(eef_xy[1], peg1_xy[1], margin)]
    round1_xy = [(peg1_xy[0] + peg2_xy[0]) / 2, safe_rand(peg1_xy[1], peg2_xy[1], margin)]
    round2_xy = [(eef_xy[0] + peg2_xy[0]) / 2, safe_rand(eef_xy[1], peg2_xy[1], margin)]

    def set_xy(joint, new_xy):
        qpos = env.sim.data.get_joint_qpos(joint)
        env.sim.data.set_joint_qpos(joint, np.concatenate([[new_xy[0], new_xy[1], qpos[2]], qpos[3:]]))

    set_xy("SquareNut_joint0", square_xy)
    set_xy("RoundNut1_joint0", round1_xy)
    set_xy("RoundNut2_joint0", round2_xy)

    env.sim.forward()



def rotate_nuts_in_env(env, angle_degrees=180, axis="z"):
    r_rot = R.from_euler(axis, angle_degrees, degrees=True)
    quat = np.roll(r_rot.as_quat(), 1)

    for nut_name in ["SquareNut_joint0", "RoundNut_joint0"]:
        pos = env.sim.data.get_joint_qpos(nut_name)[:3]
        env.sim.data.set_joint_qpos(nut_name, np.concatenate([pos, quat]))
    
    env.sim.forward()


def move_nuts_with_random_y_safe(env, peg1_body="peg1", peg2_body="peg2", arm="right", margin=0.02):
    eef_xy = env.sim.data.site_xpos[env.robots[0].eef_site_id[arm]].copy()[:2]
    peg1_xy = env.sim.data.get_body_xpos(peg1_body)[:2]
    peg2_xy = env.sim.data.get_body_xpos(peg2_body)[:2]

    square_x = (eef_xy[0] + peg1_xy[0]) / 2
    round_x = (eef_xy[0] + peg2_xy[0]) / 2

    def safe_rand(a, b, m):
        low, high = sorted([a, b])
        return np.random.uniform(low + m, high - m)

    square_y = safe_rand(eef_xy[1], peg1_xy[1], margin)
    round_y = safe_rand(eef_xy[1], peg2_xy[1], margin)

    sq = env.sim.data.get_joint_qpos("SquareNut_joint0")
    rd = env.sim.data.get_joint_qpos("RoundNut_joint0")

    env.sim.data.set_joint_qpos("SquareNut_joint0", np.concatenate([[square_x, square_y, sq[2]], sq[3:]]))
    env.sim.data.set_joint_qpos("RoundNut_joint0", np.concatenate([[round_x, round_y, rd[2]], rd[3:]]))
    env.sim.forward()


def create_env(env_name):
    # 1. Setup Environment
    controller_config = load_composite_controller_config(robot="Kinova3")

    env = suite.make(
        env_name=env_name,
        robots="Kinova3",
        controller_configs=controller_config,
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera="agentview",
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=1,
    )

    env = VisualizationWrapper(env)
    env.reset()

    rotate_three_nuts_in_env(env, angle_degrees=180, axis="z")
    # move_nuts_between_eef_and_pegs(env)
    # move_nuts_with_random_y_safe(env)
    # make_nuts_massless(env)

    return env


def get_eff_pos(env):
    eef_start = env.sim.data.site_xpos[env.robots[0].eef_site_id["right"]].copy()
    return eef_start


def get_goal_pos(env, object_name):
    goal_pos = env.sim.data.site_xpos[env.sim.model.site_name2id(object_name)].copy()
    return goal_pos

def get_peg_pos(env, object_name):
    goal_pos = env.sim.data.get_body_xpos(object_name).copy()
    return goal_pos


def run_pick_dmp(env, dmp_model, spline_traj, goal_pos):
    # Set goal in DMP model
    dmp_model.goal_y = goal_pos

    # Generate new trajectory using the trained DMP model
    T_gen, dmp_trajectory = dmp_model.open_loop()
    # print("Last point of DMP trajectory: ", dmp_trajectory[-1])
    # print("Actual Goal pos: ", goal_pos)
    all_generated_traj = [(T_gen, dmp_trajectory)]
    plot_spline_and_DMP_generated_trajectories_3D([spline_traj], all_generated_traj)

    # Playback in simulation
    grasp = np.array([-1], dtype=np.float32)  # Gripper open
    for idx, pos in enumerate(dmp_trajectory):
        current = get_eff_pos(env)
        dpos = pos - current
        drot = np.zeros(3)
        if idx == len(dmp_trajectory)-1:
            grasp = np.array([1], dtype=np.float32)
        
        action = np.concatenate([dpos, drot, grasp])

        env.step(action)
    #     env.render()
    
            

def run_place_dmp(env, dmp_model, spline_traj, goal_pos):
    # Set the goal in DMP model
    dmp_model.goal_y = goal_pos

    # Generate new trajectory
    T_gen, dmp_trajectory = dmp_model.open_loop()

    all_generated_traj = [(T_gen, dmp_trajectory)]
    print("Last point of DMP trajectory: ", dmp_trajectory[-1])
    print("Actual Goal pos: ", goal_pos)

    plot_spline_and_DMP_generated_trajectories_3D([spline_traj], all_generated_traj)

    # Keeping gripper closed throughout place motion
    grasp = np.array([1.0], dtype=np.float32)

    for idx, pos in enumerate(dmp_trajectory):
        current = get_eff_pos(env)
        print("Track ee: ", current)
        dpos = pos - current
        drot = np.zeros(3)
        if idx == len(dmp_trajectory) - 1:
            grasp = np.array([-1.0], dtype=np.float32)

        action = np.concatenate([dpos, drot, grasp])
        env.step(action)
        env.render()
        # time.sleep(0.005)



