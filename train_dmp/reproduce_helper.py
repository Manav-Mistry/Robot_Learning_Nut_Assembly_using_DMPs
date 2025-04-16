import numpy as np
from scipy.spatial.transform import Rotation as R
import time
import robosuite as suite
from robosuite import load_composite_controller_config
from robosuite.wrappers import VisualizationWrapper
from utils import plot_spline_and_DMP_generated_trajectories_3D

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


def create_env():
    # 1. Setup Environment
    controller_config = load_composite_controller_config(robot="Kinova3")

    env = suite.make(
        env_name="NutAssembly",
        robots="Kinova3",
        controller_configs=controller_config,
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera="agentview",
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
    )

    env = VisualizationWrapper(env)
    env.reset()

    rotate_nuts_in_env(env, angle_degrees=180, axis="z")
    # move_nuts_between_eef_and_pegs(env)
    move_nuts_with_random_y_safe(env)

    return env


def get_eff_pos(env):
    eef_start = env.sim.data.site_xpos[env.robots[0].eef_site_id["right"]].copy()
    return eef_start


def get_goal_pos(env, object_name):
    goal_pos = env.sim.data.site_xpos[env.sim.model.site_name2id(object_name)].copy()
    return goal_pos


def run_pick_dmp(env, dmp_model, spline_traj, goal_pos):
    # Set goal in DMP model
    dmp_model.goal_y = goal_pos

    # Generate new trajectory using the trained DMP model
    T_gen, dmp_trajectory = dmp_model.open_loop()

    all_generated_traj = [(T_gen, dmp_trajectory)]
    plot_spline_and_DMP_generated_trajectories_3D([spline_traj], all_generated_traj)

    # Playback in simulation
    grasp = np.array([-1], dtype=np.float32)  # Gripper open
    for pos in dmp_trajectory:
        current = env.sim.data.site_xpos[env.robots[0].eef_site_id["right"]].copy()
        dpos = pos - current
        drot = np.zeros(3)
        action = np.concatenate([dpos, drot, grasp])

        env.step(action)
        env.render()
        time.sleep(0.05)

    # Final adjustment if needed
    for _ in range(100):
        current = env.sim.data.site_xpos[env.robots[0].eef_site_id["right"]].copy()
        dpos = goal_pos - current
        if np.linalg.norm(dpos) < 0.02:
            break
        env.step(np.concatenate([dpos, np.zeros(3), grasp]))
        env.render()

    env.close()