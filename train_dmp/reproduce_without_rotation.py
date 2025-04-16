import numpy as np
import time
import robosuite as suite
from robosuite import load_composite_controller_config
from robosuite.wrappers import VisualizationWrapper
from scipy.spatial.transform import Rotation as R
from utils import generate_DMP_trajectories, plot_spline_and_DMP_generated_trajectories_3D

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

    return env


def get_eff_pos(env):
    eef_start = env.sim.data.site_xpos[env.robots[0].eef_site_id["right"]].copy()
    return eef_start


def get_goal_pos(env, object_name):
    goal_pos = env.sim.data.site_xpos[env.sim.model.site_name2id(object_name)].copy()
    return goal_pos


def main():
    # 1. Setup Environment
    env = create_env()
    env = VisualizationWrapper(env)
    env.reset()

    # 2. Load your DMP-generated trajectory (from original demo)
    hdf5_files = [
        r"C:\Users\Admin\robosuite_demos\friend\demo.hdf5",  # for pick action
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

    dmp_model.goal_y  = goal_pos
    # 5. Reconfigure DMP and generate new trajectory
    # dmp_model.configure(start_y=eef_start, goal_y=goal_pos)
    T_gen, dmp_trajectory = dmp_model.open_loop()

    all_generated_traj = []
    all_generated_traj.append((T_gen, dmp_trajectory))
    # plot spline and dmp generated trajectory
    plot_spline_and_DMP_generated_trajectories_3D(all_spline_trajectory, all_generated_traj)


    # 6. Playback in simulation
    grasp = np.array([-1], dtype=np.float32)  # Gripper open
    for pos in dmp_trajectory:
        current = env.sim.data.site_xpos[env.robots[0].eef_site_id["right"]].copy()
        dpos = pos - current
        drot = np.zeros(3)  # no rotation for now
        action = np.concatenate([dpos, drot, grasp])

        env.step(action)
        env.render()
        time.sleep(0.05)

    for _ in range(100):
        dpos = goal_pos - current
        if np.linalg.norm(dpos) < 0.02:
            break
        env.step(np.concatenate([dpos, np.zeros(3), grasp]))
        env.render()
        time.sleep(0.05)

    env.close()


if __name__ == "__main__":
    main()