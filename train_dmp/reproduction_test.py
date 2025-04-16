import numpy as np
import time
import robosuite as suite
from robosuite import load_composite_controller_config
from robosuite.wrappers import VisualizationWrapper
from scipy.spatial.transform import Rotation as R, Slerp
from utils import generate_DMP_trajectories, plot_spline_and_DMP_generated_trajectories_3D, plot_spline_and_DMP_generated_trajectories


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


def get_goal_ore(env, object_name):
    ore = env.sim.data.get_body_xquat(object_name).copy()
    return ore


def get_initial_eff_ore(env):
    eef_quat = env.sim.data.get_body_xquat(env.robots[0].robot_model.eef_name["right"])
    return eef_quat


def get_all_quat(q_start, q_goal, time_steps):
    times = [0, 1]  
    
    key_rots = R.from_quat([q_start, q_goal])
    slerp = Slerp(times, key_rots)

    interp_times = np.linspace(0, 1, time_steps)
    interp_rots = slerp(interp_times)

    quaternions = interp_rots.as_quat()  # shape (100, 4)
    return quaternions


def get_all_drots(quats, time_steps):
    drot_list = []

    for t in range(1, time_steps):
        # Rotation from previous to current
        r_prev = R.from_quat(quats[t - 1])
        r_curr = R.from_quat(quats[t])

        # Compute relative rotation: r_delta = r_prev.inv() * r_curr
        r_delta = r_curr * r_prev.inv()

        # Convert to rotation vector (axis-angle) -> drot
        drot = r_delta.as_rotvec()  # shape (3,)
        drot_list.append(drot)

    # To match number of timesteps
    drot_list = [np.zeros(3)] + drot_list
    
    return np.array(drot_list)


# --------------------------------------------------------------------------------

def main():
    env = create_env()

    # 2. Load DMP
    hdf5_files = [r"C:\Users\Admin\robosuite_demos\new_full_demo\demo.hdf5"]
    fixed_timestep_count = 100


    all_dmp_models, _, all_original_traj, all_original_demo_eff_pos = generate_DMP_trajectories(hdf5_files, fixed_timestep_count)
    
    
    dmp_model = all_dmp_models[0]

    # 3. Get start and goal
    eef_start = get_eff_pos(env)
    goal_pos = get_goal_pos(env, "RoundNut_handle_site") 

    # 4. Get nut orientation and eff ore
    goal_ore =  get_goal_ore(env, "RoundNut_main")
    initial_eff_ore = get_initial_eff_ore(env)


    # get the goal state from original demo
    (x, y, z) = all_original_demo_eff_pos[0][-1]
    # dmp_model.start_y = eef_start
    goal_pos[2] = z
    dmp_model.goal_y  = goal_pos
    # dmp_model.configure(start_y=eef_start, goal_y=goal_pos)
    time_steps, dmp_trajectory = dmp_model.open_loop()
    dmp_trajectory[-1] = goal_pos 

    all_dmp_traj = []
    all_dmp_traj.append((time_steps, dmp_trajectory))

    # plot_spline_and_DMP_generated_trajectories_3D(all_original_traj, all_dmp_traj)
    # plot_spline_and_DMP_generated_trajectories(all_original_traj, all_dmp_traj)

    # get all rotations
    quaternions = get_all_quat(initial_eff_ore, goal_ore, len(time_steps))
    drots = get_all_drots(quaternions, len(time_steps))

    grasp = np.array([-1], dtype=np.float32)  # Gripper open

    for idx, pos in enumerate(dmp_trajectory):
        current_pos = get_eff_pos(env)
        dpos = pos - current_pos
        
        action = np.concatenate([dpos, drots[idx], grasp])
        env.step(action)
        env.render()
        time.sleep(0.05)

    for _ in range(200):
        current_pos = get_eff_pos(env)
        dpos = goal_pos - current_pos
        if np.linalg.norm(dpos) < 0.002:
            break
        action = np.concatenate([dpos, drots[-1], grasp])
        env.step(action)
        env.render()
        time.sleep(0.05)

if __name__ == "__main__":
    main()