from robosuite import load_composite_controller_config
import h5py
import numpy as np
import robosuite as suite
from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper

controller_config = load_composite_controller_config(robot="Kinova3")

config = {
    "env_name": "NutAssembly",
    "robots": "Kinova3",
    "controller_configs": controller_config,
}


# 1. Initialize your Kinova3 NutAssembly env
env = suite.make(
        **config,
        has_renderer=True,
        renderer="mjviewer",
        has_offscreen_renderer=False,
        render_camera="agentview",
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
    )

# env = VisualizationWrapper(env)
# # 2. Setup vars
model = env.sim.model
data = env.sim.data
nq, nv = model.nq, model.nv
# print(nq, nv)
site_id = model.site_name2id("gripper0_right_grip_site")
# site_id = model.site_name2id("robot0_grip_site")  # double-check if custom

# 3. Load your demo.hdf5 and extract accurate EEF trajectories
hdf5_path = r"C:\Users\Admin\robosuite_demos\demo_split_0_to_566\demo.hdf5"
eef_trajs = {}

print(nq, nv)

with h5py.File(hdf5_path, 'r') as f:
    data_group = f['data']

    for demo_key in data_group:
        if demo_key.startswith("demo"):
            demo_group = data_group[demo_key]
            states = demo_group["states"][:]
            eef_positions = []

            for state in states:
                # qpos = state[:nq]

                # # Get raw qvel (could be too long)
                # qvel_raw = state[nq:]
                # qvel = qvel_raw[:25]
                qpos = state[:27]
                qvel = state[27:53]

                # Truncate if needed
                if len(qvel) > 25:
                    qvel = qvel[:25]
                
                # Restore sim state from qpos and qvel
                env.sim.set_state_from_flattened(np.concatenate([qpos, qvel]))
                env.sim.forward()

                # Extract EEF position
                eef_pos = data.site_xpos[site_id].copy()
                eef_positions.append(eef_pos)

            eef_trajs[demo_key] = np.array(eef_positions)
            print(f"{demo_key}: extracted {len(eef_positions)} EEF positions")

print("✅ All trajectories extracted!")