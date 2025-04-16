import numpy as np
import time
import robosuite as suite
from robosuite import load_composite_controller_config
from robosuite.wrappers import GymWrapper
from scipy.spatial.transform import Rotation as R

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

env = GymWrapper(env)

# 2. Load your DMP-generated trajectory (replace with your actual array)
# Shape: (T, 3) → x, y, z
dmp_traj = np.load("generated_dmp_xyz.npy")  # or load from memory
n_steps = dmp_traj.shape[0]

# 3. Settings
eef_ori = R.identity().as_matrix()           # Fixed orientation (no rotation)
grasp = np.array([1], dtype=np.float32)    # Open gripper

# 4. Reset & get initial position
env.reset()
initial_eef = env.env._eef_xpos
print("Initial EEF position:", initial_eef)

for i in range(n_steps):
    target = dmp_traj[i]
    current = env.env._eef_xpos
    dpos = target - current  # Compute delta to desired position

    # Use OSC_POSE controller: [dpos (3), drot (3), grasp (1)]
    drot = np.zeros(3)  # Optional: you can rotate if needed
    action = np.concatenate([dpos, drot, grasp])

    env.step(action)
    env.render()
    time.sleep(0.05)

env.close()
print("✅ DMP Trajectory executed in simulation.")
