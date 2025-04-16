import numpy as np
import time
import robosuite as suite
from robosuite import load_composite_controller_config
from robosuite.wrappers import VisualizationWrapper

from scipy.spatial.transform import Rotation as R
from utils import generate_DMP_trajectories

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

# 2. Load your DMP-generated trajectory (replace with your actual array)
# Shape: (T, 3) → x, y, z
hdf5_files = [
    r"C:\Users\Admin\robosuite_demos\friend\demo.hdf5", # for pick action
]  
fixed_timestep_count = 100  # resample

# all_dmp_models is the array of models after dmp.imitate() my demonstration
all_dmp_models, _, _ = generate_DMP_trajectories(hdf5_files, fixed_timestep_count)

# Write your code here:
# I need a code that configure the dmp model (start position: end-effector position, end position: squared nut handle postion)
# then reproduce the trajectory in simulation



