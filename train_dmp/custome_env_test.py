import numpy as np
import robosuite as suite
from reproduce_helper import *

env = suite.make(
    env_name="NutAssemblyThreeNuts",
    # ThreeNutAssembly
    robots="Kinova3",
    has_renderer=True,
    renderer="mjviewer",
    has_offscreen_renderer=False,
    render_camera="agentview",
    ignore_done=True,
    use_camera_obs=False,
    reward_shaping=True,
    control_freq=20,
)

env.reset()
rotate_three_nuts_in_env(env)
# move_three_nuts_with_random_y_safe(env)


for i in range(500):
    action = np.random.randn(*env.action_spec[0].shape) * 0.1
    obs, reward, done, info = env.step(action)  # take action in the environment

# obs = env._get_observations()
obs = env.sim.get_state()
# for k, v in obs.items():
#     print(f"{k}: {v.shape}")

print(obs)