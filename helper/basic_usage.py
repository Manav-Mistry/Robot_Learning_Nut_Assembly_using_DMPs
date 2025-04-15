import numpy as np
import robosuite as suite
from robosuite import load_composite_controller_config


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
# reset the environment
env.reset()

for i in range(1000):
    action = np.random.randn(*env.action_spec[0].shape) * 0.1
    obs, reward, done, info = env.step(action)  # take action in the environment
    env.render()  # render on display