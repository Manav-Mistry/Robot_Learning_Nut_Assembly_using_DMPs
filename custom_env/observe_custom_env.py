import numpy as np
import robosuite as suite
from scipy.spatial.transform import Rotation as R

def rotate_three_nuts_in_env(env, angle_degrees=180, axis="z"):
    r_rot = R.from_euler(axis, angle_degrees, degrees=True)
    quat = np.roll(r_rot.as_quat(), 1)  # Mujoco uses wxyz

    nut_joints = ["SquareNut_joint0", "RoundNut1_joint0", "RoundNut2_joint0"]

    for joint in nut_joints:
        pos = env.sim.data.get_joint_qpos(joint)[:3]
        env.sim.data.set_joint_qpos(joint, np.concatenate([pos, quat]))

    env.sim.forward()


env = suite.make(
    env_name="NutAssemblyThreeNuts", #NutAssemblyRound NutAssemblyThreeNuts
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
# rotate_nuts_in_env(env)
# with open("three_nut_env.xml", "w") as f:
#     f.write(env.model.get_xml())
# rotate_three_nuts_in_env(env, angle_degrees=180, axis="z")

for i in range(100):
    action = np.random.randn(*env.action_spec[0].shape) * 0.1
    obs, reward, done, info = env.step(action)  # take action in the environment

# obs = env._get_observations()
# obs = env.sim.get_state()
# for k, v in obs.items():
#     print(f"{k}: {v.shape}")

# print(obs)