import robosuite as suite
from robosuite import load_composite_controller_config
from robosuite.wrappers import VisualizationWrapper
from scipy.spatial.transform import Rotation as R

# 1. Load controller and environment
controller_config = load_composite_controller_config(robot="Kinova3")

env = suite.make(
    env_name="NutAssembly",
    robots="Kinova3",
    controller_configs=controller_config,
    has_renderer=True,
    render_camera="agentview",
    has_offscreen_renderer=False,
    use_camera_obs=False,
    reward_shaping=True,
    control_freq=20,
)

env = VisualizationWrapper(env)
env.reset()

# 2. Get the square nut orientation
nut_body = "SquareNut_main"
if nut_body in env.sim.model.body_names:
    nut_quat = env.sim.data.get_body_xquat(nut_body).copy()
    rot_matrix = R.from_quat(nut_quat).as_matrix()
    
    print(f"✅ Nut Quaternion (xyzw): {nut_quat}")
    print(f"🧭 Rotation Matrix:\n{rot_matrix}")

    print("End Effector Quat: ")
    eef_quat = env.sim.data.get_body_xquat(env.robots[0].robot_model.eef_name["right"])
    print(eef_quat)
else:
    print(f"⚠️ Nut body '{nut_body}' not found!")

env.close()
