import robosuite as suite
from robosuite import load_composite_controller_config

# Load your environment (NutAssembly with Kinova3)
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

env.reset()

# Access the robot instance
robot = env.robots[0]

# Extract EEF site ID for the "right" arm
site_id = robot.eef_site_id["right"]
site_name = env.sim.model.site_id2name(site_id)
eef_pos = env.sim.data.site_xpos[site_id].copy()

# Print results
print(f"✅ EEF Site Name: {site_name}")
print(f"🆔 Site ID: {site_id}")
print(f"📍 EEF Position: {eef_pos}")

env.close()
