import robosuite as suite
from robosuite import load_composite_controller_config

# Set up the environment
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

# Site names based on your XML
site_names = ["RoundNut_handle_site", "SquareNut_handle_site"]

print("📍 Handle Site Positions:\n")

for site_name in site_names:
    if site_name in env.sim.model.site_names:
        site_id = env.sim.model.site_name2id(site_name)
        pos = env.sim.data.site_xpos[site_id].copy()
        print(f" - {site_name}: {pos}")
    else:
        print(f" - ⚠️ Site '{site_name}' not found!")

env.close()
