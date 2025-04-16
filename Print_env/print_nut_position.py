import robosuite as suite
from robosuite import load_composite_controller_config

# Set up environment
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

# List of known nut object bodies
nut_bodies = ["SquareNut_main", "RoundNut_main"]

print("📍 Nut Positions at Environment Reset:\n")

for nut in nut_bodies:
    if nut in env.sim.model.body_names:
        pos = env.sim.data.get_body_xpos(nut)
        print(f" - {nut}: {pos}")
    else:
        print(f" - ⚠️ '{nut}' not found in body_names!")

env.close()
