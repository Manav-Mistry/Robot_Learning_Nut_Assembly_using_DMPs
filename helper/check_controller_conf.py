import robosuite as suite
from robosuite import load_composite_controller_config
from robosuite.wrappers import VisualizationWrapper

# 1. Load controller config
controller_config = load_composite_controller_config(robot="Kinova3")

# Optional: modify config to ensure absolute position control
# controller_config["body_parts"]["right"]["absolute_position_control"] = True
# controller_config["body_parts"]["right"]["control_type"] = "position"
# controller_config["body_parts"]["right"]["input_type"] = "absolute"

# 2. Create environment
env = suite.make(
    env_name="NutAssembly",
    robots="Kinova3",
    controller_configs=controller_config,
    has_renderer=False,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    reward_shaping=True,
    control_freq=20,
)
env = VisualizationWrapper(env)

# 3. Print out the controller settings
print("\n✅ Controller Config for RIGHT Arm:\n")
for key, value in env.robots[0].composite_controller.part_controllers["right"].__dict__.items():
    print(f"{key}: {value}")
