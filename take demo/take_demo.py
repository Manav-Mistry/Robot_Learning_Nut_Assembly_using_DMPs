import os
import time
import h5py
import pygame
import numpy as np

import robosuite as suite
from robosuite import load_composite_controller_config
from robosuite.wrappers import VisualizationWrapper
from robosuite.devices import Keyboard
# from scipy.spatial.transform import Rotation as R
from utils import move_nuts_with_random_y_safe, rotate_nuts_in_env

np.random.seed(42)

def check_pygame_keys():
    marked, quit_signal = False, False
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                quit_signal = True
            elif event.key == pygame.K_m:
                marked = True
    return marked, quit_signal

# ----------------------------
# 1. Init Environment & Keyboard
# ----------------------------
pygame.init()
pygame.display.set_mode((300, 100))

controller_config = load_composite_controller_config(robot="Kinova3")

env = suite.make(
    env_name="NutAssembly",
    robots="Kinova3",
    controller_configs=controller_config,
    has_renderer=True,
    renderer="mjviewer",
    has_offscreen_renderer=False,
    render_camera="agentview",
    ignore_done=True,
    use_camera_obs=False,
    reward_shaping=True,
    control_freq=20,
)
env = VisualizationWrapper(env)



controller = Keyboard(env=env)
controller.start_control()

env.reset()


rotate_nuts_in_env(env, angle_degrees=180, axis="z")
# move_nuts_between_eef_and_pegs(env)
move_nuts_with_random_y_safe(env)

site_id = env.sim.model.site_name2id("gripper0_right_grip_site")

# ----------------------------
# 2. HDF5 Setup
# ----------------------------
timestamp = time.strftime("%Y%m%d_%H%M%S")
hdf5_path = f"new_full_demo_keyboard_{timestamp}.hdf5"
f = h5py.File(hdf5_path, "w")
data_group = f.create_group("data")
data_group.attrs["date"] = time.strftime("%Y-%m-%d")
data_group.attrs["time"] = time.strftime("%H:%M:%S")
data_group.attrs["env"] = "NutAssembly"
data_group.attrs["repository_version"] = "keyboard_eef_dmp_v1"

# ----------------------------
# 3. Demo Recording Loop
# ----------------------------
print("\n🎮 Keyboard Control Ready — Press [Q] to finish demo, [M] to mark steps.")

demo_index = 1
states = [] 
actions = []
marked_steps = []

# obs = env.reset()
step = 0

while True:
    state = controller.get_controller_state()
    marked, quit_signal = check_pygame_keys()

    if marked:
        print(f"⭐ Step {step} marked")
        marked_steps.append(step)

    if quit_signal or state["reset"]:
        print(f"🛑 Ending demo after {step} steps.")
        break

    if controller._enabled:
        # Build action: [dpos, drot, grasp]
        dpos, drot = controller._postprocess_device_outputs(state["dpos"], state["raw_drotation"])
        grasp = np.array([1.0 if state["grasp"] else -1.0])
        action = np.concatenate([dpos, drot, grasp])
        env.step(action)

        # Collect state: [qpos + qvel + eef]
        mujoco_state = env.sim.get_state().flatten()
        eef_pos = env.sim.data.site_xpos[site_id].copy()
        full_state = np.concatenate([mujoco_state, eef_pos])

        states.append(full_state)
        # only append end effector states
        # states.append(eef_pos)
        actions.append(action)

        env.render()
        time.sleep(0.05)
        step += 1

# ----------------------------
# 4. Save to HDF5
# ----------------------------
demo_group = data_group.create_group(f"demo{demo_index}")
demo_group.create_dataset("states", data=np.array(states))
demo_group.create_dataset("actions", data=np.array(actions))
demo_group.attrs["model_file"] = env.sim.model.get_xml()

f.close()
env.close()
pygame.quit()

print(f"\n✅ Demo saved to {hdf5_path}")
