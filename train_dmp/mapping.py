import numpy as np
from robosuite import load_part_controller_config, load_composite_controller_config
import robosuite as suite
from scipy.spatial.transform import Rotation as R, Slerp
# from robosuite.controllers.parts.arm.osc import Controller


CONTROLLER_PATH = r"C:\Users\Admin\anaconda3\envs\robosuite\Lib\site-packages\robosuite\controllers\config\robots\default_kinova3.json"

# Load Robosuite environment
# controller_config = load_part_controller_config(default_controller="OSC_POSITION")
controller_config = load_composite_controller_config(robot="Kinova3")
# controller_config = Controller()

env = suite.make(
        env_name="NutAssembly",
        robots="Kinova3",
        controller_configs=controller_config,
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera="frontview",
        use_camera_obs=False,
        control_freq=5, 
    )

# NOTE : env.action(?)
def execute_trajectory(trajectory, ee_ori_resampled):
    for ee_pos, ee_ori in zip(trajectory, ee_ori_resampled):
        # 3D pos + 4D quaternion
        # desired_quat = np.array([0, 1, 0, 0])  # Placeholder
        action = np.concatenate([ee_pos, ee_ori])

        obs, reward, done, info = env.step(action)
        env.render()

# Mapping
def execute_plan_with_dmps(plan, dmp_models, all_ee_ori, all_original_time):
    action_to_dmp = {
        "assemble_hex_nut robot1 hexnut1 peg1 table": dmp_models[0],
        "assemble_square_nut robot1 squarenut1 peg2 table": dmp_models[1]
    }

    ee_ori_to_dmp = {
        "assemble_hex_nut robot1 hexnut1 peg1 table": all_ee_ori[0],
        "assemble_square_nut robot1 squarenut1 peg2 table": all_ee_ori[1]
    }

    original_time_to_dmp = {
        "assemble_hex_nut robot1 hexnut1 peg1 table": all_original_time[0],
        "assemble_square_nut robot1 squarenut1 peg2 table": all_original_time[1]
    }

    for action_str in plan:
        action = action_str.strip("()")
        dmp = action_to_dmp.get(action)
        ee_ori = ee_ori_to_dmp.get(action)
        original_time = original_time_to_dmp.get(action)

        if dmp is None:
            print(f"No DMP found for: {action}")
            continue
        T_gen, ee_trajectory = dmp.open_loop()

        #NOTE resample ee_ori
        rotations = R.from_quat(ee_ori)  # [x, y, z, w]
        slerp = Slerp(original_time, rotations)
        resampled_rotations = slerp(T_gen)
        ee_ori_resampled = resampled_rotations.as_quat() 


        execute_trajectory(ee_trajectory, ee_ori_resampled)


def main():
    from train_multiple_dmp_smoothing_resample import generate_DMP_trajectories

    hdf5_files = [
        r"C:\\Users\\Admin\\robosuite_demos\\1741904602_8723783\\demo.hdf5",
        r"C:\\Users\\Admin\\robosuite_demos\\1741724444_7722592\\demo.hdf5",
    ]
    EE_POS_IDX = slice(14, 17)

    EE_ORI_IDX = slice(17, 21)

    fixed_timestep_count = 100
    dmp_models, _, _, all_ee_ori, all_original_time = generate_DMP_trajectories(hdf5_files, fixed_timestep_count, EE_POS_IDX, EE_ORI_IDX)

    plan = [
        "(assemble_hex_nut robot1 hexnut1 peg1 table)",
        "(assemble_square_nut robot1 squarenut1 peg2 table)"
    ]

    execute_plan_with_dmps(plan, dmp_models, all_ee_ori, all_original_time)

if __name__ == "__main__":
    main()
