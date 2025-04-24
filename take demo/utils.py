# utils.py

import numpy as np
from scipy.spatial.transform import Rotation as R

def rotate_nuts_in_env(env, angle_degrees=180, axis="z"):
    r_rot = R.from_euler(axis, angle_degrees, degrees=True)
    quat = np.roll(r_rot.as_quat(), 1)

    for nut_name in ["SquareNut_joint0", "RoundNut_joint0"]:
        pos = env.sim.data.get_joint_qpos(nut_name)[:3]
        env.sim.data.set_joint_qpos(nut_name, np.concatenate([pos, quat]))
    
    env.sim.forward()


def move_nuts_with_random_y_safe(env, peg1_body="peg1", peg2_body="peg2", arm="right", margin=0.02):
    eef_xy = env.sim.data.site_xpos[env.robots[0].eef_site_id[arm]].copy()[:2]
    peg1_xy = env.sim.data.get_body_xpos(peg1_body)[:2]
    peg2_xy = env.sim.data.get_body_xpos(peg2_body)[:2]

    square_x = (eef_xy[0] + peg1_xy[0]) / 10
    round_x = (eef_xy[0] + peg2_xy[0]) / 3

    def safe_rand(a, b, m):
        low, high = sorted([a, b])
        return np.random.uniform(low + m, high - m)

    square_y = safe_rand(eef_xy[1], peg1_xy[1], margin)
    round_y = safe_rand(eef_xy[1], peg2_xy[1], margin)

    sq = env.sim.data.get_joint_qpos("SquareNut_joint0")
    rd = env.sim.data.get_joint_qpos("RoundNut_joint0")

    env.sim.data.set_joint_qpos("SquareNut_joint0", np.concatenate([[square_x, square_y, sq[2]], sq[3:]]))
    env.sim.data.set_joint_qpos("RoundNut_joint0", np.concatenate([[round_x, round_y, rd[2]], rd[3:]]))
    env.sim.forward()
