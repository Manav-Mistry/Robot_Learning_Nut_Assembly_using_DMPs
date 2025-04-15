import h5py
import numpy as np
# Load your demonstration file
hdf5_path = r"C:\Users\Admin\robosuite_demos\1741724444_7722592\demo.hdf5"  # Change this to your actual file path

with h5py.File(hdf5_path, "r") as f:
    # List available episodes
    episodes = list(f["data"].keys())
    print("Available episodes:", episodes) # output: ['demo_1']

    # Load first (or only) demonstration
    ep = f["data"][episodes[0]] 
    print(ep.keys())

    print(ep["actions"].shape)
    print(ep["states"].shape)
    # # Get the full trajectory
    actions = ep["actions"][:]                  # [T, action_dim]
    # obs = ep["obs"]["robot0_eef_pos"][:]      # [T, 3]
    # states = ep["states"][:]                  # [T, state_dim]

    print("gripper status:", (actions[:, 6]))

    first_neg_one = np.where(actions[:, 6] == -1)[0][0] 
    first_pos_one = np.where(actions[:, 6] == 1)[0][0]

    print("first -1:",first_neg_one)
    print("first 1:",first_pos_one)
