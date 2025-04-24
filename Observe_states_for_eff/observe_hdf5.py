import h5py
import numpy as np

# Load your demonstration file
hdf5_path = r"C:\Users\Admin\robosuite_demos\demo_going_from_top\demo.hdf5"  # Change this to your actual file path

with h5py.File(hdf5_path, "r") as f:
    # List available episodes
    episodes = list(f["data"].keys())
    print("Available episodes:", episodes) # output: ['demo_1']

    # Load first (or only) demonstration
    ep = f["data"][episodes[0]] 
    print(ep.keys())

    print(ep["actions"].shape)
    print(ep["states"].shape)
    
    # Get the full trajectory
    actions = ep["actions"][:]                # [T, action_dim]
    states = ep["states"][:]                  # [T, state_dim]

    # print("Trajectory length:", len(actions))
    gripper_action = actions[:, 6]
    print("First time when gripper is open")
    print(np.where(gripper_action == -1)[0][0])

    print("First time when gripper is close")
    print(np.where(gripper_action == 1)[0][0])

    print("Transition for : means finding a point where gripper free the object")
    transitions = np.where((gripper_action[:-1] == 1) & (gripper_action[1:] == -1))[0]

    if transitions.size > 0:
        first_transition_idx = transitions[0] + 1  # +1 because transition happens at the next index
        print("First -1 to 1 transition at index:", first_transition_idx)

    print("states vector lenght:", len(states))
    print("First index from states vector: ", states[0])

    print("-------------- observe all end effector states --------------------")
    # for state in states:
    #     print(state[-3:])


