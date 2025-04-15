import h5py

# Path to your hdf5 file
file_path = r'C:\Users\Admin\robosuite_demos\demo_split_0_to_566.hdf5'  # Change this if your filename is different

with h5py.File(file_path, 'r') as f:
    data_group = f['data']

    for demo_key in data_group:
        if demo_key.startswith('demo'):
            demo_group = data_group[demo_key]
            print(f"\n--- {demo_key} ---")

            states = demo_group['states'][:]
            actions = demo_group['actions'][:]

            for t in range(len(states)):
                print(f"Time step {t}:")
                print(f"  State:  {states[t]}")
                print(f"  Action: {actions[t]}")
