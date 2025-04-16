import h5py

# Input and output file paths
input_file = r"C:\Users\Admin\robosuite_demos\revised_full_demo\demo.hdf5"
output_file = r"C:\Users\Admin\robosuite_demos\splitted_traj_for_revised_demo\demo_split_place.hdf5"

# Time step range (inclusive)
start_t = 534
end_t = 1056

with h5py.File(input_file, 'r') as f_in, h5py.File(output_file, 'w') as f_out:
    # Create top-level group and copy attributes
    data_in = f_in['data']
    data_out = f_out.create_group('data')

    for attr_key, attr_val in data_in.attrs.items():
        data_out.attrs[attr_key] = attr_val

    for demo_key in data_in:
        if demo_key.startswith('demo'):
            demo_in = data_in[demo_key]
            demo_out = data_out.create_group(demo_key)

            # Copy model_file attribute
            demo_out.attrs['model_file'] = demo_in.attrs['model_file']

            # Slice and save datasets: states and actions
            demo_out.create_dataset('states', data=demo_in['states'][start_t:end_t + 1])
            demo_out.create_dataset('actions', data=demo_in['actions'][start_t:end_t + 1])

print(f"✅ New HDF5 file saved from timestep {start_t} to {end_t} at: {output_file}")
