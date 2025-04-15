import h5py
import os

# Input and output file paths
input_file = r"C:\Users\Admin\robosuite_demos\1741904602_8723783\demo.hdf5"
output_file = r"C:\Users\Admin\robosuite_demos\demo_split_0_to_566.hdf5"

# Cutoff timestep (inclusive)
split_timestep = 566

with h5py.File(input_file, 'r') as f_in, h5py.File(output_file, 'w') as f_out:
    # Copy top-level 'data' group and its attributes
    data_in = f_in['data']
    data_out = f_out.create_group('data')  

    for attr_key, attr_val in data_in.attrs.items():
        data_out.attrs[attr_key] = attr_val

    for demo_key in data_in:
        if demo_key.startswith('demo'):
            demo_in = data_in[demo_key]
            demo_out = data_out.create_group(demo_key)

            # Copy the model_file attribute
            demo_out.attrs['model_file'] = demo_in.attrs['model_file']

            # Copy truncated datasets (0 to 566)
            demo_out.create_dataset('states', data=demo_in['states'][:split_timestep + 1])
            demo_out.create_dataset('actions', data=demo_in['actions'][:split_timestep + 1])

print(f"✅ New HDF5 file saved at: {output_file}")
