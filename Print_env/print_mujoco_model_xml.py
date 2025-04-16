import h5py
import numpy as np

# Load your demonstration file
file_path = r"C:\Users\Admin\robosuite_demos\1741724444_7722592\demo.hdf5"  # Change this to your actual file path

with h5py.File(file_path, 'r') as f:
    data_group = f['data']

    for demo_name in data_group:
        if demo_name.startswith('demo'):
            demo_group = data_group[demo_name]

            # Get model_file attribute
            model_file = demo_group.attrs.get('model_file')
            
            if model_file:
                # Convert bytes to string if necessary
                if isinstance(model_file, bytes):
                    model_file = model_file.decode('utf-8')

                # Define filename to save the xml
                file_name = f"sample_mujoco_{demo_name}.xml"
                
                with open(file_name, "w", encoding='utf-8') as xml_file:
                    xml_file.write(model_file)
                
                print(f"Saved {file_name}")
            else:
                print(f"No model_file found in {demo_name}")