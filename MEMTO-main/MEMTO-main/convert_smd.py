import os
import numpy as np
import sys

# We use double backslashes to avoid path errors in Windows
source = r"C:\Users\satyajit sarma\Downloads\ServerMachineDataset\train"
target = r"C:\Users\satyajit sarma\Downloads\MEMTO-main\MEMTO-main\data\SMD\SMD"

print(f"DEBUG: Looking for files in: {source}")

try:
    if not os.path.exists(source):
        print(f"ERROR: The folder '{source}' was not found. Please check the spelling.")
        sys.exit()

    files = [f for f in os.listdir(source) if f.endswith('.txt')]
    print(f"DEBUG: Found {len(files)} files.")

    if len(files) == 0:
        print("ERROR: No .txt files found! Check if the files are directly in the 'train' folder.")
        sys.exit()

    if not os.path.exists(target):
        os.makedirs(target)
        print(f"DEBUG: Created target folder: {target}")

    all_data = []
    for f in sorted(files):
        # Using a generic loader that handles different separators
        d = np.genfromtxt(os.path.join(source, f), delimiter=',')
        all_data.append(d)
    
    final_matrix = np.concatenate(all_data, axis=0)
    save_path = os.path.join(target, 'SMD_train.npy')
    np.save(save_path, final_matrix)
    print(f"SUCCESS: Created {save_path}")
    print(f"Matrix shape: {final_matrix.shape}")

except Exception as e:
    print(f"AN UNEXPECTED ERROR OCCURRED: {e}")                                