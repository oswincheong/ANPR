import os
import shutil

def combine_datasets(input_folders, output_folder):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Iterate over each input folder
    for folder in input_folders:
        # Get a list of all files in the current input folder
        file_list = os.listdir(folder)
        # Filter out directories (if any)
        file_list = [file for file in file_list if os.path.isfile(os.path.join(folder, file))]
        
        # Copy each file to the output folder
        for file in file_list:
            shutil.copy(os.path.join(folder, file), os.path.join(output_folder, file))

def prompt_for_folders():
    input_folders = []
    while True:
        folder = input("Enter the path to a dataset folder (leave empty to finish): ").strip()
        if not folder:
            break
        input_folders.append(folder)
    return input_folders

# Prompt user to input paths for dataset folders
input_folders = prompt_for_folders()

# Prompt user to input path for the output folder
output_folder = input("Enter the path for the combined dataset folder: ").strip()

# Combine datasets
combine_datasets(input_folders, output_folder)
