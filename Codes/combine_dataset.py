import os
import shutil

def combine_datasets(input_folders, output_folder):
    """
    Combine multiple dataset folders into a single folder.

    This function takes a list of input folders containing datasets and merges them into a single output folder. 
    It iterates over each input folder, copies all files from each folder to the output folder, and maintains 
    the directory structure.

    Args:
        input_folders (list): A list of strings, each representing the path to an input dataset folder.
        output_folder (str): The path to the output folder where the combined dataset will be stored.

    Returns:
        None
    """
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
