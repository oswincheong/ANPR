import os
import shutil

def split_images_by_name():
    # Prompt user to input the path to the dataset folder
    original_folder = input("Enter the path to the dataset folder: ").strip()

    # Create two new folders for the split
    original_folder = os.path.abspath(original_folder)
    original_images_folder = os.path.join(original_folder, 'original')
    warped_images_folder = os.path.join(original_folder, 'warped')
    os.makedirs(original_images_folder, exist_ok=True)
    os.makedirs(warped_images_folder, exist_ok=True)

    # Get a list of all files in the original folder
    file_list = os.listdir(original_folder)
    # Filter out directories (if any)
    file_list = [file for file in file_list if os.path.isfile(os.path.join(original_folder, file))]

    # Move files to the appropriate folder based on their names
    for file in file_list:
        if file.startswith("original"):  # Adjust the condition based on your naming convention
            shutil.move(os.path.join(original_folder, file), os.path.join(original_images_folder, file))
        else:
            shutil.move(os.path.join(original_folder, file), os.path.join(warped_images_folder, file))

split_images_by_name()
