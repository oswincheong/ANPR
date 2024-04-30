import os
import shutil

def split_dataset():
    # Prompt user to input the path to the dataset folder
    original_folder = input("Enter the path to the dataset folder: ").strip()

    # Create two new folders for the split
    train_folder = os.path.join(original_folder, 'train')
    test_folder = os.path.join(original_folder, 'valid')
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    # Get a list of all files in the original folder
    file_list = os.listdir(original_folder)
    # Filter out directories (if any)
    file_list = [file for file in file_list if os.path.isfile(os.path.join(original_folder, file))]

    # Prompt user to input the split ratio
    split_ratio = float(input("Enter the split ratio (e.g., 0.8 for 80-20 split): ").strip())

    # Calculate the number of files to move to the training set
    num_train_files = int(len(file_list) * split_ratio)

    # Move files to the train folder
    for file in file_list[:num_train_files]:
        shutil.move(os.path.join(original_folder, file), os.path.join(train_folder, file))

    # Move remaining files to the test folder
    for file in file_list[num_train_files:]:
        shutil.move(os.path.join(original_folder, file), os.path.join(test_folder, file))

split_dataset()
