import os
import shutil

# Prompt the user to input paths and ratios
image_dir = input("Enter the path to the directory containing the images: ")
label_file = input("Enter the path to the sorted label text file: ")
train_test_dir = input("Enter the path to save the train and test sets: ")
train_dir = os.path.join(train_test_dir, "train")
valid_dir = os.path.join(train_test_dir, "valid")
test_dir = os.path.join(train_test_dir, "test")

# Create directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(valid_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Read sorted label file
with open(label_file, "r") as file:
    lines = file.readlines()

# Determine number of images for each set
total_images = len(lines)
train_ratio = float(input("Enter the ratio of images for the train set (e.g., 0.8): "))
valid_ratio = float(input("Enter the ratio of images for the validation set (e.g., 0.1): "))
test_ratio = 1 - train_ratio - valid_ratio
num_train = int(train_ratio * total_images)
num_valid = int(valid_ratio * total_images)
num_test = total_images - num_train - num_valid

# Iterate through sorted labels and copy images to appropriate directories
for i, line in enumerate(lines):
    image_name, label = line.strip().split('\t')
    if i < num_train:
        destination = train_dir
    elif i < num_train + num_valid:
        destination = valid_dir
    else:
        destination = test_dir
    source_path = os.path.join(image_dir, image_name)
    destination_path = os.path.join(destination, image_name)
    shutil.copyfile(source_path, destination_path)

print("Images have been split into train, validation, and test sets.")
