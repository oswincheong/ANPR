import os
import shutil

# Prompt the user to input paths and ratios
image_dir = input("Enter the path to the directory containing the images: ")
label_file = input("Enter the path to the sorted label text file: ")
output_dir = input("Enter the path to save the train, validation, and test sets: ")

# Create directories for train, validation, and test sets
train_dir = os.path.join(output_dir, "train")
valid_dir = os.path.join(output_dir, "valid")
test_dir = os.path.join(output_dir, "test")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(valid_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Read sorted label file
with open(label_file, "r") as file:
    lines = file.readlines()

# Determine number of images for each set
total_images = len(lines)
train_test_ratio = 0.8
train_valid_ratio = 0.8
num_train_test = int(train_test_ratio * total_images)
num_train = int(train_valid_ratio * num_train_test)
num_valid = num_train_test - num_train
num_test = total_images - num_train_test

# Function to filter image files
def is_image(filename):
    return filename.lower().endswith(('.jpg', '.jpeg', '.png'))

# Filter image files in the image directory
image_files = [f for f in os.listdir(image_dir) if is_image(f)]

# Split images into train and test sets
for i, line in enumerate(lines):
    image_name, label = line.strip().split('\t')
    if image_name in image_files:
        if i < num_train_test:
            destination = train_dir if i < num_train else valid_dir
        else:
            destination = test_dir
        source_path = os.path.join(image_dir, image_name)
        destination_path = os.path.join(destination, image_name)
        shutil.copyfile(source_path, destination_path)

print("Images have been split into train, validation, and test sets.")
