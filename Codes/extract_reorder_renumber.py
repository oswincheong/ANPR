import os
import re

# Prompt the user to enter the folder path containing the images
folder_path = input("Enter the path to the folder containing your images: ").strip()

# Ensure the folder path is valid
while not os.path.isdir(folder_path):
    print("Invalid folder path. Please enter a valid path.")
    folder_path = input("Enter the path to the folder containing your images: ").strip()

# List all image files in the folder
image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

# Define a regex pattern to extract the true label from the filename
pattern = re.compile(r'original_(\d+)_(.*)\.jpg')

# Function to extract true label from filename
def extract_true_label(filename):
    match = pattern.match(filename)
    if match:
        # Get the last part of the filename after the last underscore
        return match.group(2).rsplit('_', 1)[-1]
    return None

# Extract true labels
true_labels = []
for filename in image_files:
    true_label = extract_true_label(filename)
    if true_label:
        true_labels.append(true_label)

# Sort true labels alphabetically and renumber them
true_labels.sort()
renumbered_labels = {true_label: f'label_{i+1:05}' for i, true_label in enumerate(true_labels)}

# Create a list to hold the filename and true label pairs
file_label_pairs = []

# Iterate over the image files to extract true labels and renumber them
for filename in image_files:
    true_label = extract_true_label(filename)
    if true_label:
        renumbered_label = renumbered_labels[true_label]
        file_label_pairs.append((filename, renumbered_label))

# Prompt the user to enter the path for the output text file
output_file_path = input("Enter the path for the output text file: ").strip()

# Create the text file if it doesn't exist
if not os.path.exists(output_file_path):
    with open(output_file_path, 'w') as f:
        for file_name, renumbered_label in file_label_pairs:
            f.write(f"{file_name} {renumbered_label}\n")
    print("Text file has been successfully created.")
else:
    print("Text file already exists. No action taken.")
