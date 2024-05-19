import os
import re

# Prompt the user to enter the source directory
source_dir = input("Enter the path to your images directory: ").strip()

# Define a regex pattern to match filenames and extract information
pattern = re.compile(r'(.*)_(original|warped)_(.*)_(\d+)_(.*)')

# Create dictionaries to keep track of the new numbering for each type
counters = {
    'original': 1,
    'warped': 1
}

# Iterate over the files in the source directory
for filename in os.listdir(source_dir):
    # Full path to the file
    file_path = os.path.join(source_dir, filename)
    
    # Skip directories
    if os.path.isdir(file_path):
        continue

    # Match the filename pattern
    match = pattern.match(filename)
    if match:
        # Extract relevant parts from the filename
        image_type = match.group(2)
        true_label = match.group(5)
        
        # Get the current counter and increment it
        number = counters[image_type]
        counters[image_type] += 1
        
        # Construct new filename with zero-padded number
        new_filename = f"{image_type}_{str(number).zfill(5)}_{true_label}"
        
        # Full path to the new file
        new_file_path = os.path.join(source_dir, new_filename)
        
        # Rename the file
        os.rename(file_path, new_file_path)

print("Files have been successfully renamed and renumbered.")
