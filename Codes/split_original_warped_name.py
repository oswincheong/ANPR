import os
import shutil

# Prompt the user to enter the source directory
source_dir = input("Enter the path to your images directory: ").strip()

# Define the paths for 'original' and 'warped' directories
original_dir = os.path.join(source_dir, 'original')
warped_dir = os.path.join(source_dir, 'warped')

# Create 'original' and 'warped' directories if they don't exist
os.makedirs(original_dir, exist_ok=True)
os.makedirs(warped_dir, exist_ok=True)

# Iterate over the files in the source directory
for filename in os.listdir(source_dir):
    # Full path to the file
    file_path = os.path.join(source_dir, filename)
    
    # Skip directories
    if os.path.isdir(file_path):
        continue
    
    # Check if the filename contains '_original_'
    if '_original_' in filename:
        # Move to 'original' directory
        shutil.move(file_path, os.path.join(original_dir, filename))
    else:
        # Otherwise, move to 'warped' directory
        shutil.move(file_path, os.path.join(warped_dir, filename))

print("Files have been successfully organized into 'original' and 'warped' folders.")
