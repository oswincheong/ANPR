import os

def rename_files(folder_path, prefix):
    # List all files in the folder
    files = os.listdir(folder_path)
    
    # Filter out directories and only keep files
    files = [file for file in files if os.path.isfile(os.path.join(folder_path, file))]
    
    # Sort the files to ensure they are renamed in the correct order
    files.sort()
    
    # Rename each file with a sequential index
    for index, old_filename in enumerate(files, start=1):
        # Construct the new filename
        new_filename = f"{prefix}{index+1758:05d}.jpg"  # Adjust the extension if needed
        
        # Rename the file
        os.rename(os.path.join(folder_path, old_filename), os.path.join(folder_path, new_filename))
        
        print(f"Renamed '{old_filename}' to '{new_filename}'")

# Prompt for the path to the folder containing your files
folder_path = input("Enter the path to the folder containing your files: ").strip()

# Prefix for the new filenames
prefix = "combined"  # Change this prefix to whatever you need

# Rename the files
rename_files(folder_path, prefix)
