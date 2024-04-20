import os
from PIL import Image

def check_labels(folder_path, labels_file):
    # Open the labels file
    with open(labels_file, 'r') as file:
        labels = file.readlines()
    
    # Iterate through each line in the labels file
    for line in labels:
        # Split the line into filename and label
        filename, label = line.strip().split('\t')
        
        # Open and display the image
        image_path = os.path.join(folder_path, filename)
        image = Image.open(image_path)
        image.show()
        
        # Print the filename and label
        print(f"Image: {filename}, Label: {label}")

# Prompt for the path to the folder containing your images
folder_path = input("Enter the path to the folder containing your images: ").strip()

# Prompt for the filename of the labels file
labels_file = input("Enter the filename of the labels file (e.g., labels.txt): ").strip()

# Check the labels
check_labels(folder_path, labels_file)
