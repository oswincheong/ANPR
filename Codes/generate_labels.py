import os
from PIL import Image

# Function to get label input for an image
def get_label(image_filename):
    # # Open and display the image
    # image_path = os.path.join(folder_path, image_filename)
    # image = Image.open(image_path)
    # image.show()

    label = input(f"Enter label for image '{image_filename}': ")
    return label

# Path to the folder containing your images
folder_path = input("Enter the path to the folder containing your images: ")

# List all image files in the folder
image_files = [file for file in os.listdir(folder_path) if file.endswith(('jpg', 'jpeg', 'png', 'bmp'))]

# Create or open a text file to save the labels
output_filename = input("Enter the filename to save labels (e.g., labels.txt): ")
with open(output_filename, 'w') as file:
    # Iterate through each image file
    for image_file in image_files:
        # Get the label for the image
        label = get_label(image_file)
        
        # Write the image filename and label to the text file
        file.write(f"{image_file}\t{label}\n")

print("Labeling complete. Labels saved in", output_filename)
