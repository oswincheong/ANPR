import os

# Function to extract true label from image file name
def extract_true_label(file_name):
    parts = file_name.split('_')
    if len(parts) >= 3:
        return parts[-1].split('.')[0]
    else:
        return None

# Prompt user to input folder path containing the image files
image_dir = input("Enter the folder path containing the image files: ")

# Validate if the input directory exists
while not os.path.isdir(image_dir):
    print("Invalid directory path. Please try again.")
    image_dir = input("Enter the folder path containing the image files: ")

# Prompt user to input the desired output text file name
output_file_name = input("Enter the name of the output text file (e.g., labels.txt): ")

# Prompt user to input the path where to save the output text file
output_file_path = input("Enter the path where you want to save the output text file: ")

# Construct the full output file path
output_file = os.path.join(output_file_path, output_file_name)

# Open output file in write mode
with open(output_file, 'w') as f_out:
    # Iterate over each file in the directory
    for file_name in os.listdir(image_dir):
        # Check if file is a JPEG image
        if file_name.endswith('.jpg'):
            true_label = extract_true_label(file_name)
            if true_label:
                # Write file name and true label to output file
                f_out.write(f"{file_name} {true_label}\n")

print("Labels extraction completed. Output file saved at:", output_file)
