# Add necessary imports
import os

# Prompt the user to input the file name of the true label text file
file_name = input("Enter the file name of the true label text file: ")

# Check if the file exists
if not os.path.isfile(file_name):
    print("Error: File not found.")
    exit()

# Read the true label text file
with open(file_name, "r") as file:
    lines = file.readlines()

# Parse the lines into image file names and labels
data = []
for line in lines:
    parts = line.strip().split()
    if len(parts) >= 2:
        data.append((parts[0], ' '.join(parts[1:])))
    else:
        print(f"Warning: Skipping line '{line.strip()}' - expected at least two parts.")

# Sort the data based on the labels
sorted_data = sorted(data, key=lambda x: x[1])

# Construct the file path for the output file
output_file = os.path.join(os.path.dirname(file_name), "sorted_" + os.path.basename(file_name))

# Write the sorted data back to a new text file
with open(output_file, "w") as file:
    for image_file, label in sorted_data:
        file.write(f"{image_file}\t{label}\n")

print(f"Sorted labels have been written to {output_file}.")
