# Get input file path from user
input_file_path = input("Enter the path of the input file: ")

# Get output file path from user
output_file_path = input("Enter the path of the output file: ")

# Read the content of the input file
with open(input_file_path, "r") as input_file:
    data = input_file.read()

# Replace all occurrences of "original" with "warped"
modified_data = data.replace("warped", "combined")

# Write the modified content to the output file
with open(output_file_path, "w") as output_file:
    output_file.write(modified_data)

print("Content successfully replaced and saved to the output file.")
