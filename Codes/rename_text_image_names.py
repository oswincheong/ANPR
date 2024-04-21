def rename_image_names(input_file, output_file):
    """
    Rename image names in a text file to start with "combined00001" followed by a sequential number.

    Args:
        input_file (str): The path to the input text file.
        output_file (str): The path to the output text file where the modified content will be stored.

    Returns:
        None
    """
    with open(input_file, 'r') as f:
        content = f.read()

    # Split the content into lines
    lines = content.split('\n')

    # Modify each line to start with "combined00001" followed by a sequential number
    modified_lines = []
    for i, line in enumerate(lines, start=1):
        if line.strip():  # Check if line is not empty
            image_name, label = line.split('\t')
            new_image_name = f"combined{str(i).zfill(5)}.jpg"  # Format the number to have leading zeros
            modified_lines.append(f"{new_image_name}\t{label}")

    modified_content = '\n'.join(modified_lines)

    with open(output_file, 'w') as f:
        f.write(modified_content)

def prompt_for_file_paths():
    """
    Prompt the user to input the path for the input and output text files.

    Returns:
        str: The path to the input text file.
        str: The path to the output text file.
    """
    input_file = input("Enter the path to the input text file: ").strip()
    output_file = input("Enter the path for the output text file: ").strip()
    return input_file, output_file

# Prompt user to input paths for the input and output text files
input_file, output_file = prompt_for_file_paths()

# Rename image names in the input file and save the modified content to the output file
rename_image_names(input_file, output_file)

print("Image names modified and saved to", output_file)
