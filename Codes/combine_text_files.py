import os

def combine_text_files(input_files, output_folder, output_filename):
    """
    Combine the contents of multiple text files into a single text file.

    Args:
        input_files (list): A list of paths to input text files.
        output_folder (str): The path to the output folder where the combined text file will be stored.
        output_filename (str): The filename of the output text file.

    Returns:
        None
    """
    # Combine contents of input files
    combined_content = ''
    for file_path in input_files:
        with open(file_path, 'r') as file:
            combined_content += file.read() + '\n'

    # Write combined content to output file
    output_path = os.path.join(output_folder, output_filename)
    with open(output_path, 'w') as output_file:
        output_file.write(combined_content)

def prompt_for_files_and_output():
    """
    Prompt the user to input paths for multiple text files and specify the output folder and filename.

    Returns:
        list: A list of paths to input text files.
        str: The path to the output folder.
        str: The filename of the output text file.
    """
    input_files = []
    while True:
        file_path = input("Enter the path to an input text file (leave blank to finish): ").strip()
        if not file_path:
            break
        input_files.append(file_path)
    output_folder = input("Enter the path for the output folder: ").strip()
    output_filename = input("Enter the filename for the output text file (eg: labels.txt): ").strip()
    return input_files, output_folder, output_filename

# Prompt user to input paths for text files, output folder, and output filename
input_files, output_folder, output_filename = prompt_for_files_and_output()

# Combine the contents of the text files
combine_text_files(input_files, output_folder, output_filename)

print("Combined text files successfully and saved to", os.path.join(output_folder, output_filename))
