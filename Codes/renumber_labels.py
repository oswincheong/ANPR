# Prompt the user for the input file path and output file path
ground_truth_file = input("Enter the path to the ground truth file: ")
output_file = input("Enter the path for the output file: ")

# Read the ground truth file
with open(ground_truth_file, 'r') as file:
    lines = file.readlines()

# Parse and sort the lines by filename
entries = [line.strip().split('\t') for line in lines]
entries.sort(key=lambda x: x[0])

# Write the sorted and numbered entries to a new file
with open(output_file, 'w') as file:
    for i, (filename, label) in enumerate(entries, start=1):
        numbered_filename = f"{i:05d}_{filename}"
        file.write(f"{numbered_filename}\t{label}\n")

print(f"File has been written to {output_file}")
