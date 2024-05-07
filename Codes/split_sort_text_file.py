import os

# Prompt for file path
file_path = input("Enter the path to the text file containing true labels: ")

# Check if the file exists
if not os.path.isfile(file_path):
    print("File does not exist.")
    exit()

# Read the text file and parse contents into a list of tuples
with open(file_path, "r") as file:
    lines = file.readlines()

data = [(line.split()[0], line.split()[1]) for line in lines]

# Prompt for ratios
train_ratio = float(input("Enter the ratio for the train set (e.g., 0.8): "))
valid_ratio = float(input("Enter the ratio for the validation set (e.g., 0.1): "))
test_ratio = float(input("Enter the ratio for the test set (e.g., 0.1): "))

# Calculate sizes for train, validation, and test sets
train_size = int(len(data) * train_ratio)
valid_size = int(len(data) * valid_ratio)
test_size = len(data) - train_size - valid_size

# Split the data into train, validation, and test sets
train_data = data[:train_size]
valid_data = data[train_size:train_size + valid_size]
test_data = data[train_size + valid_size:]

# Function to write data to file
def write_data_to_file(file_path, data):
    with open(file_path, "w") as file:
        for item in data:
            file.write(f"{item[0]}\t{item[1]}\n")

# Get directory path
directory = os.path.dirname(file_path)

# Write train data to file
train_file_path = os.path.join(directory, "train.txt")
write_data_to_file(train_file_path, train_data)

# Write validation data to file
valid_file_path = os.path.join(directory, "valid.txt")
write_data_to_file(valid_file_path, valid_data)

# Write test data to file
test_file_path = os.path.join(directory, "test.txt")
write_data_to_file(test_file_path, test_data)

print("Data split and saved successfully.")
