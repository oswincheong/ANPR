import pandas as pd

# Paths to the text files
file1_path = r'C:\Users\Oswin\Desktop\FYP\ANPR-1\Datasets\Dataset_all_char_freq\original\sorted_label.txt'
file2_path = r'C:\Users\Oswin\Desktop\FYP\ANPR-1\Datasets\Dataset_all_char_freq\warped\sorted_labels.txt'

# Load the data into DataFrames
df1 = pd.read_csv(file1_path, sep='\t', header=None, names=['file_name', 'true_label'])
df2 = pd.read_csv(file2_path, sep='\t', header=None, names=['file_name', 'true_label'])

# Get unique number plates for each file separately
unique_labels_file1 = df1['true_label'].unique()
unique_labels_file2 = df2['true_label'].unique()

# Combine and get unique number plates across both files
combined_unique_labels = set(unique_labels_file1).union(set(unique_labels_file2))

# Count unique labels
unique_count_file1 = len(unique_labels_file1)
unique_count_file2 = len(unique_labels_file2)
combined_unique_count = len(combined_unique_labels)

print(f"Unique number plates in file 1: {unique_count_file1}")
print(f"Unique number plates in file 2: {unique_count_file2}")
print(f"Combined unique number plates: {combined_unique_count}")
