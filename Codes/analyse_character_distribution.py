# import os
# import pandas as pd
# import matplotlib.pyplot as plt

# # Function to read text file and extract true labels
# def read_and_extract_labels(file_path):
#     df = pd.read_csv(file_path, sep='\t', header=None, names=['file_name', 'true_label'])
#     return df['true_label']

# # Function to calculate character frequency for all labels
# def calculate_char_frequency(labels):
#     char_freq = {}
#     for label in labels:
#         for char in label:
#             char_freq[char] = char_freq.get(char, 0) + 1
#     return char_freq

# # Function to plot character distribution and save as image
# def plot_char_distribution(char_freq, file_name, save_path):
#     sorted_char_freq = sorted(char_freq.items(), key=lambda x: x[0])
#     chars, freqs = zip(*sorted_char_freq)
#     plt.figure(figsize=(10, 6))
#     bars = plt.bar(chars, freqs)
#     plt.title(f'Character Distribution for Text File: {file_name}')
#     plt.xlabel('Character')
#     plt.ylabel('Frequency')

#     # Annotate each bar with the frequency count
#     for bar, freq in zip(bars, freqs):
#         plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 0.5,
#                  f'{freq}', ha='center', va='bottom', color='black', fontsize=10)

#     # Save the plot
#     save_file_path = os.path.join(save_path, f'{file_name}_char_distribution.png')
#     plt.savefig(save_file_path)
#     plt.close()

# # Analyze character distribution for each text file
# def analyze_character_distribution(files, save_path):
#     for file_path in files:
#         if not file_path:
#             continue
#         true_labels = read_and_extract_labels(file_path)
#         char_freq = calculate_char_frequency(true_labels)
#         plot_char_distribution(char_freq, os.path.basename(file_path), save_path)

# # Prompt user to input file paths containing the text files
# file_paths = []
# while True:
#     file_path = input("Enter the file path (leave blank if no more files need to be analyzed): ")
#     if not file_path:
#         break
#     file_paths.append(file_path)

# # Prompt user to input directory to save plots
# save_directory = input("Enter the directory to save the plots: ")

# # Analyze character distribution
# analyze_character_distribution(file_paths, save_directory)

import os
import pandas as pd
import string

# Function to read text file and extract true labels
def read_and_extract_labels(file_path):
    df = pd.read_csv(file_path, sep='\t', header=None, names=['file_name', 'true_label'])
    return df['true_label']

# Function to initialize character frequency with all possible characters
def initialize_char_frequency():
    char_freq = {char: 0 for char in string.ascii_uppercase + string.digits}
    return char_freq

# Function to calculate character frequency for all labels
def calculate_char_frequency(labels):
    char_freq = initialize_char_frequency()
    for label in labels:
        for char in label:
            if char in char_freq:  # Ensure only valid characters are counted
                char_freq[char] += 1
    return char_freq

# Analyze character frequency for each text file and combine into one table
def analyze_and_combine_character_frequencies(files):
    combined_freq = {char: [] for char in string.ascii_uppercase + string.digits}
    file_names = []
    
    for file_path in files:
        if not file_path:
            continue
        true_labels = read_and_extract_labels(file_path)
        char_freq = calculate_char_frequency(true_labels)
        for char in combined_freq.keys():
            combined_freq[char].append(char_freq.get(char, 0))
        file_names.append(os.path.basename(file_path).split('.')[0])

    return combined_freq, file_names

# Save combined character frequencies to a single CSV file
def save_combined_char_frequency_table(combined_freq, file_names, save_path):
    df = pd.DataFrame.from_dict(combined_freq, orient='index', columns=file_names)
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'Character'}, inplace=True)
    save_file_path = os.path.join(save_path, 'combined_char_frequency.csv')
    df.to_csv(save_file_path, index=False)

# Prompt user to input file paths containing the text files
file_paths = []
while True:
    file_path = input("Enter the file path (leave blank if no more files need to be analyzed): ")
    if not file_path:
        break
    file_paths.append(file_path)

# Prompt user to input directory to save the combined table
save_directory = input("Enter the directory to save the combined table: ")

# Analyze character frequency and save combined table
combined_freq, file_names = analyze_and_combine_character_frequencies(file_paths)
save_combined_char_frequency_table(combined_freq, file_names, save_directory)

print("Character frequency analysis completed and saved to combined_char_frequency.csv")
