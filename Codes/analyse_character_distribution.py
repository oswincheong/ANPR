import os
import pandas as pd
import matplotlib.pyplot as plt

# Function to read text file and extract true labels
def read_and_extract_labels(file_path):
    df = pd.read_csv(file_path, sep='\t', header=None, names=['file_name', 'true_label'])
    return df['true_label']

# Function to calculate character frequency for all labels
def calculate_char_frequency(labels):
    char_freq = {}
    for label in labels:
        for char in label:
            char_freq[char] = char_freq.get(char, 0) + 1
    return char_freq

# Function to plot character distribution
def plot_char_distribution(char_freq, file_name):
    sorted_char_freq = sorted(char_freq.items(), key=lambda x: x[0])
    chars, freqs = zip(*sorted_char_freq)
    plt.figure(figsize=(10, 6))
    bars = plt.bar(chars, freqs)
    plt.title(f'Character Distribution for Text File: {file_name}')
    plt.xlabel('Character')
    plt.ylabel('Frequency')

    # Annotate each bar with the frequency count
    for bar, freq in zip(bars, freqs):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 0.5,
                 f'{freq}', ha='center', va='bottom', color='black', fontsize=10)

    plt.show()

# Analyze character distribution for each text file
def analyze_character_distribution(files):
    for file_path in files:
        if not file_path:
            continue
        true_labels = read_and_extract_labels(file_path)
        char_freq = calculate_char_frequency(true_labels)
        plot_char_distribution(char_freq, os.path.basename(file_path))

# Prompt user to input file paths containing the text files
file_paths = []
while True:
    file_path = input("Enter the file path (leave blank if no more files need to be analyzed): ")
    if not file_path:
        break
    file_paths.append(file_path)

# Analyze character distribution
analyze_character_distribution(file_paths)
