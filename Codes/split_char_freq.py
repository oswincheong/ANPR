import os
import pandas as pd
import random
import shutil
from collections import Counter

# Read the text file containing the labels and files
def read_labels(file_path):
    df = pd.read_csv(file_path, sep='\t', header=None, names=['file_name', 'true_label'])
    return df

# Calculate character frequency for a given series of labels
def calculate_char_frequency(labels):
    char_freq = Counter()
    for label in labels:
        char_freq.update(label)
    return char_freq

# Ensure balanced distribution of characters and exact image split ratios
def split_labels_balanced(df, train_size=0.8, val_size=0.1, test_size=0.1, random_state=42):
    random.seed(random_state)
    labels = df['true_label'].unique().tolist()
    random.shuffle(labels)

    total_images = len(df)
    train_limit = int(total_images * train_size)
    val_limit = int(total_images * val_size)
    test_limit = total_images - train_limit - val_limit

    # Initialize DataFrames and character counters for each subset
    train_df, val_df, test_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    train_chars, val_chars, test_chars = Counter(), Counter(), Counter()

    # Assign labels to the smallest skew subset
    for label in labels:
        label_images = df[df['true_label'] == label]
        
        # Simulate addition to each subset and calculate the skewness
        options = []
        subsets = [
            ('train', train_df, train_chars, train_limit),
            ('val', val_df, val_chars, val_limit),
            ('test', test_df, test_chars, test_limit)
        ]
        for subset_name, subset_df, subset_chars, limit in subsets:
            if len(subset_df) + len(label_images) <= limit:
                temp_chars = subset_chars.copy()
                temp_chars.update(label)
                skewness = max(temp_chars.values()) - min(temp_chars.values())
                options.append((skewness, subset_name, subset_df, subset_chars))

        # Sort by skewness and choose the best subset
        options.sort()
        best_skewness, best_subset_name, best_subset_df, best_subset_chars = options[0]
        best_subset_df = pd.concat([best_subset_df, label_images])
        best_subset_chars.update(label)

        if best_subset_name == 'train':
            train_df, train_chars = best_subset_df, best_subset_chars
        elif best_subset_name == 'val':
            val_df, val_chars = best_subset_df, best_subset_chars
        elif best_subset_name == 'test':
            test_df, test_chars = best_subset_df, best_subset_chars

    # Return the dataframes
    return train_df, val_df, test_df

# Count images per label
def count_images_per_label(df):
    label_counts = Counter(df['true_label'])
    return label_counts

# Delete excess images for each label
def delete_excess_images(df, max_images_per_label, image_folder):
    label_counts = count_images_per_label(df)
    for label, count in label_counts.items():
        if count > max_images_per_label:
            excess_count = count - max_images_per_label
            excess_images = df[df['true_label'] == label].head(excess_count)
            for _, row in excess_images.iterrows():
                image_path = os.path.join(image_folder, row['file_name'])
                os.remove(image_path)
            df = df.drop(excess_images.index)
    return df

# Copy images to respective folders
def copy_images(df, src_folder, dest_folder):
    os.makedirs(dest_folder, exist_ok=True)
    for _, row in df.iterrows():
        src_file = os.path.join(src_folder, row['file_name'])
        dest_file = os.path.join(dest_folder, row['file_name'])
        shutil.copy(src_file, dest_file)

# Main function
def main():
    file_path = input("Enter the path to the text file containing the labels: ")
    image_folder = input("Enter the path to the folder containing the images: ")
    max_images_per_label = int(input("Enter the maximum number of images per label: "))
    output_folder = os.path.dirname(file_path)

    # Read labels and delete excess images
    df = read_labels(file_path)
    df = delete_excess_images(df, max_images_per_label, image_folder)

    # Split the data into train, val, and test sets
    train_df, val_df, test_df = split_labels_balanced(df)

    # Save the split labels to text files
    train_df.to_csv(os.path.join(output_folder, 'train.txt'), sep='\t', index=False, header=False)
    val_df.to_csv(os.path.join(output_folder, 'val.txt'), sep='\t', index=False, header=False)
    test_df.to_csv(os.path.join(output_folder, 'test.txt'), sep='\t', index=False, header=False)

    # Copy images to their respective folders
    copy_images(train_df, image_folder, os.path.join(output_folder, 'train'))
    copy_images(val_df, image_folder, os.path.join(output_folder, 'val'))
    copy_images(test_df, image_folder, os.path.join(output_folder, 'test'))

    print("Excess images deleted, data split and saved to train.txt, val.txt, and test.txt")
    print("Images have been copied to train, val, and test folders")

if __name__ == "__main__":
    main()
