import os
import pandas as pd
import random
import shutil
from collections import Counter

# Read the text file containing the labels and files
def read_labels(file_path):
    df = pd.read_csv(file_path, sep='\t', header=None, names=['file_name', 'true_label'])
    return df

# Obtain a unique list of the number plates
def get_unique_labels(df):
    unique_labels = df['true_label'].unique()
    return unique_labels

# Function to calculate character frequency
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

    train_df, val_df, test_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    for label in labels:
        label_images = df[df['true_label'] == label]

        if len(train_df) + len(label_images) <= train_limit:
            train_df = pd.concat([train_df, label_images])
        elif len(val_df) + len(label_images) <= val_limit:
            val_df = pd.concat([val_df, label_images])
        else:
            test_df = pd.concat([test_df, label_images])

    # Check if we need to adjust due to any overflows
    if len(train_df) < train_limit:
        deficit = train_limit - len(train_df)
        extra_images = test_df.iloc[:deficit]
        train_df = pd.concat([train_df, extra_images])
        test_df = test_df.iloc[deficit:]

    if len(val_df) < val_limit:
        deficit = val_limit - len(val_df)
        extra_images = test_df.iloc[:deficit]
        val_df = pd.concat([val_df, extra_images])
        test_df = test_df.iloc[deficit:]

    return train_df, val_df, test_df

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
    output_folder = os.path.dirname(file_path)

    df = read_labels(file_path)
    train_df, val_df, test_df = split_labels_balanced(df)

    train_df.to_csv(os.path.join(output_folder, 'train.txt'), sep='\t', index=False, header=False)
    val_df.to_csv(os.path.join(output_folder, 'val.txt'), sep='\t', index=False, header=False)
    test_df.to_csv(os.path.join(output_folder, 'test.txt'), sep='\t', index=False, header=False)

    copy_images(train_df, image_folder, os.path.join(output_folder, 'train'))
    copy_images(val_df, image_folder, os.path.join(output_folder, 'val'))
    copy_images(test_df, image_folder, os.path.join(output_folder, 'test'))

    print("Data has been split and saved to train.txt, val.txt, and test.txt")
    print("Images have been copied to train, val, and test folders")

if __name__ == "__main__":
    main()