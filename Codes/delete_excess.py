import os
import pandas as pd
import shutil
from collections import Counter

def read_labels(file_path):
    df = pd.read_csv(file_path, sep='\t', header=None, names=['file_name', 'true_label'])
    return df

def count_images_per_label(df):
    label_counts = Counter(df['true_label'])
    return label_counts

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

def main():
    labels_file = input("Enter the path to the text file containing the labels: ")
    image_folder = input("Enter the path to the folder containing the images: ")
    max_images_per_label = int(input("Enter the maximum number of images per label: "))

    df = read_labels(labels_file)
    df = delete_excess_images(df, max_images_per_label, image_folder)

    # Save the updated labels file
    df.to_csv(labels_file, sep='\t', index=False, header=False)

    print("Excess images and corresponding labels deleted successfully.")

if __name__ == "__main__":
    main()
