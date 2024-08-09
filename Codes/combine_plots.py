import os
from PIL import Image
import matplotlib.pyplot as plt

def resize_images_to_same_height(images):
    max_height = max(img.height for img in images)
    resized_images = [img.resize((img.width, max_height), Image.Resampling.LANCZOS) for img in images]
    return resized_images

def combine_images_with_labels(image_paths, labels, output_path):
    images = [Image.open(image_path) for image_path in image_paths]
    images = resize_images_to_same_height(images)  # Resize images to the same height

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))  # Adjust figsize to control the overall size

    # Plot the original and warped images in the first row
    for i, (img, label) in enumerate(zip(images[:2], labels[:2])):
        axs[0, i].imshow(img)
        axs[0, i].set_title(label, fontsize=12)  # Adjust font size
        axs[0, i].axis('off')

    # Plot the combined image in the second row, center it
    axs[1, 0].axis('off')
    axs[1, 1].imshow(images[2])
    axs[1, 1].set_title(labels[2], fontsize=12)  # Adjust font size
    axs[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

# Prompt user to input the dataset type and run number
dataset_type = input("Please enter the dataset type (e.g., 'all_random', 'dataset_all', etc.): ")
run_number = input("Please enter the run number (e.g., 'run1', 'run2', etc.): ")

# Construct the base folder paths for each dataset type and run number based on the user input
base_path = r'C:\Users\Oswin\Desktop\FYP\ANPR-1\Datasets'
base_folders = {
    'original': os.path.join(base_path, f'{dataset_type}', 'original', 'plots', 'trocr-small-printed', run_number),
    'warped': os.path.join(base_path, f'{dataset_type}', 'warped', 'plots', 'trocr-small-printed', run_number),
    'combined': os.path.join(base_path, f'{dataset_type}', 'combined', 'plots', 'trocr-small-printed', run_number)
}

# Image filenames
image_filenames = ['accuracy.png', 'cer.png', 'train_val_loss.png']

# Labels for the combined images
labels = ["Original", "Warped", "Combined"]

# Output directory
output_dir = os.path.join(base_path, f'{dataset_type}')

# Combine images of the same metric across different dataset types
for image_filename in image_filenames:
    combined_image_paths = [os.path.join(base_folders[dataset_type], image_filename) for dataset_type in base_folders]
    combined_output_path = os.path.join(output_dir, f'combined_{image_filename}')
    combine_images_with_labels(combined_image_paths, labels, combined_output_path)
    print(f"Combined image saved as: {combined_output_path}")
