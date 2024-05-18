import os

def get_image_type(filename):
    # Extract the image type which is the second part of the filename
    return filename.split('_')[1]

def get_image_number(filename):
    # Extract the image number which is the second to last part of the filename
    return int(filename.split('_')[-2])

def get_true_label(filename):
    # Extract the true label which is the part after the last underscore and before the file extension
    return filename.split('_')[-1].split('.')[0]

def prompt_user_for_label(filename):
    true_label = get_true_label(filename)
    print(f"\nFilename: {filename}")
    print(f"Extracted label: {true_label}")
    
    while True:
        response = input("Is this label correct? (y/n): ").strip().lower()
        if response in ['y', 'n']:
            break
        print("Please enter 'y' or 'n'.")
    
    if response == 'n':
        new_label = input("Enter the correct label: ").strip()
        return new_label
    else:
        return true_label

def rename_file_with_new_label(directory, filename, new_label):
    base, ext = os.path.splitext(filename)
    parts = base.split('_')
    parts[-1] = new_label
    new_filename = '_'.join(parts) + ext
    old_filepath = os.path.join(directory, filename)
    new_filepath = os.path.join(directory, new_filename)
    os.rename(old_filepath, new_filepath)
    print(f"Renamed {filename} to {new_filename}")

def process_images(directory, image_type, start_image_number):
    processed_any_file = False
    for filename in sorted(os.listdir(directory)):
        if filename.endswith(('.jpg', '.png', '.jpeg')):  # Add other image formats if needed
            try:
                if get_image_type(filename) == image_type:
                    image_number = get_image_number(filename)
                    if image_number >= start_image_number:
                        processed_any_file = True
                        new_label = prompt_user_for_label(filename)
                        old_label = get_true_label(filename)
                        if new_label != old_label:
                            rename_file_with_new_label(directory, filename, new_label)
                        else:
                            print(f"No changes needed for {filename}")
            except ValueError:
                print(f"Skipping file with invalid format: {filename}")
    if not processed_any_file:
        print(f"No {image_type} images with a number >= {start_image_number} found in the directory.")

if __name__ == "__main__":
    print("This script will help you verify and update image labels.")
    print("Please make sure your filenames are in the format: elidExit_warped_YYYYMMDD-HHMMSS_XXXXX_XAXXXX.jpg or elidExit_original_YYYYMMDD-HHMMSS_XXXXX_XAXXXX.jpg")
    print("Where the text after the last underscore is the true label.")
    
    image_directory = input("Enter the path to your image directory: ").strip()
    if not os.path.isdir(image_directory):
        print("The provided path is not a valid directory. Please run the script again with a valid directory.")
    else:
        while True:
            image_type = input("Do you want to start with 'warped' or 'original' images? ").strip().lower()
            if image_type in ['warped', 'original']:
                break
            print("Please enter 'warped' or 'original'.")
        
        while True:
            try:
                start_image_number = int(input("Enter the starting image number: ").strip())
                break
            except ValueError:
                print("Please enter a valid integer for the starting image number.")
        
        process_images(image_directory, image_type, start_image_number)
