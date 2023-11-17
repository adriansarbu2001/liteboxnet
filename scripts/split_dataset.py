import os
import shutil
import random
random.seed(1234)


def split_data(source_folder, destination_folder, split_ratio):
    # Get the list of files in the images subfolder of the source folder
    image_folder = os.path.join(source_folder, 'image_2')
    label_folder = os.path.join(source_folder, 'label_2')

    image_files = os.listdir(image_folder)

    # Calculate the number of files to move for validation
    num_files_to_move = int(len(image_files) * split_ratio)

    # Randomly select files to move
    files_to_move = random.sample(image_files, num_files_to_move)

    # Create subfolders in the destination folder
    os.makedirs(os.path.join(destination_folder, 'image_2'), exist_ok=True)
    os.makedirs(os.path.join(destination_folder, 'label_2'), exist_ok=True)

    # Move selected files to the validation folder
    for file in files_to_move:
        # Move image file
        image_source_path = os.path.join(image_folder, file)
        image_dest_path = os.path.join(destination_folder, 'image_2', file)
        shutil.move(image_source_path, image_dest_path)

        # Move label file
        label_file = file.replace('.png', '.txt')
        label_source_path = os.path.join(label_folder, label_file)
        label_dest_path = os.path.join(destination_folder, 'label_2', label_file)
        shutil.move(label_source_path, label_dest_path)


if __name__ == "__main__":
    # Set your source and destination folders
    source_folder = "D:/UBB/Inteligenta Computationala Aplicata EN/SEM I/datasets/liteboxnet/training"
    destination_folder = "D:/UBB/Inteligenta Computationala Aplicata EN/SEM I/datasets/liteboxnet/validating"

    # Set the split ratio (e.g., 0.2 for 20% validation)
    split_ratio = 0.2

    # Call the function to split the data
    split_data(source_folder, destination_folder, split_ratio)

    print("Data split successfully!")
