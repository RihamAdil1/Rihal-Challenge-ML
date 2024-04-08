import os
import shutil

# Define the directory to save the filtered images for validation on the D: partition
filtered_image_directory_validation = 'D:/filtered_images_test'

# Create the directory if it doesn't exist
os.makedirs(filtered_image_directory_validation, exist_ok=True)

# Read the image IDs from the text file
with open('image_ids_test.txt', 'r') as file:
    image_ids_validation = file.read().splitlines()

# Filter and copy images based on image IDs
for image_id in image_ids_validation:
    image_file = f'{image_id}.jpg'  # Assuming the image format is JPG
    original_image_path = os.path.join(os.getcwd(), image_file)  # Get the current working directory
    filtered_image_path = os.path.join(filtered_image_directory_validation, image_file)
    
    # Check if the image exists in the current directory
    if os.path.exists(original_image_path):
        # Copy the image to the filtered directory for validation
        shutil.copy(original_image_path, filtered_image_path)
