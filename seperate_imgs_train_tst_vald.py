"""
This code extracts and filters images only for training from the whole other images, I extracted them by taking only the images_ids from 
the training set and sve those images in the folder
"""

import os
import shutil

# Define the directory to save the filtered images on the D: partition for training
filtered_image_directory = 'D:/filtered_images'

# Create the directory if it doesn't exist
os.makedirs(filtered_image_directory, exist_ok=True)

# Read the image IDs from the text file
with open('image_ids.txt', 'r') as file:
    image_ids = file.read().splitlines()

# Filter and copy images based on image IDs
for image_id in image_ids:
    image_file = f'{image_id}.jpg'  # Assuming the image format is JPG
    original_image_path = os.path.join(os.getcwd(), image_file)  # Get the current working directory
    filtered_image_path = os.path.join(filtered_image_directory, image_file)
    
    # Check if the image exists in the current directory
    if os.path.exists(original_image_path):
        # Copy the image to the filtered directory
        shutil.copy(original_image_path, filtered_image_path)



