import json
import os

import numpy as np
from PIL import Image
from tqdm import tqdm


# Function to apply padding to the bounding box
def apply_padding(bbox, image_width, image_height, padding_percentage=0.05):
    x_min, y_min, width, height = bbox
    # Calculate the padding
    padding_x = int(width * padding_percentage)
    padding_y = int(height * padding_percentage)

    # Apply padding while ensuring it stays within image bounds
    x_min = max(0, x_min - padding_x)
    y_min = max(0, y_min - padding_y)
    x_max = min(image_width, x_min + width + padding_x)
    y_max = min(image_height, y_min + height + padding_y)

    # Ensure coordinates are valid (x_min < x_max and y_min < y_max)
    if x_min >= x_max or y_min >= y_max:
        return None  # Return None for invalid coordinates

    return [x_min, y_min, x_max, y_max]


# Function to get the next available image name in the class folder
def get_next_image_name(class_name, image_counter):
    # Get the next available image ID for the class
    next_image_id = image_counter.get(class_name, 0)
    image_counter[class_name] = next_image_id + 1
    return next_image_id


# Function to process each image and save cropped bounding boxes
def process_images(image_folder, annotation_file, output_folder, image_counter):
    # Load the annotations
    with open(annotation_file, "r") as f:
        annotations = json.load(f)

    # Get the image information
    images = annotations["images"]
    annotations_data = annotations["annotations"]
    categories = {
        category["id"]: category["name"] for category in annotations["categories"]
    }

    # Process each image
    for image_info in tqdm(images):
        image_id = image_info["id"]
        image_filename = image_info["file_name"]
        image_path = os.path.join(image_folder, image_filename)

        # Open the image using Pillow
        try:
            image = Image.open(image_path)
        except IOError:
            continue  # Skip images that can't be read

        image_width, image_height = image.size

        # Find annotations related to this image
        relevant_annotations = [
            ann for ann in annotations_data if ann["image_id"] == image_id
        ]

        for ann in relevant_annotations:
            bbox = ann["bbox"]  # COCO format: [x, y, width, height]
            category_id = ann["category_id"]
            class_name = categories[category_id]

            # Apply padding to the bounding box
            padded_bbox = apply_padding(bbox, image_width, image_height)

            # If the bounding box is invalid, print the image_id and annotation_id
            if padded_bbox is None:
                print(
                    f"Invalid crop coordinates for image_id {image_id}, annotation_id {ann['id']}"
                )
                continue

            # Crop the image based on the bounding box
            x_min, y_min, x_max, y_max = padded_bbox
            cropped_image = image.crop((x_min, y_min, x_max, y_max))

            # Create the folder for the class if it doesn't exist
            class_folder = os.path.join(
                output_folder, os.path.basename(image_folder), class_name
            )
            os.makedirs(class_folder, exist_ok=True)

            # Get the next available image name (sequential)
            next_image_id = get_next_image_name(class_name, image_counter)

            # Save the cropped image with the next image ID
            cropped_image_filename = f"{class_name}_{next_image_id}.jpg"
            cropped_image_path = os.path.join(class_folder, cropped_image_filename)
            cropped_image.save(cropped_image_path)


# Function to get root and output directories from the user
def get_user_input():
    # Get the root directory for datasets
    root_dir = input(
        "Please enter the root directory where the 'train', 'valid', and 'test' folders are located: "
    )
    if not os.path.isdir(root_dir):
        print("The provided root directory does not exist. Exiting...")
        exit(1)

    # Get the output directory for cropped images
    output_dir = input("Please enter the output directory to save cropped images: ")
    if not os.path.isdir(output_dir):
        print("The provided output directory does not exist. Creating...")
        os.makedirs(output_dir)

    return root_dir, output_dir


# Main execution
def main():
    # Dictionary to maintain counters for each class
    image_counter = {}

    # Get root and output directories
    root_dir, output_folder = get_user_input()

    # Define the paths for the annotation files in each dataset
    train_annotation_file = os.path.join(root_dir, "train", "_annotations.coco.json")
    valid_annotation_file = os.path.join(root_dir, "valid", "_annotations.coco.json")
    test_annotation_file = os.path.join(root_dir, "test", "_annotations.coco.json")

    # Check if annotation files exist
    if not os.path.isfile(train_annotation_file):
        print(f"Train annotation file not found: {train_annotation_file}")
        exit(1)
    if not os.path.isfile(valid_annotation_file):
        print(f"Validation annotation file not found: {valid_annotation_file}")
        exit(1)
    if not os.path.isfile(test_annotation_file):
        print(f"Test annotation file not found: {test_annotation_file}")
        exit(1)

    # Process the datasets (train, valid, test)
    process_images(
        os.path.join(root_dir, "train"),
        train_annotation_file,
        output_folder,
        image_counter,
    )
    total = 0
    print(image_counter)
    for key, val in image_counter.items():
        total += val

    print(total)
    image_counter = {}  # Reset the image counter for validation and test sets
    process_images(
        os.path.join(root_dir, "valid"),
        valid_annotation_file,
        output_folder,
        image_counter,
    )
    image_counter = {}  # Reset the image counter for validation and test sets
    process_images(
        os.path.join(root_dir, "test"),
        test_annotation_file,
        output_folder,
        image_counter,
    )

    print("Processing completed!")


if __name__ == "__main__":
    main()


# /home/robosteps/repos/SimCLR/datasets/ff_devices
