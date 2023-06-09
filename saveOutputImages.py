import os
import cv2


def saveOutputImages(samples):
    # Create a directory to save the images
    output_folder = "output_images"
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over the samples and save the images with different names
    for i, sample in enumerate(samples):
        # Generate a unique name for the image
        img_name = f"image_{i}.jpg"
        # Construct the full path to save the image
        img_path = os.path.join(output_folder, img_name)
        # Write the image to the specified path
        cv2.imwrite(img_path, sample)

    print("Images saved successfully.")
