import os
import cv2
import numpy as np


def saveOutputImages(samples, labels):
    # Create a directory to save the images
    output_folder = "output_images"
    os.makedirs(output_folder, exist_ok=True)

    label_map = {0: "N5", 1: "N6", 2: "N7"}
    # labels = np.argmax(labels)
    labels = [label_map.get(label, label) for label in labels]

    # Iterate over the samples and save the images with different names
    for i, sample in enumerate(samples):
        # Get the label for the current sample
        label = labels[i]
        # Generate a unique name for the image
        img_name = f"img_{i}_{label}.jpg"
        # Construct the full path to save the image
        img_path = os.path.join(output_folder, img_name)
        # Write the image to the specified path
        cv2.imwrite(img_path, sample)

    print("Images saved successfully.")
