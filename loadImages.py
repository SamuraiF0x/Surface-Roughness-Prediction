import os
import json
import numpy as np
import cv2
from preprocessImage import preprocessImage
from augmentImage import augmentImage
from keras.utils import to_categorical


def loadImages():
    # Set the image size
    img_size = 1024

    # Load the dataset
    with open("dataset.json", "r") as f:
        data = json.load(f)

    # Define arrays
    samples = []
    labels = []

    # Define the mapping from string labels to integer values
    label_map = {"N5": 0, "N6": 1, "N7": 2}

    # Convert labels to their corresponding label_map values
    labels = [label_map.get(label, label) for label in labels]

    sampleCount = 0

    for d in data:
        sample_id = d["sampleId"]
        roughness = d["roughness"]
        img_path = os.path.join("samples", sample_id + ".jpg")
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load sample image

        # Samples size info
        sampleCount += 1  # Increment the count of samples
        sampleSize = len(data)  # Number of samples

        # Image preprocessing
        img = preprocessImage(img)

        # Image preprocessing progress
        print(
            f"[INFO] ... [{sampleCount}/{sampleSize}] images processed", end="\r")

        # Image augmentation
        samples, labels = augmentImage(
            img, img_size, samples, labels, label_map, roughness
        )

    print(f"[INFO] Number of loaded images: {sampleSize}  ")

    # Convert the samples and labels to numpy arrays
    samples = np.array(samples)
    labels = np.array(labels)

    # # Expand the dimensions of the samples array to include the channels dimension
    samples = np.expand_dims(samples, axis=-1)
    # # Convert the labels to one-hot encoding
    # labels = to_categorical(labels, len(label_map))

    return img_size, samples, labels, label_map
