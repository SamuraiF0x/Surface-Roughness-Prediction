import json
import os
import cv2
import numpy as np
from keras.utils import to_categorical
from preprocessImage import preprocessImage
from augmentImage import augmentImage


def loadImages():
    # Set the image size
    img_size = 800

    # Load the dataset
    with open("dataset.json", "r") as f:
        data = json.load(f)

    # Define the mapping from string labels to integer values
    label_map = {
        label: i for i, label in enumerate(set([d["roughness"] for d in data]))
    }

    # Sort labels
    sorted_labels = sorted(label_map.items(), key=lambda x: x[0])
    label_map = {label: i for i, (label, value) in enumerate(sorted_labels)}

    # Define arrays
    samples = []
    labels = []

    sampleCount = 0
    sampleSize = 0

    for d in data:
        sample_id = d["sampleId"]
        roughness = d["roughness"]
        img_path = os.path.join("samples", sample_id + ".jpg")
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load sample image

        # Samples size info
        sampleCount += 1  # Increment the count of samples
        sampleSize = len(data)  # Number of samples

        # # Image preprocessing
        img = preprocessImage(img)

        # Image preprocessing progress
        print(f"[INFO] ... [{sampleCount}/{sampleSize}] images processed", end="\r")

        # Image augmentation
        samples, labels = augmentImage(
            img, img_size, samples, labels, label_map, roughness
        )

    print(f"[INFO] Number of loaded images: {sampleSize}  ")

    # Convert the samples and labels to numpy arrays
    samples = np.array(samples)
    labels = np.array(labels)

    # Expand the dimensions of the samples array to include the channels dimension
    samples = np.expand_dims(samples, axis=-1)
    # Convert the labels to one-hot encoding
    labels = to_categorical(labels, len(label_map))

    print("[INFO] Number of augmented images:", len(samples))

    return img_size, samples, labels, label_map
