import os
import json
import numpy as np
import cv2
from preprocessImage import preprocessImage
from keras.utils import to_categorical


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

    # Load the images and labels
    samples = []
    labels = []

    sampleCount = 0

    for d in data:
        sample_id = d["sampleId"]
        roughness = d["roughness"]
        img_path = os.path.join("samples", sample_id + ".jpg")
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load sample image

        # Image augmentation & preprocessing
        img = preprocessImage(img, img_size)

        samples.append(img)
        # Replace the string label with its corresponding integer value
        labels.append(label_map[roughness])

        # Samples size info
        sampleCount += 1  # Increment the count of samples
        sampleSize = len(data)  # Number of samples

        # Preprocessing progress
        print(f"[INFO] ... [{sampleCount}/{sampleSize}] images processed", end="\r")

    print(
        f"[INFO] ... [{sampleCount}/{sampleSize}] images were preprocessed successfully"
    )

    # Convert the samples and labels to numpy arrays
    samples = np.array(samples)
    labels = np.array(labels)

    # Expand the dimensions of the samples array to include the channels dimension
    samples = np.expand_dims(samples, axis=-1)
    # Convert the labels to one-hot encoding
    labels = to_categorical(labels, len(label_map))

    return img_size, samples, labels, label_map
