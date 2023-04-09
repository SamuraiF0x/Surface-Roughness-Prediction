import numpy as np
from keras.preprocessing.image import ImageDataGenerator


def generateImages(samples, labels):
    # Define the data augmentation generator
    datagen = ImageDataGenerator(
        rotation_range=90, horizontal_flip=True, vertical_flip=True, rescale=1 / 255
    )

    # Generate augmented images and append them to samples with the same labels
    augmented_samples = []
    augmented_labels = []

    for i in range(len(samples)):
        # Get the original image and its label
        sample = samples[i]
        label = labels[i]

        # Add the original image and its label to the augmented samples and labels arrays
        augmented_samples.append(sample)
        augmented_labels.append(label)

        # Generate augmented images with the same label as the original image
        aug_iter = datagen.flow(sample.reshape((1,) + sample.shape), batch_size=32)

        # generate 7 unique images from each original image
        for j in range(7):
            aug_sample = next(aug_iter)[0]
            augmented_samples.append(aug_sample)
            augmented_labels.append(label)

    # Convert the samples and labels to numpy arrays
    augmented_samples = np.array(augmented_samples)
    augmented_labels = np.array(augmented_labels)

    print("[INFO] Number of generated images:", len(augmented_samples))

    return augmented_samples, augmented_labels
