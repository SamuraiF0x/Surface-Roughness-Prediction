import os
import json
import numpy as np
import cv2
from preprocessImage import preprocessImage
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.optimizers import Adam

# Set the image size and channels
img_size = 800
img_channels = 1

# Load the dataset
with open("dataset.json", "r") as f:
    data = json.load(f)

# Define the mapping from string labels to integer values
label_map = {label: i for i, label in enumerate(set([d["roughness"] for d in data]))}

# Load the images and labels
samples = []
labels = []

sampleCount = 0

for d in data:
    sample_id = d["sampleId"]
    roughness = d["roughness"]
    img_path = os.path.join("samples", sample_id + ".jpg")
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load sample image

    # Samples size info
    sampleCount += 1  # Increment the count of samples
    sampleSize = len(data)  # Number of samples

    # Image augmentation & preprocessing
    img = preprocessImage(img, img_size)

    samples.append(img)
    # Replace the string label with its corresponding integer value
    labels.append(label_map[roughness])

    # Preprocessing progress
    print(f"... [{sampleCount}/{sampleSize}] images processed", end="\r")

print(f"... [{sampleCount}/{sampleSize}] images were preprocessed successfully")

# Convert the samples and labels to numpy arrays
samples = np.array(samples)
labels = np.array(labels)

# Expand the dimensions of the samples array to include the channels dimension
samples = np.expand_dims(samples, axis=-1)
# Convert the labels to one-hot encoding
labels = to_categorical(labels, len(label_map))

# Assume samples and labels are your preprocessed data
X_train, X_val, y_train, y_val = train_test_split(
    samples, labels, test_size=0.2, stratify=labels
)

# Define the data augmentation generator
datagen = ImageDataGenerator(
    rotation_range=90, horizontal_flip=True, vertical_flip=True
)

# Generate augmented images
augmentedData = datagen.flow(X_train, y_train, batch_size=32)

print("Number of generated images:", augmentedData.n * augmentedData.batch_size)

# Define the CNN architecture
model = Sequential()
model.add(
    Conv2D(
        32,
        kernel_size=(3, 3),
        activation="relu",
        input_shape=(img_size, img_size, img_channels),
    )
)
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dense(len(label_map), activation="softmax"))

# Compile the model
model.compile(
    loss="categorical_crossentropy",
    optimizer=Adam(learning_rate=0.001),
    metrics=["accuracy"],
)
# model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the model with data augmentation
history = model.fit(
    augmentedData,
    steps_per_epoch=len(X_train) / 32,
    epochs=1,
    validation_data=(X_val, y_val),
)

print("... len(X_train) ...", len(X_train))
print("... Saving surface roughness prediction model ...")

# Save the model so it can be imported to
model.save("roughness_classifier3.h5")

print("... The surface roughness prediction model was built successfully")
