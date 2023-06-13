import numpy as np
from loadImages import loadImages
from generateImages import generateImages
from saveOutputImages import saveOutputImages
from accuracyPlot import accuracyPlot
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential

# from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.layers import (
    Dense,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dropout,
    BatchNormalization,
)
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

img_size, samples, labels, label_map = loadImages()
img_size = img_size // 3

saveOutputImages(samples, labels)

# Generate augmented images
augmented_samples, augmented_labels = generateImages(samples, labels)

# saveOutputImages(augmented_samples, augmented_labels)


# Split the data
print("[INFO] Splitting data into train and validation sets...")
x_train, x_val, y_train, y_val = train_test_split(
    augmented_samples,
    augmented_labels,
    test_size=0.2,
    stratify=augmented_labels,
    shuffle=True,
)

print("[INFO] Number of samples used for training ...", len(x_train))
print("[INFO] Number of samples used for validation ...", len(x_val))

x_train = x_train.astype("float32") / 255.0
x_val = x_val.astype("float32") / 255.0
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)


# Define the CNN architecture
model = Sequential()

### 1
model.add(
    Conv2D(
        16,
        kernel_size=(3, 3),
        activation="relu",
        input_shape=(img_size, img_size, 1),
    )
)
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(32, kernel_size=(3, 3), activation="sigmoid"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
# model.add(Dense(128, activation="relu"))
# model.add(Dense(len(label_map), name="outputs"))
# model.add(Dense(len(label_map), activation="sigmoid"))
model.add(Dense(len(label_map), activation="softmax"))

### 2
# model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(img_size, img_size, 1)))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

# model.add(Conv2D(64, (3, 3), activation="relu"))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

# model.add(Conv2D(128, (3, 3), activation="relu"))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

# model.add(Flatten())
# model.add(Dense(256, activation="relu"))
# model.add(Dense(len(label_map), activation="sigmoid"))
# # model.add(Dense(len(label_map), activation="softmax"))

### 3
# model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(img_size, img_size, 1)))
# model.add(MaxPooling2D((2, 2)))
# model.add(Flatten())
# model.add(Dense(100, activation="relu"))
# model.add(Dense(3, activation="softmax"))


# Compile the model
model.compile(
    loss="categorical_crossentropy",
    optimizer=Adam(learning_rate=0.001),
    # optimizer="adam",
    metrics=["accuracy"],
)

epochs = 100
# target_accuracy = 0.65

# Define early stopping callback
# early_stop = EarlyStopping(monitor="val_accuracy", patience=3, verbose=1, mode="max")
early_stop = EarlyStopping(
    monitor="val_loss", patience=3, verbose=1, mode="min", restore_best_weights=True
)

# Train the model with data augmentation and early stopping
print("[INFO] Initialising the model training...")
# while True:
history = model.fit(
    x_train,
    y_train,
    # steps_per_epoch=len(x_train) // 64,
    epochs=epochs,
    batch_size=32,
    validation_data=(x_val, y_val),
    callbacks=[early_stop],
)

#     epochs += 1

#     print("[INFO] Current validation accuracy:", history.history["val_accuracy"][-1])

#     if history.history["val_accuracy"][-1] >= target_accuracy:
#         break


print(model.summary())

print("[INFO] ... Saving surface roughness prediction model ...")
# Save the model so it can be imported to
model.save("roughness_classifier/roughness_classifier5.h5")

print("[INFO] The surface roughness prediction model was built SUCCESSFULLY")

print("[INFO] Plotting training statistics...")
# Plot the training and validation accuracy
accuracyPlot(history, metric="loss")
accuracyPlot(history, metric="accuracy")
print("[INFO] Plot completed...")

print("[INFO] Run predictRoughness.py to test the model")
