import numpy as np
from loadImages import loadImages
from generateImages import generateImages
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

img_size, samples, labels, label_map = loadImages()
augmented_samples, augmented_labels = generateImages(samples, labels)

x_train, x_val, y_train, y_val = train_test_split(
    augmented_samples, augmented_labels, test_size=0.2, stratify=augmented_labels
)

print("[INFO] Number of samples used for training ...", len(x_train))
print("[INFO] Number of samples used for validation ...", len(x_val))

# Define the CNN architecture
model = Sequential()
model.add(
    Conv2D(
        32,
        kernel_size=(3, 3),
        activation="relu",
        input_shape=(img_size, img_size, 1),
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

epochs = 1
target_accuracy = 0.85

# Define early stopping callback
early_stop = EarlyStopping(monitor="val_accuracy", patience=3, verbose=1, mode="max")

# Train the model with data augmentation and early stopping
while True:
    history = model.fit(
        augmented_samples,
        augmented_labels,
        steps_per_epoch=len(x_train) // 32,
        epochs=epochs,
        validation_data=(x_val, y_val),
        callbacks=[early_stop],
    )

    epochs += 1

    print(f"[INFO] Current validation accuracy:", history.history["val_accuracy"][-1])

    if history.history["val_accuracy"][-1] >= target_accuracy:
        break

print("[INFO] ... Saving surface roughness prediction model ...")
# Save the model so it can be imported to
model.save("roughness_classifier/roughness_classifier2.h5")

print("[INFO] The surface roughness prediction model was built SUCCESSFULLY")
print("[INFO] Run predictRoughness to test the model")
