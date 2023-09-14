import os
import cv2
import numpy as np
from keras.models import load_model
from preprocessImage import preprocessImage
from cropImage import cropImage
import matplotlib.pyplot as plt
import pandas as pd

# Set the image size and channels
img_size = 1024
# Set the image prediction label
label_map = {0: "N5", 1: "N6", 2: "N7"}

# Load all sample images in the "samples" folder
sample_folder = "samples"
sample_images = [file for file in os.listdir(
    sample_folder) if file.endswith(".jpg")]

# Create a DataFrame to store prediction results
results_df = pd.DataFrame(
    columns=["Image", "True Label", "Predicted Label", "Classifier"]
)

# Loop through each sample image
for sample_image in sample_images:
    print("[INFO] Processing surface roughness for", sample_image)

    # Load the test image
    img_path = os.path.join(sample_folder, sample_image)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    print("[INFO] Image", sample_image, "was loaded successfully...")

    # Image augmentation & preprocessing
    print("[INFO] Processing image...")
    img = preprocessImage(img)
    # img = cropImage(img, img_size)
    cropped_size = img_size // 3
    first = cropped_size
    mid = 2 * cropped_size
    img = img[first:mid, first:mid]
    print("[INFO] Image was processed successfully...")

    # Add an extra dimension to match the shape of the training data
    img = np.expand_dims(img, axis=0)
    # Expand the dimensions of the img array to include the channels dimension
    img = np.expand_dims(img, axis=-1)

    # Loop through all roughness classifiers in the "roughness_classifier" folder
    model_folder = "roughness_classifier"
    roughness_models = [
        file for file in os.listdir(model_folder) if file.endswith(".h5")
    ]

    for model_name in roughness_models:
        model_path = os.path.join(model_folder, model_name)
        print(
            "[INFO] Loading surface roughness prediction model",
            model_name,
            "for",
            sample_image,
        )
        model = load_model(model_path)
        print(
            "[INFO] The surface roughness prediction model was loaded successfully..."
        )

        print("[INFO] Predicting surface roughness for", sample_image)
        # Use the model to predict the roughness of the test image
        pred = model.predict(img)
        # Map the predicted integer label back to its string value
        roughness_pred = label_map[np.argmax(pred)]

        # Get the true label from the filename
        true_label = sample_image.split("_")[0]

        # Store the results in the DataFrame
        results_df = results_df.append(
            {
                "Image": sample_image,
                "True Label": true_label,
                "Predicted Label": roughness_pred,
                "Classifier": model_name,
            },
            ignore_index=True,
        )

# Create a bar chart to visualize correct and incorrect predictions for each classifier
correct_predictions = results_df["True Label"] == results_df["Predicted Label"]
classifier_counts = results_df.groupby("Classifier")["Image"].count()
correct_counts = results_df[correct_predictions].groupby("Classifier")[
    "Image"].count()


# Print the overall accuracy for each classifier
for classifier, total_count in classifier_counts.items():
    correct_count = correct_counts.get(classifier, 0)
    accuracy = (correct_count / total_count) * 100
    print(f"Classifier {classifier}: Accuracy {accuracy:.2f}%")

# Plot the results
plt.figure(figsize=(10, 6))
plt.bar(
    classifier_counts.index,
    classifier_counts.values,
    label="Total Predictions",
    alpha=0.7,
)
plt.bar(
    correct_counts.index, correct_counts.values, label="Correct Predictions", alpha=0.7, color='green'
)


plt.xlabel("Classifier")
plt.ylabel("Number of Predictions")
plt.title("Correct vs. Total Predictions for Each Classifier")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ! NOT WORKING CORRECTLY
