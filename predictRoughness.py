import cv2
import numpy as np
from keras.models import load_model
from preprocessImage import preprocessImage

# Set the image size and channels
img_size = 800
# Set the image prediction label
label_map = {0: "N5", 1: "N6", 2: "N7", 3: "N8"}

# Load the test image
img = cv2.imread("samples/Sample9B.jpg", cv2.IMREAD_GRAYSCALE)

# Image augmentation & preprocessing
img = preprocessImage(img, img_size)

# Preprocessing progress
print("... Images was preprocessed successfully")

# Save the preprocessed image
cv2.imwrite("preprocessed_img.jpg", img)
print("... Preprocessed image was saved")

# Add an extra dimension to match the shape of the training data
img = np.expand_dims(img, axis=0)
# Expand the dimensions of the img array to include the channels dimension
img = np.expand_dims(img, axis=-1)

# Load the saved model
model = load_model("roughness_classifier3.h5")
print("... The surface roughness prediction model was loaded successfully")

# Use the model to predict the roughness of the test image
pred = model.predict(img)
# Map the predicted integer label back to its string value
roughness_pred = label_map[np.argmax(pred)]

print("... The surface roughness of the test sample is:", roughness_pred)
