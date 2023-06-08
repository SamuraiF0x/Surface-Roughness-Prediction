import cv2
import numpy as np
from keras.models import load_model
from preprocessImage import preprocessImage
from cropImage import cropImage


print("[INFO] Initialising surface roughness prediction model...")
# Set the image size and channels
img_size = 500
# Set the image prediction label
label_map = {0: "N5", 1: "N6", 2: "N7"}

# Load the test image
print("[INFO] Loading test image...")
img = cv2.imread("samples/Sample7A.jpg", cv2.IMREAD_GRAYSCALE)
print("[INFO] Image was loaded successfully...")

# Image augmentation & preprocessing
print("[INFO] Processing image...")
img = cropImage(img, img_size)
img = preprocessImage(img)
print("[INFO] Image was processed successfully...")

# Add an extra dimension to match the shape of the training data
img = np.expand_dims(img, axis=0)
# Expand the dimensions of the img array to include the channels dimension
img = np.expand_dims(img, axis=-1)

# Load the saved model
print("[INFO] Loading surface roughness prediction model...")
model = load_model("roughness_classifier/roughness_classifier3.h5")
print("[INFO] The surface roughness prediction model was loaded successfully...")

print("[INFO] Predicting surface roughness...")
# Use the model to predict the roughness of the test image
pred = model.predict(img)
# Map the predicted integer label back to its string value
roughness_pred = label_map[np.argmax(pred)]

print("[INFO] The surface roughness of the test sample is:", roughness_pred)
