import cv2
import numpy as np


def preprocessImage(img, img_size):
    # Image augmentation
    img = img[:img_size, :img_size]  # Crop the image to a square of fixed size
    img = cv2.resize(img, (img_size, img_size))  # Resize the image to a fixed size

    # # Image preprocessing
    # img = cv2.fastNlMeansDenoising(img, None, 20, 20, 15)  # Apply denoising
    # img = img.astype("float32") / 255.0  # Apply normalization
    # img = np.array(img, dtype="float") / 255.0  # Apply normalization
    # img = (img * 255).astype(np.uint8)  # Convert the image to CV_8U data type
    # img = cv2.Canny(img, 50, 150)  # Apply edge detection
    # edges = cv2.Canny(img, 25, 250)  # Apply edge detection
    # img = cv2.addWeighted(img, 0.8, edges, 0.2, 0)  # Stack edges on image

    return img
