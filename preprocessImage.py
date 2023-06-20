import cv2
import numpy as np


def preprocessImage(img):
    # img = cv2.fastNlMeansDenoising(img)  # Apply denoising
    # img = img.astype("float32") / 255.0  # Apply normalization
    # img = np.array(img, dtype="float") / 255.0  # Apply normalization
    # img = (img * 255).astype(np.uint8)  # Convert the image to CV_8U data type
    sharpen_filter = np.array(
        [[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]
    )  # Sharpening kernel
    img = cv2.filter2D(img, -1, sharpen_filter)  # Apply sharpening
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.Canny(img, 100, 250)  # Apply edge detection
    # edges = cv2.Canny(img, 35, 175)  # Apply edge detection
    # img = cv2.addWeighted(img, 0.8, edges, 0.2, 0)  # Stack edges on image

    return img
