import cv2
import numpy as np


def preprocessImage(img):
    sharpen_filter = np.array(
        [[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]
    )  # Sharpening kernel
    img = cv2.filter2D(img, -1, sharpen_filter)  # Apply sharpening
    img = cv2.GaussianBlur(img, (5, 5), 0)  # Apply gaussian blur
    img = cv2.Canny(img, 100, 250)  # Apply canny edge detection

    return img
