import cv2


def cropImage(img, img_size, right=False, bottom=False):
    if right:
        img = img[:img_size, -img_size:]  # Crop the img to a top-right corner
    elif bottom:
        img = img[-img_size:, :img_size]  # Crop the img to a bottom-left corner
    else:
        img = img[:img_size, :img_size]  # Crop the img to a top-left corner

    img = cv2.resize(img, (img_size, img_size))  # Resize img to a square of fixed size

    return img
