import cv2


def cropImage(img, img_size):
    half_height = img_size // 2

    cropped_size = img_size // 3
    cropped_height = cropped_size // 2

    y_start = half_height - cropped_height
    y_end = half_height + cropped_height

    imgL = img[y_start:y_end, 0:cropped_size]
    imgC = img[y_start:y_end, cropped_size : 2 * cropped_size]
    imgR = img[y_start:y_end, 2 * cropped_size : 3 * cropped_size]

    imgL = cv2.resize(imgL, (cropped_size, cropped_size))
    imgC = cv2.resize(imgC, (cropped_size, cropped_size))
    imgR = cv2.resize(imgR, (cropped_size, cropped_size))

    return imgL, imgC, imgR
