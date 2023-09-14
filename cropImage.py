import cv2


def cropImage(img, img_size):

    cropped_size = img_size // 3

    first = cropped_size
    mid = 2 * cropped_size
    last = 3 * cropped_size

    imgL = img[first:mid, 0:first]
    imgC = img[first:mid, first:mid]
    imgR = img[first:mid, mid:last]

    imgL = cv2.resize(imgL, (cropped_size, cropped_size))
    imgC = cv2.resize(imgC, (cropped_size, cropped_size))
    imgR = cv2.resize(imgR, (cropped_size, cropped_size))

    return (imgL, imgC, imgR,)
