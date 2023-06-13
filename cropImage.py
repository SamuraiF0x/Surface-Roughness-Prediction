import cv2


def cropImage(img, img_size):
    half_height = img_size // 2

    cropped_size = img_size // 3

    first = cropped_size
    mid = 2 * cropped_size
    last = 3 * cropped_size

    imgFull = img[0:img_size, 0:img_size]

    imgLT = img[0:first, 0:first]
    imgCT = img[0:first, first:mid]
    imgRT = img[0:first, mid:last]

    imgL = img[first:mid, 0:first]
    imgC = img[first:mid, first:mid]
    imgR = img[first:mid, mid:last]

    imgLB = img[mid:last, 0:first]
    imgCB = img[mid:last, first:mid]
    imgRB = img[mid:last, mid:last]

    imgLT = cv2.resize(imgLT, (cropped_size, cropped_size))
    imgCT = cv2.resize(imgCT, (cropped_size, cropped_size))
    imgRT = cv2.resize(imgRT, (cropped_size, cropped_size))

    imgL = cv2.resize(imgL, (cropped_size, cropped_size))
    imgC = cv2.resize(imgC, (cropped_size, cropped_size))
    imgR = cv2.resize(imgR, (cropped_size, cropped_size))

    imgLB = cv2.resize(imgLB, (cropped_size, cropped_size))
    imgCB = cv2.resize(imgCB, (cropped_size, cropped_size))
    imgRB = cv2.resize(imgRB, (cropped_size, cropped_size))

    return (
        imgFull,
        imgLT,
        imgCT,
        imgRT,
        imgL,
        imgC,
        imgR,
        imgLB,
        imgCB,
        imgRB,
    )
