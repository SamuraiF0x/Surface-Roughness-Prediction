from cropImage import cropImage


def augmentImage(img, img_size, samples, labels, label_map, roughness):
    imgL, imgC, imgR = cropImage(img, img_size)

    samples.append(imgL)
    labels.append(label_map[roughness])

    samples.append(imgC)
    labels.append(label_map[roughness])

    samples.append(imgR)
    labels.append(label_map[roughness])

    return samples, labels
