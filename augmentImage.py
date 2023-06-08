from cropImage import cropImage


def augmentImage(img, img_size, samples, labels, label_map, roughness):
    imgL = cropImage(img, img_size)
    samples.append(imgL)
    # Replace the string label with its corresponding integer value
    labels.append(label_map[roughness])

    imgR = cropImage(img, img_size, right=True)
    samples.append(imgR)
    # Replace the string label with its corresponding integer value
    labels.append(label_map[roughness])

    imgB = cropImage(img, img_size, bottom=True)
    samples.append(imgB)
    # Replace the string label with its corresponding integer value
    labels.append(label_map[roughness])
    
    return samples, labels
