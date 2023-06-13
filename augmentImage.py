from cropImage import cropImage


def augmentImage(img, img_size, samples, labels, label_map, roughness):
    imgFull, imgLT, imgCT, imgRT, imgL, imgC, imgR, imgLB, imgCB, imgRB = cropImage(
        img, img_size
    )

    # samples.append(imgFull)
    # labels.append(label_map[roughness])

    # samples.append(imgLT)
    # labels.append(label_map[roughness])

    # samples.append(imgCT)
    # labels.append(label_map[roughness])

    # samples.append(imgRT)
    # labels.append(label_map[roughness])

    samples.append(imgL)
    labels.append(label_map[roughness])

    samples.append(imgC)
    labels.append(label_map[roughness])

    samples.append(imgR)
    labels.append(label_map[roughness])

    # samples.append(imgLB)
    # labels.append(label_map[roughness])

    # samples.append(imgCB)
    # labels.append(label_map[roughness])

    # samples.append(imgRB)
    # labels.append(label_map[roughness])

    return samples, labels
