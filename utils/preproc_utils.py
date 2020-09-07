import cv2


def binarization(img):
    """
    Apply adaptive binarization with different kernel size and choose the one
    with less components (=less noise), invert the result
    """
    img_binary_17 = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 17, 17
    )
    img_inv_17 = cv2.bitwise_not(img_binary_17)
    nlabels_17, labels_17, stats_17, centroids_17 = \
        cv2.connectedComponentsWithStats(
            img_inv_17, None, None, None, 4, cv2.CV_32S
        )

    img_binary_15 = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 15
    )
    img_inv_15 = cv2.bitwise_not(img_binary_15)
    nlabels_15, labels_15, stats_15, centroids_15 = \
        cv2.connectedComponentsWithStats(
            img_inv_15, None, None, None, 4, cv2.CV_32S
        )
    if nlabels_17 <= nlabels_15:
        return img_inv_17
    else:
        return img_inv_15
