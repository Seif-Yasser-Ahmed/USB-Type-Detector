import cv2
import numpy as np

def sharpen_image(cls, image):

    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened_img = cv2.filter2D(image, -1, kernel)

    return sharpened_img