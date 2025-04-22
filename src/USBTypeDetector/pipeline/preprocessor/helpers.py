import cv2
import numpy as np
from ..utils.yaml import Config
cfg = Config.load()


def sharpen_image(image):

    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened_img = cv2.filter2D(image, -1, kernel)

    return sharpened_img


def calculate_histogram(image):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    return hist


def binarize_image(image, threshold=cfg['binarization_threshold']):
    _, binary_img = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return binary_img
