import cv2
import numpy as np

from ..utils.yaml import Config

cfg = Config.load()

def get_aspect_ratio():
    pass


def get_n_pins():
    pass


def get_corner_count():
    pass


def get_pin_positions():
    pass


def corner_detection():
    pass


def edge_detection(img, method='canny', sigma=cfg['canny_sigma']):
    if method == 'canny':
        v = np.median(img)

        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edges = cv2.Canny(img, lower, upper)

    elif method == 'sobel':
        sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
        sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
        edges = cv2.magnitude(sobel_x, sobel_y)

    elif method == 'laplacian':
        edges = cv2.Laplacian(img, cv2.CV_64F)
        edges = np.uint8(np.absolute(edges))
    else:
        raise ValueError("Unsupported edge detection method")

    return edges


def morphological_operations(img, operation='dilate', kernel_size=cfg['morph_kernel_size'], iterations=cfg['morph_iterations']):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    if operation == 'dilate':
        img = cv2.dilate(img, kernel, iterations=iterations)
    elif operation == 'erode':
        img = cv2.erode(img, kernel, iterations=iterations)
    elif operation == 'open':
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=iterations)
    elif operation == 'close':
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    elif operation == 'connected_components':
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img)
        # What to do with the connected components? This is just a placeholder.
    else:
        raise ValueError("Unsupported morphological operation")

    return img


def histogram_gradients():
    pass
