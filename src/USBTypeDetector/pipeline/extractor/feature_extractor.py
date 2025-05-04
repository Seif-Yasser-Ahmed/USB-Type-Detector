import cv2
import numpy as np
import os
from ..preprocessor import image_preprocessor
from .geomtry_extractor import edge_detection, morphological_operations, histogram_of_gradients
from ..preprocessor.helpers import resize_and_pad


class FeatureExtractor:
    def __init__(self):
        pass

    @classmethod
    def extract(cls, image, type, cell_size=32):
        """
        logic
        """
        image = edge_detection(image)
        image = morphological_operations(image)
        image = resize_and_pad(
            image, (640, 480), interp=cv2.INTER_AREA, pad_color=(0, 0, 0))
        if type == 'hog':
            image = histogram_of_gradients(image, cell_size=cell_size)
        elif type == 'geometry':
            # image = get_n_pins(image)
            # image = get_pin_positions(image)
            pass
        return image
