from ..extractor.feature_extractor import FeatureExtractor
from ..preprocessor.image_preprocessor import ImagePreprocessor
import cv2
from .geometry_classifier import classify_knn, classify_by_aspect_ratio, classify_by_number_pins, classify_by_position_pins, classify_by_geomtry


class TypeClassifier:
    def __init__(self, hist_dict, classification_type):
        self.hist_dict = hist_dict
        self.classify_type = classification_type

    @classmethod
    def run(cls, image, hist_dict, type='geomtry', cell_size=32):
        # Preprocess the image
        image = ImagePreprocessor.preprocess(image)
        # Extract features
        image = FeatureExtractor.extract(image, type, cell_size=cell_size)
        if type == 'geometry':
            # Classify using geometry-based method
            type = classify_by_geomtry(image, hist_dict)
        elif type == 'hog':
            type = classify_knn(image, hist_dict,
                                metric=cv2.HISTCMP_CHISQR)
        return type
