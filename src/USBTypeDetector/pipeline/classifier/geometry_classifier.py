# from ..extractor import geomtry_extractor
import cv2


# class geomtry_classifier:

def classify_by_aspect_ratio():
    pass


def classify_by_number_pins():
    pass


def classify_by_position_pins():
    pass


def classify_by_geomtry():
    classify_by_position_pins()
    classify_by_number_pins()
    classify_by_aspect_ratio()
    pass

# @classmethod


def classify_knn(test_hog, hist_dict, metric=cv2.HISTCMP_CHISQR):
    """
    test_hog:  1D numpy array (floats)
    hist_dict: { class_label: [hog1, hog2, ...], ...}
    metric:    one of cv2.HISTCMP_* constants
    """
    best_label = None
    best_score = float("inf")
    for label, hog_list in hist_dict.items():
        for train_hog in hog_list:
            # convert to float32 histograms
            d = cv2.compareHist(test_hog.astype('float32'),
                                train_hog.astype('float32'),
                                metric)
            if d < best_score:
                best_score, best_label = d, label
    return best_label
