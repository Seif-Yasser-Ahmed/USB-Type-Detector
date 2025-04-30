import cv2
import numpy as np
import os
from preprocessor import image_preprocessor


class FeatureExtractor:
    def __init__(self):
        pass

    @classmethod
    def extract(cls, image):
        """
        logic
        """
        return image
    
    @classmethod
    def harris_corner_detection(cls, img, block_size=2, ksize=3, k=0.04, thresh=0.02, min_distance=12):
        if img.ndim == 2:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            gray = img
        else:
            img_bgr = img.copy()
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        gray = cv2.GaussianBlur(gray, (5, 5), 1)
        gray = np.float32(gray)

        dst = cv2.cornerHarris(gray, block_size, ksize, k)
        dst = cv2.dilate(dst, None)

        corners_subpix = cv2.goodFeaturesToTrack(gray, maxCorners=1000, qualityLevel=thresh, minDistance=min_distance)
        if corners_subpix is not None:
            for pt in corners_subpix:
                x, y = pt.ravel().astype(int)
                cv2.circle(img_bgr, (x, y), 2, (0, 0, 255), -1)
            count = len(corners_subpix)
        else:
            count = 0

        return img_bgr, count
