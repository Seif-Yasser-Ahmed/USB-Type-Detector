import cv2
from .conan import DetectorFixer
# Input
# -- Image Preprocessor --
# Greyscale
# Detect Noise & Fix
# - Salt & Pepper
# - Brightness
# - Contrast
# Sharpening
# Thresholding (Binarization)

class ImagePreprocessor:

    @classmethod
    def _convert_to_grayscale(cls, image):
        """
        Converts an image to grayscale.

        Args:
            image (numpy.ndarray): The input image to be converted.
        """
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray_img
    
    @classmethod
    def preprocess(cls, image):
        # Convert to grayscale
        image = cls._convert_to_grayscale(image)
        # Detect and fix noise (salt & pepper, brightness, contrast)
        image = DetectorFixer.run(image)
        # Apply sharpening
        # Apply thresholding (binarization)
        # save image
        return image