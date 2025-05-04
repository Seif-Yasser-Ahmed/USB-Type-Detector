import cv2
from .conan import DetectorFixer
from .helpers import sharpen_image, binarize_image, resize_and_pad, extract_port_roi, blur_image
import matplotlib.pyplot as plt
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
        # image = cls._convert_to_grayscale(image)
        # plt.imshow(image, cmap='gray')
        # image = resize_and_pad(image, (640, 480), interp=cv2.INTER_AREA, pad_color=(0, 0, 0))
        # Detect and fix noise (salt & pepper, brightness, contrast)
        image = DetectorFixer.run(image)
        # plt.imshow(image, cmap='gray')
        image = blur_image(image)
        image = sharpen_image(image)
        image = extract_port_roi(image)
        # Apply sharpening
        # Apply thresholding (binarization)
        image = binarize_image(image)
        # save image
        return image
