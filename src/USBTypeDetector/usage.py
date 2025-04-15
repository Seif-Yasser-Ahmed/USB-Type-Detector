import cv2
from .pipeline.preprocessor import ImagePreprocessor

class USBTypeDetector:

    @staticmethod
    def run(image_path: str, save_path: str):

        image = cv2.imread(image_path)

        # Initialize the image preprocessor
        image_preprocessed = ImagePreprocessor.preprocess(image)

        cv2.imwrite(save_path, image_preprocessed)

        return image_preprocessed