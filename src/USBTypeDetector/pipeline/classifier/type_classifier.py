from extractor.feature_extractor import FeatureExtractor


class TypeClassifier:
    def __init__(self, model):
        pass

    @classmethod
    def classify(cls, image):
        """
        logic

        """

        # Preprocess the image
        image = FeatureExtractor.extract(image)

        # Classify the type of USB connector
        # logic

        return type
