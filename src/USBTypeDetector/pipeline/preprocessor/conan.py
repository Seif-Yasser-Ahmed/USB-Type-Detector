import cv2
import numpy as np

# from ..utils import Config
from ..utils.yaml import Config


from .salt_pepper import run_salt_pepper
from .blur import run_blur
from .contrast import run_contrast
#from .brightness import run_brightness_analysis
from .helpers import sharpen_image
from .sine_wave import run_sine_wave

class DetectorFixer:

    @classmethod
    def run(cls, image):

        # 1. Detect and fix salt and pepper noise
        image = run_salt_pepper(image)

        # 2. Detect and fix blur
        image = run_blur(image)

        # 3. Detect and fix contrast
        image = run_contrast(image)

        # 4. Detect and fix brightness
        #image = run_brightness_analysis(image)

        # 5. Detect and fix sine wave noise
        image = run_sine_wave(image)

        # 6. Apply Sharpening
        image = sharpen_image(image)
        
        return image
