import cv2
import numpy as np


class ImageAugmentation:
    """
    A class for performing various image augmentation techniques.
    The augmentation functions now take an image as an argument.
    """

    @classmethod
    def augment_images_flipping(self, image, **kwargs):
        """
        Augment a given image by flipping it horizontally, vertically, or both.
        Returns the augmented image.
        """
        if image is None:
            return None

        if kwargs.get('horizontal_flip', False):
            return cv2.flip(image, 1)
        elif kwargs.get('vertical_flip', False):
            return cv2.flip(image, 0)
        else:
            # If no specific flag provided, flip both axes
            return cv2.flip(image, -1)

    @classmethod
    def augment_images_rotation(self, image, **kwargs):
        """
        Augment a given image by rotating it by a specified angle.
        Returns the rotated image.
        """
        if image is None:
            return None

        h, w = image.shape[:2]
        center = (w // 2, h // 2)

        if kwargs.get('rotate_45', False):
            angle = 45
        elif kwargs.get('rotate_90', False):
            angle = 90
        elif kwargs.get('rotate_135', False):
            angle = 135
        elif kwargs.get('rotate_225', False):
            angle = 225
        elif kwargs.get('rotate_270', False):
            angle = 270
        elif kwargs.get('rotate_315', False):
            angle = 315
        else:
            # if rotation flag not provided, return original image
            return image

        M = cv2.getRotationMatrix2D(center, angle, 1)
        rotated_img = cv2.warpAffine(image, M, (w, h))
        return rotated_img

    @classmethod
    def augment_images_cropping(self, image, **kwargs):
        """
        Augment a given image by cropping its center to various sizes.
        Returns the cropped image.
        """
        if image is None:
            return None

        h, w = image.shape[:2]
        if kwargs.get('crop_50_percent', False):
            cropped_img = image[int(h*0.25):int(h*0.75),
                                int(w*0.25):int(w*0.75)]
        elif kwargs.get('crop_25_percent', False):
            cropped_img = image[int(h*0.375):int(h*0.625),
                                int(w*0.375):int(w*0.625)]
        elif kwargs.get('crop_10_percent', False):
            cropped_img = image[int(h*0.45):int(h*0.55),
                                int(w*0.45):int(w*0.55)]
        else:
            return image

        return cropped_img

    @classmethod
    def augment_images_blurring(self, image, **kwargs):
        """
        Apply Gaussian blur to a given image.
        Returns the blurred image.
        """
        if image is None:
            return None

        if kwargs.get('blur_5x5', False):
            return cv2.GaussianBlur(image, (5, 5), 0)
        elif kwargs.get('blur_15x15', False):
            return cv2.GaussianBlur(image, (15, 15), 0)
        elif kwargs.get('blur_25x25', False):
            return cv2.GaussianBlur(image, (25, 25), 0)
        else:
            return image

    @classmethod
    def augment_images_salt_pepper(self, image, **kwargs):
        """
        Add salt and/or pepper noise to a given image.
        Returns the noisy image.
        """
        if image is None:
            return None

        s_vs_p = 0.5
        amount = 0.04
        out = np.copy(image)

        if kwargs.get('salt_only', False):
            num_salt = np.ceil(amount * image.size * s_vs_p)
            coords = [np.random.randint(0, i, int(num_salt))
                      for i in image.shape[:2]]
            out[tuple(coords)] = 255
        elif kwargs.get('pepper_only', False):
            num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i, int(num_pepper))
                      for i in image.shape[:2]]
            out[tuple(coords)] = 0
        elif kwargs.get('salt_and_pepper', False):
            num_salt = np.ceil(amount * image.size * s_vs_p)
            coords = [np.random.randint(0, i, int(num_salt))
                      for i in image.shape[:2]]
            out[tuple(coords)] = 255
            num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i, int(num_pepper))
                      for i in image.shape[:2]]
            out[tuple(coords)] = 0
        else:
            return image

        return out

    @classmethod
    def augment_images_contrast(self, image, **kwargs):
        """
        Adjust the contrast of a given image.
        Returns the contrast-adjusted image.
        """
        if image is None:
            return None

        if kwargs.get('contrast_0.1', False):
            alpha = 0.1
        elif kwargs.get('contrast_0.5', False):
            alpha = 0.5
        elif kwargs.get('contrast_0.75', False):
            alpha = 0.75
        elif kwargs.get('contrast_1', False):
            alpha = 1.0
        elif kwargs.get('contrast_1.5', False):
            alpha = 1.5
        elif kwargs.get('contrast_2', False):
            alpha = 2.0
        elif kwargs.get('contrast_2.5', False):
            alpha = 2.5
        elif kwargs.get('contrast_3', False):
            alpha = 3.0
        else:
            return image

        contrast_img = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
        return contrast_img

    @classmethod
    def augment_images_overlay(self, image, **kwargs):
        """
        Overlay a solid color layer (black for darken or white for lighten) on a given image.
        Returns the augmented image.
        """
        mode = kwargs.get('mode', 'darken')
        opacity = kwargs.get('opacity', 0.7)

        if mode not in ('darken', 'lighten'):
            raise ValueError("mode must be 'darken' or 'lighten'")
        if not (0.0 <= opacity <= 1.0):
            raise ValueError("opacity must be between 0.0 and 1.0")

        if image is None:
            return None

        if mode == 'darken':
            overlay = np.zeros_like(image)
        else:  # lighten
            overlay = np.full_like(image, 255)

        out = cv2.addWeighted(image, 1.0 - opacity, overlay, opacity, 0)
        return out

    @classmethod
    def augment_images_brightness(self, image, **kwargs):
        """
        Adjust the brightness of a given image.
        Returns the brightness-adjusted image.
        """
        if image is None:
            return None

        if kwargs.get('brightness_-100', False):
            beta = -100
        elif kwargs.get('brightness_-50', False):
            beta = -50
        elif kwargs.get('brightness_-30', False):
            beta = -30
        elif kwargs.get('brightness_-20', False):
            beta = -20
        elif kwargs.get('brightness_-10', False):
            beta = -10
        elif kwargs.get('brightness_0', False):
            beta = 0
        elif kwargs.get('brightness_10', False):
            beta = 10
        elif kwargs.get('brightness_20', False):
            beta = 20
        elif kwargs.get('brightness_30', False):
            beta = 30
        elif kwargs.get('brightness_50', False):
            beta = 50
        elif kwargs.get('brightness_100', False):
            beta = 100
        elif kwargs.get('brightness_150', False):
            beta = 150
        else:
            return image

        brightness_img = cv2.convertScaleAbs(image, alpha=1.0, beta=beta)
        return brightness_img

    @staticmethod
    def augment_images_wave(image: np.ndarray, point: tuple) -> np.ndarray:
        """
        Take a 2D spatial-domain image, boost the magnitude at a given frequency coordinate,
        then return the modified image in the spatial domain.

        Parameters:
            image (2D numpy array): Input grayscale image.
            point (tuple): (u, v) frequency coordinate relative to center (0,0).

        Returns:
            img_mod (2D numpy array): Real-valued spatial-domain image after modification.
        """
        # 1) Forward FFT
        f = np.fft.fft2(image)
        fshift = np.fft.fftshift(f)

        # 2) Locate center and map (u,v)
        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2
        u, v = point
        r_idx = crow + v
        c_idx = ccol + u
        if not (0 <= r_idx < rows and 0 <= c_idx < cols):
            raise ValueError(
                f"Point {point} is outside frequency-domain bounds.")

        # 3) Boost that bin to the current maximum magnitude
        mag = np.abs(fshift)
        max_mag = mag.max()
        fshift[r_idx, c_idx] = max_mag

        # 4) Inverse FFT back to spatial domain
        f_ishift = np.fft.ifftshift(fshift)
        img_back = np.fft.ifft2(f_ishift)

        # 5) Return the real component
        return np.real(img_back)
