import cv2
import numpy as np
# from ..utils import Config

from .helpers import sharpen_image

from ..utils.yaml import Config

cfg = Config.load()

# worked better


def detect_blur_variance(image, threshold=cfg['blur_variance_threshold']):

    # Compute the Laplacian
    lap = cv2.Laplacian(image, cv2.CV_64F)
    var = lap.var()
    # print(f"Variance of Laplacian: {var:.2f}")
    return var < threshold


# didn't work well
def detect_blur_fft(image, size=cfg['blur_fft_size'], thresh=cfg['blur_fft_threshold']):

    # Compute FFT and shift zero‑freq to center
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    # Zero out the central low‑frequency region
    h, w = image.shape
    cy, cx = h // 2, w // 2
    fshift[cy-size//2:cy+size//2, cx-size//2:cx+size//2] = 0
    # Inverse shift & inverse FFT
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    # Count how many pixels exceed the threshold
    count = np.sum(img_back > thresh)
    # print(f"High‑freq coefficient count: {count}")
    # If too few high‑freq pixels, it's blurred
    return count > (image.shape[0] * image.shape[1] * 0.01)


def run_blur(image, method='variance'):
    """
    Detects if an image is blurred using the specified method.

    Parameters:
        image (numpy.ndarray): The input image.
        method (str): The method to use for blur detection ('variance' or 'fft').

    Returns:
        bool: True if the image is blurred, False otherwise.
    """

    # print("-- Running blur detection --")
    if method == 'variance':
        while detect_blur_variance(image, threshold=cfg['blur_variance_threshold']):
            # print("Image is blurry, sharpening...")
            image = sharpen_image(image)
        # print("-- Blur detection complete --")
        return image
    elif method == 'fft':
        while detect_blur_fft(image, size=cfg['blur_fft_size'], thresh=cfg['blur_fft_threshold']):
            # print("Image is blurry, sharpening...")
            image = sharpen_image(image)
        # print("-- Blur detection complete --")
        return image
    else:
        raise ValueError("Invalid method. Use 'variance' or 'fft'.")
