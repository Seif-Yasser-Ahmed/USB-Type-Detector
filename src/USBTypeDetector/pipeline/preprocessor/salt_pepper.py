import cv2
import numpy as np
from ..utils.yaml import Config

# from pipeline.utils.yaml import Config

cfg = Config.load()


def detect_salt_pepper_noise(image, kernel_size=cfg['kernel_size_salt_pepper'], diff_threshold=cfg['diff_threshold'], min_fraction=cfg['min_fraction']):
    median_img = cv2.medianBlur(image, kernel_size)
    diff = image.astype(np.int16) - median_img.astype(np.int16)

    salt_pixels = diff >= diff_threshold
    pepper_pixels = diff <= -diff_threshold

    salt_count = np.count_nonzero(salt_pixels)
    pepper_count = np.count_nonzero(pepper_pixels)

    total_pixels = image.shape[0] * image.shape[1]

    salt_detected = (salt_count / total_pixels) >= min_fraction
    pepper_detected = (pepper_count / total_pixels) >= min_fraction

    if salt_detected and pepper_detected:
        return "salt and pepper"
    elif salt_detected:
        return "salt"
    elif pepper_detected:
        return "pepper"
    else:
        return "no noise"


def fix_salt_pepper(image, noise_type, kernel_size=cfg["kernel_size_salt_pepper"]):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    if noise_type.lower() == "salt":
        # Minimum filter removes bright (salt) noise.
        fixed = cv2.erode(image, kernel)
    elif noise_type.lower() == "pepper":
        # Maximum filter removes dark (pepper) noise.
        fixed = cv2.dilate(image, kernel)
    elif noise_type.lower() == "salt and pepper":
        fixed = cv2.medianBlur(image, kernel_size)    # Median filter for both.
    else:
        fixed = image.copy()
    return fixed


def run_salt_pepper(image, kernel_size=cfg["kernel_size_salt_pepper"], diff_threshold=cfg["diff_threshold"], min_fraction=cfg["min_fraction"]):
    print(
        f"-- Detecting salt and pepper noise with kernel size: {kernel_size}, diff threshold: {diff_threshold}, min fraction: {min_fraction} --")
    noise_type = detect_salt_pepper_noise(
        image, kernel_size, diff_threshold, min_fraction)
    print(f"Detected noise type: {noise_type}")
    fixed_image = fix_salt_pepper(image, noise_type, kernel_size)
    print("-- Salt and pepper noise fixed --")
    return fixed_image
