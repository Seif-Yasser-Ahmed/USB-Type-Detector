import cv2
import numpy as np
from ..utils.yaml import Config

cfg = Config.load()


def detect_sine_wave_noise(image, n_cycles=cfg.get("n_cycles", 5), angle_deg=cfg.get("angle_deg", 45), detection_threshold=cfg.get("sine_detection_threshold", 2.0)):
    """
    Detect sine wave pattern in the image by analyzing frequency components.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # FFT and magnitude spectrum
    f_transform = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f_transform)
    magnitude = np.abs(f_shift)

    # Frequency domain analysis
    rows, cols = gray.shape
    center_row, center_col = rows // 2, cols // 2
    diag_length = np.sqrt(rows ** 2 + cols ** 2)
    freq_radius = n_cycles / diag_length
    angle_rad = np.deg2rad(angle_deg)

    expected_row = int(center_row + freq_radius * np.sin(angle_rad) * rows)
    expected_col = int(center_col + freq_radius * np.cos(angle_rad) * cols)

    region_size = 5
    region = magnitude[
        max(0, expected_row - region_size):min(rows, expected_row + region_size + 1),
        max(0, expected_col - region_size):min(cols, expected_col + region_size + 1)
    ]

    peak_value = np.max(region)
    mean_value = np.mean(magnitude)
    spectrum_std = np.std(magnitude)

    # Optional filter: ignore flat spectra
    if spectrum_std < 20:
        return False

    return peak_value > (mean_value * detection_threshold)


def fix_sine_wave(image, n_cycles=cfg.get("n_cycles", 5), angle_deg=cfg.get("angle_deg", 45), opacity=cfg.get("opacity", 0.2)):
    """
    Remove sine wave pattern from the image.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    rows, cols = gray.shape
    diag_length = np.sqrt(rows ** 2 + cols ** 2)
    frequency = n_cycles / diag_length

    angle_rad = np.deg2rad(angle_deg)
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)

    x = np.arange(cols)
    y = np.arange(rows)
    X, Y = np.meshgrid(x, y)

    sine_wave = np.cos(2 * np.pi * frequency * (X * cos_angle + Y * sin_angle))
    sine_lines = ((1 - sine_wave) * 255 * opacity).astype(np.uint8)

    cleaned = cv2.subtract(gray, sine_lines)
    return np.clip(cleaned, 0, 255).astype(np.uint8)


def run_sine_wave(image):
    """
    Detect and remove sine wave noise if present.
    """
    print("-- Detecting sine wave noise pattern --")
    if detect_sine_wave_noise(image):
        print(":white_check_mark: Sine wave noise detected. Removing...")
        return fix_sine_wave(image)
    else:
        print(":green_circle: No sine wave noise detected.")
        return image