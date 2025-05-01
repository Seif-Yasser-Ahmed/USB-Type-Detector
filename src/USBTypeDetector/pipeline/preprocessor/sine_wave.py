import cv2
import numpy as np
from ..utils.yaml import Config

cfg = Config.load()

def detect_sine_wave_noise(image, n_cycles=cfg.get("n_cycles", 5), angle_deg=cfg.get("angle_deg", 45)):
    """
    Detect sine wave pattern in the image by analyzing frequency components.
    
    Args:
        image: Input grayscale image
        n_cycles: Expected number of sine wave cycles
        angle_deg: Expected angle of the sine wave pattern
    
    Returns:
        bool: True if sine wave pattern is detected
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply FFT
    f_transform = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f_transform)
    magnitude = np.abs(f_shift)
    
    # Create mask for expected frequency range
    rows, cols = gray.shape
    center_row, center_col = rows//2, cols//2
    
    # Calculate expected frequency location based on n_cycles and angle
    angle_rad = np.deg2rad(angle_deg)
    diag_length = np.sqrt(rows**2 + cols**2)
    freq_radius = n_cycles / diag_length
    
    expected_row = center_row + int(freq_radius * np.sin(angle_rad) * rows)
    expected_col = center_col + int(freq_radius * np.cos(angle_rad) * cols)
    
    # Check magnitude at expected frequency
    region_size = 5
    region = magnitude[
        max(0, expected_row-region_size):min(rows, expected_row+region_size+1),
        max(0, expected_col-region_size):min(cols, expected_col+region_size+1)
    ]
    
    peak_value = np.max(region)
    mean_value = np.mean(magnitude)
    
    # If peak value is significantly higher than mean, sine wave pattern is present
    return peak_value > mean_value * cfg.get("sine_detection_threshold", 2.0)

def fix_sine_wave(image, n_cycles=cfg.get("n_cycles", 5), angle_deg=cfg.get("angle_deg", 45), opacity=cfg.get("opacity", 0.2)):
    """
    Remove sine wave pattern from the image.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    angle = np.deg2rad(angle_deg)
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    
    rows, cols = gray.shape
    x = np.arange(cols)
    y = np.arange(rows)
    X, Y = np.meshgrid(x, y)
    
    diag_length = np.sqrt(rows**2 + cols**2)
    frequency = n_cycles / diag_length
    
    # Reconstruct the sine wave pattern
    sine_wave = np.cos(2 * np.pi * frequency * (X * cos_angle + Y * sin_angle))
    sine_lines = ((1 - sine_wave) * 255 * opacity).astype(np.uint8)
    
    # Subtract the pattern
    cleaned = cv2.subtract(gray, sine_lines)
    cleaned = np.clip(cleaned, 0, 255).astype(np.uint8)
    
    return cleaned

def run_sine_wave(image):
    """
    Detect and fix sine wave noise in the image.
    """
    print("-- Detecting sine wave noise pattern --")
    has_sine_noise = detect_sine_wave_noise(image)
    
    if has_sine_noise:
        print("Sine wave noise pattern detected")
        fixed_image = fix_sine_wave(image)
        print("-- Sine wave noise pattern removed --")
        return fixed_image
    else:
        print("No sine wave noise pattern detected")
        return image