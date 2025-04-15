import cv2
import numpy as np

from .helpers import calculate_histogram

def detect_contrast(image):
    hist = calculate_histogram(image)
    total_pixels = image.size

    mean_val = np.mean(image)
    median_val = np.median(image)
    std_dev = np.std(image)

    hist_norm = hist / total_pixels
    center_mass = np.sum(hist_norm[100:150])
    edges_mass = np.sum(hist_norm[:50]) + np.sum(hist_norm[200:])

    is_close_mean_median = abs(mean_val - median_val) < 10
    is_low_spread = std_dev < 40
    is_edge_heavy = edges_mass > center_mass
    is_high_spread = std_dev > 90  # high contrast: big spread

    if is_close_mean_median and is_low_spread and is_edge_heavy:
        return "low"
    elif is_high_spread and (edges_mass > 0.4):  # lots of whites/blacks
        return "high"
    else:
        return "normal"
    
def fix_low_contrast(image):
    """
    Enhances contrast using histogram equalization (applied to Y channel in YCrCb).
    Returns a BGR image with improved contrast.
    """
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y_eq = cv2.equalizeHist(y)
    ycrcb_eq = cv2.merge((y_eq, cr, cb))
    result = cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)
    return result

def run_contrast(image):
    print("-- Contrast Adjustment --")
    contrast = detect_contrast(image)
    if contrast == "low":
        return fix_low_contrast(image)
    else:
        print("No contrast adjustment needed, contrast is:", contrast)
    print("-- Contrast Adjustment Complete --")
    return image