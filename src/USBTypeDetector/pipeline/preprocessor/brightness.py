import cv2
import numpy as np

from .helpers import calculate_histogram

def detect_global_brightness_mean(hist, threshold=30):
    """
    Detects if an image is bright or dark based on the mean pixel value.

    Args:
        hist (numpy array):Histogram of the input image
        threshold (int): Threshold for brightness detection

    Returns:
        str: 'bright' or 'dark'
    """
    mean_val = np.mean(hist)
    if mean_val < threshold:
        return "low brightness"
    elif mean_val > 255-threshold :
        return "high brightness"
    else:
        return "balanced brightness"
    
def detect_global_brightness_percentile(hist,threshold=50):
    """
    Evaluates the exposure of an image based on its histogram.

    Args:
        hist (numpy.ndarray): Histogram of the image.

    Returns:
        str: Exposure evaluation ('low brightness', 'high brightness', or 'balanced brightness').
    """
    cdf = hist.cumsum() / hist.sum()
    dark_level = np.searchsorted(cdf, 0.05)    # 5th percentile
    bright_level = np.searchsorted(cdf, 0.95)  # 95th percentile

    if bright_level < 255-threshold:
        return "low brightness"
    elif dark_level > threshold:
        return "high brightness"
    else:
        return "balanced brightness"

def detect_local_low_high_brightness_blocks(image, block_size=16,threshold=30):
    h, w = image.shape
    
    dark_blocks = []
    bright_blocks = []
    
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            block = image[y:y+block_size, x:x+block_size]
            m = block.mean()
            if m < threshold:
                dark_blocks.append((x, y))
            if m > 255-threshold:
                bright_blocks.append((x, y))

    return dark_blocks, bright_blocks

def analyze_dark_bright_channels_single(image, kernel_size=(15, 15)):
    """
    Analyzes dark and bright channels of a single image to detect over-exposed and under-exposed regions.

    Args:
        img_path (str): File path for the image.
        kernel_size (tuple): Size of the structuring element for erosion and dilation.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    dark = cv2.erode(np.min(image, axis=2), kernel)
    bright = cv2.dilate(np.max(image, axis=2), kernel)
    # Measure overall brightness
    if np.percentile(bright, 90) > 240:
        print("high brightness")
    if np.percentile(dark, 10) < 5:
        print("low brightness")
    print()

def global_gamma_correction(image, gamma=1.0):
    """
    img   : uint8 BGR image
    gamma : >0, <1 brightens, >1 darkens
    """
    # Build lookup table: 0–255 → [0,1]^(gamma) → 0–255
    inv_gamma = 1.0 / gamma
    table = np.array([
        ((i / 255.0) ** inv_gamma) * 255
        for i in np.arange(256)]).astype("uint8")

    # Apply LUT
    return cv2.LUT(image, table)

def fix_low_brightness_stretch(image, low_pct=5, high_pct=95):
    """
    Brightness fix by linearly stretching the V channel so that
    the low_pct percentile → 0 and the high_pct percentile → 255.
    
    Parameters
    ----------
    img : uint8 BGR image
    low_pct, high_pct : percentiles for clipping
    
    Returns
    -------
    uint8 BGR image
    """
    # 1) to HSV
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # 2) find the percentile values
    lo = np.percentile(v, low_pct)
    hi = np.percentile(v, high_pct)
    if hi <= lo:  # avoid divide-by-zero
        return image.copy()

    # 3) stretch
    v_stretched = np.clip((v.astype(np.float32) - lo) * 255.0/(hi - lo), 0, 255)
    v_stretched = v_stretched.astype(np.uint8)

    # 4) merge back
    hsv_stretch = cv2.merge([h, s, v_stretched])
    image = cv2.cvtColor(hsv_stretch, cv2.COLOR_HSV2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

def local_gamma_correction_blocks(image, block_size=16, threshold=30,
                                  gamma_dark=0.5, gamma_bright=2.0,
                                  blur_ksize=None):
    """
    img             : uint8 BGR image
    block_size      : same as in detect_…
    threshold       : same as in detect_…
    gamma_dark      : γ to apply on dark blocks ( <1 to brighten )
    gamma_bright    : γ to apply on bright blocks ( >1 to darken )
    blur_ksize      : kernel size for smoothing γ‐map; defaults to 2*block_size+1
    """
    # 1) detect blocks on the luminance
    dark_blocks, bright_blocks = detect_local_low_high_brightness_blocks(
        image, block_size, threshold
    )

    # 2) build a γ‐map initialized to 1.0
    h, w = image.shape
    gamma_map = np.ones((h, w), dtype=np.float32)

    # 3) assign block‐wise γ
    for x, y in dark_blocks:
        gamma_map[y:y+block_size, x:x+block_size] = gamma_dark
    for x, y in bright_blocks:
        gamma_map[y:y+block_size, x:x+block_size] = gamma_bright

    # 4) smooth the γ‐map to avoid hard edges
    if blur_ksize is None:
        blur_ksize = 2 * block_size + 1
    gamma_map = cv2.GaussianBlur(gamma_map, (blur_ksize, blur_ksize), 0)

    # 5) apply per‐pixel gamma
    img_f = image.astype(np.float32) / 255.0
    out = np.zeros_like(img_f)
    for c in range(3):
        out[:, :, c] = np.power(img_f[:, :, c], gamma_map)

    return (np.clip(out, 0, 1) * 255).astype(np.uint8)

def run_brightness_analysis(image,method="global",detector="percentile",fixer="stretch"):
    print("-- Brightness analysis --")
    if method=="global":
        if detector=="dark-channels":
            image = analyze_dark_bright_channels_single(image, kernel_size=(15, 15))
        elif detector=="mean":
            image = detect_global_brightness_mean(hist=calculate_histogram(image), threshold=30)
        elif detector=="percentile":
            image = detect_global_brightness_percentile(hist=calculate_histogram(image), threshold=30)
        
    elif method=="local":
        if detector=="blocks":
            image = detect_local_low_high_brightness_blocks(image, block_size=16, threshold=30)
    else:
        raise ValueError("Invalid method. Choose 'global' or 'local'.")
    
    if fixer=="gamma":
        image = global_gamma_correction(image, gamma=1.5)
    elif fixer=="stretch":
        image = fix_low_brightness_stretch(image, low_pct=5, high_pct=95)


    print("-- End of brightness analysis --")
    return image
