import cv2
import numpy as np
from ..utils.yaml import Config

cfg = Config.load()


# def detect_sine_wave_noise(image, n_cycles=cfg.get("n_cycles", 5), angle_deg=cfg.get("angle_deg", 45), detection_threshold=cfg.get("sine_detection_threshold", 2.0)):
#     """
#     Detect sine wave pattern in the image by analyzing frequency components.
#     """
#     if len(image.shape) == 3:
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     else:
#         gray = image.copy()

#     # FFT and magnitude spectrum
#     f_transform = np.fft.fft2(gray)
#     f_shift = np.fft.fftshift(f_transform)
#     magnitude = np.abs(f_shift)

#     # Frequency domain analysis
#     rows, cols = gray.shape
#     center_row, center_col = rows // 2, cols // 2
#     diag_length = np.sqrt(rows ** 2 + cols ** 2)
#     freq_radius = n_cycles / diag_length
#     angle_rad = np.deg2rad(angle_deg)

#     expected_row = int(center_row + freq_radius * np.sin(angle_rad) * rows)
#     expected_col = int(center_col + freq_radius * np.cos(angle_rad) * cols)

#     region_size = 5
#     region = magnitude[
#         max(0, expected_row - region_size):min(rows, expected_row + region_size + 1),
#         max(0, expected_col - region_size):min(cols, expected_col + region_size + 1)
#     ]

#     peak_value = np.max(region)
#     mean_value = np.mean(magnitude)
#     spectrum_std = np.std(magnitude)

#     # Optional filter: ignore flat spectra
#     if spectrum_std < 20:
#         return False

#     return peak_value > (mean_value * detection_threshold)


# def fix_sine_wave(image, n_cycles=cfg.get("n_cycles", 5), angle_deg=cfg.get("angle_deg", 45), opacity=cfg.get("opacity", 0.2)):
#     """
#     Remove sine wave pattern from the image.
#     """
#     if len(image.shape) == 3:
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     else:
#         gray = image.copy()

#     rows, cols = gray.shape
#     diag_length = np.sqrt(rows ** 2 + cols ** 2)
#     frequency = n_cycles / diag_length

#     angle_rad = np.deg2rad(angle_deg)
#     cos_angle = np.cos(angle_rad)
#     sin_angle = np.sin(angle_rad)

#     x = np.arange(cols)
#     y = np.arange(rows)
#     X, Y = np.meshgrid(x, y)

#     sine_wave = np.cos(2 * np.pi * frequency * (X * cos_angle + Y * sin_angle))
#     sine_lines = ((1 - sine_wave) * 255 * opacity).astype(np.uint8)

#     cleaned = cv2.subtract(gray, sine_lines)
#     return np.clip(cleaned, 0, 255).astype(np.uint8)


# def run_sine_wave(image):
#     """
#     Detect and remove sine wave noise if present.
#     """
#     print("-- Detecting sine wave noise pattern --")
#     if detect_sine_wave_noise(image):
#         print(":white_check_mark: Sine wave noise detected. Removing...")
#         return fix_sine_wave(image)
#     else:
#         print(":green_circle: No sine wave noise detected.")
#         return image


def detect_pattern_noise(image: np.ndarray) -> tuple[int, int]:
    """
    Find the strongest non-DC frequency in a real-valued image's 2D FFT,
    and return the coordinate in the positive half-plane.

    Parameters:
        image (2D numpy array): Input grayscale image.

    Returns:
        (u, v): Frequency offset from center, choosing
                the “real” (user-set) location rather than its conjugate twin.
    """
    # 1) FFT & center
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)

    # 2) Magnitude and suppress DC
    mag = np.abs(fshift)
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    mag[crow, ccol] = 0

    # 3) Find any max index
    idx = np.argmax(mag)
    r_idx, c_idx = np.unravel_index(idx, mag.shape)

    # 4) Convert to (u,v) relative coords
    u = c_idx - ccol
    v = r_idx - crow

    # 5) If we landed in the “negative” mirror, flip to the positive half
    if (v < 0) or (v == 0 and u < 0):
        u, v = -u, -v

    return (u, v)


def run_sine_wave(image: np.ndarray) -> np.ndarray:
    """
    Find the strongest non-DC frequency in `image`, set that coefficient and its conjugate mirror
    in the frequency domain to the lowest value (0), then invert back to the spatial domain and return
    the real-valued result.

    Parameters:
        image (2D numpy array): Input grayscale image.

    Returns:
        img_mod (2D numpy array): Spatial-domain image after removing the peak frequency.
    """
    # 1) Compute centered FFT
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)

    # 2) Locate the strongest non-DC frequency (u, v) relative to center
    u, v = detect_pattern_noise(image)

    # 3) Map (u, v) to array indices in the shifted spectrum
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    r_idx = crow + v
    c_idx = ccol + u
    if not (0 <= r_idx < rows and 0 <= c_idx < cols):
        raise ValueError(f"Mapped index {(r_idx, c_idx)} out of bounds.")

    # 4) Zero out the coefficient and its conjugate mirror.
    lowest = 0  # lowest value is set to 0
    fshift[r_idx, c_idx] = lowest

    mirror_r_idx = crow - v
    mirror_c_idx = ccol - u
    if 0 <= mirror_r_idx < rows and 0 <= mirror_c_idx < cols:
        fshift[mirror_r_idx, mirror_c_idx] = lowest

    # 5) Inverse shift and inverse FFT back to spatial domain
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)

    # 6) Return the real part
    return np.real(img_back)
