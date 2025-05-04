import cv2
import numpy as np

from ..utils.yaml import Config

cfg = Config.load()


def get_aspect_ratio():
    pass


def get_n_pins():
    pass


def get_pin_positions():
    pass


def edge_detection(img, method='canny', sigma=cfg['canny_sigma']):
    if method == 'canny':
        v = np.median(img)

        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edges = cv2.Canny(img, lower, upper)

    elif method == 'sobel':
        sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
        sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
        edges = cv2.magnitude(sobel_x, sobel_y)

    elif method == 'laplacian':
        edges = cv2.Laplacian(img, cv2.CV_64F)
        edges = np.uint8(np.absolute(edges))
    else:
        raise ValueError("Unsupported edge detection method")

    return edges


def morphological_operations(img, operation='close', kernel_size=cfg['morph_kernel_size'], iterations=cfg['morph_iterations']):
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (kernel_size, kernel_size))

    if operation == 'dilate':
        img = cv2.dilate(img, kernel, iterations=iterations)
    elif operation == 'erode':
        img = cv2.erode(img, kernel, iterations=iterations)
    elif operation == 'open':
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN,
                               kernel, iterations=iterations)
    elif operation == 'close':
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE,
                               kernel, iterations=iterations)
    elif operation == 'connected_components':
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            img)
        # What to do with the connected components? This is just a placeholder.
    else:
        raise ValueError("Unsupported morphological operation")

    return img


def histogram_of_gradients(img, cell_size=32, bin_n=8):
    """
    Compute a HOG-like visualization:
      – img: input BGR or grayscale image (uint8)
      – cell_size: size of each cell in pixels
      – bin_n: number of orientation bins (e.g. 8 for 0–360° in 45° steps)
    Returns:
      – hog_img: a grayscale image with arrows showing per-cell dominant gradient directions
    """
    # 1) Grayscale & gradients
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=1)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)

    # 2) Prepare bins & output image
    h, w = gray.shape
    cell_x = w // cell_size
    cell_y = h // cell_size
    # blank canvas for visualization
    hog_img = np.zeros((h, w), dtype=np.uint8)

    # 3) Loop over cells
    for i in range(cell_y):
        for j in range(cell_x):
            # cell boundaries
            y0, y1 = i*cell_size, (i+1)*cell_size
            x0, x1 = j*cell_size, (j+1)*cell_size

            # slice out this cell’s mags & angles
            cell_mag = mag[y0:y1, x0:x1].ravel()
            cell_ang = ang[y0:y1, x0:x1].ravel()

            # histogram of orientations weighted by magnitude
            bin_width = 360 / bin_n
            # map angles into [0, bin_n-1]
            bin_idxs = np.floor(cell_ang / bin_width).astype(int) % bin_n
            hist = np.bincount(bin_idxs, weights=cell_mag, minlength=bin_n)

            # normalize for display
            if hist.sum() != 0:
                hist = hist / hist.max()

            # draw an arrow for each bin
            center_x = x0 + cell_size // 2
            center_y = y0 + cell_size // 2
            for b in range(bin_n):
                # angle for this bin (in radians)
                theta = (b * bin_width + bin_width/2) * np.pi / 180
                length = (cell_size//2) * hist[b]
                dx = length * np.cos(theta)
                dy = length * np.sin(theta)

                x_start = int(center_x - dx/2)
                y_start = int(center_y - dy/2)
                x_end = int(center_x + dx/2)
                y_end = int(center_y + dy/2)

                # draw white line (thickness=1)
                cv2.line(hog_img,
                         (x_start, y_start),
                         (x_end,   y_end),
                         color=255,
                         thickness=1)

    return hog_img


def corner_detection_count(img, block_size=2, ksize=3, k=0.04, thresh=0.02, min_distance=12):
    if img.ndim == 2:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        gray = img
    else:
        img_bgr = img.copy()
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray, (5, 5), 1)
    gray = np.float32(gray)

    dst = cv2.cornerHarris(gray, block_size, ksize, k)
    dst = cv2.dilate(dst, None)

    corners_subpix = cv2.goodFeaturesToTrack(
        gray, maxCorners=1000, qualityLevel=thresh, minDistance=min_distance)
    if corners_subpix is not None:
        for pt in corners_subpix:
            x, y = pt.ravel().astype(int)
            cv2.circle(img_bgr, (x, y), 2, (0, 0, 255), -1)
        count = len(corners_subpix)
    else:
        count = 0

    return img_bgr, count
