import cv2
import numpy as np
from ..utils.yaml import Config
from ..extractor.geomtry_extractor import edge_detection
cfg = Config.load()


def sharpen_image(image):

    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened_img = cv2.filter2D(image, -1, kernel)

    return sharpened_img


def blur_image(image, blur_strength=7):
    blurred_img = cv2.GaussianBlur(image, (blur_strength, blur_strength), 0)
    return blurred_img


def calculate_histogram(image):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    return hist


def binarize_image(image, threshold=cfg['binarization_threshold']):
    _, binary_img = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return binary_img


def resize_and_pad(img, target_size, interp=cv2.INTER_AREA, pad_color=(0, 0, 0)):
    """
    Resize an image to fit inside target_size, then pad to exactly match target_size.

    Parameters
    ----------
    img : ndarray
        Input image (HxW or HxWxC).
    target_size : tuple (width, height)
        Desired output dimensions.
    interp : int
        cv2 interpolation flag (INTER_AREA for shrinking, INTER_CUBIC for enlarging).
    pad_color : tuple
        BGR color for the padding; e.g. (0,0,0) for black.

    Returns
    -------
    padded : ndarray
        The resized+padded image of shape exactly target_size.
    """
    orig_h, orig_w = img.shape[:2]
    target_w, target_h = target_size

    # 1. Compute scale to fit image inside target box
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)

    # 2. Resize
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)

    # 3. Compute padding amounts
    pad_w = target_w - new_w
    pad_h = target_h - new_h
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    # 4. Pad and return
    padded = cv2.copyMakeBorder(
        resized,
        top, bottom, left, right,
        borderType=cv2.BORDER_CONSTANT,
        value=pad_color
    )
    return padded

def extract_port_roi(orig_bgr):
    """
    orig_bgr:    your original color (or gray) image
    edge_mask:   binary edge image (0 background, 255 edges)
    returns roi_bgr, (x,y,w,h)
    """
    edge_mask = edge_detection(orig_bgr)
    # 1) (Optional) Close small gaps in the edge mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edge_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 2) Fill interior to get a solid blob
    #    invert, flood‑fill from (0,0), invert back, OR with closed
    inv = cv2.bitwise_not(closed)
    h, w = inv.shape
    flood = inv.copy()
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(flood, mask, (0, 0), 255)
    filled = cv2.bitwise_not(flood)  # now the port interior is white

    blob = cv2.bitwise_or(closed, filled)

    # 3) Find external contours
    cnts, _ = cv2.findContours(
        blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None

    # 4) Pick the contour with the largest area
    c = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    # 5) Crop and return
    roi = orig_bgr[y:y+h, x:x+w].copy()
    return roi
