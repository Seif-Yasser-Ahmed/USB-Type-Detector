import os
import cv2
import csv
import numpy as np
from USBTypeDetector.pipeline.preprocessor.image_preprocessor import ImagePreprocessor
from USBTypeDetector.pipeline.extractor.feature_extractor import FeatureExtractor
from USBTypeDetector.pipeline.augmenter.image_augmentor import ImageAugmentation
from USBTypeDetector.pipeline.classifier.geometry_classifier import classify_knn
from USBTypeDetector.pipeline.utils.yaml import Config
def main(hog_dict):
    # def open_camera_with_noise_buttons(hog_dict):
    def my_classification_function(img):
        img = ImagePreprocessor.preprocess(img)
        hog_img=FeatureExtractor.extract(img, 'hog',cell_size=8)
        print("HOG Image shape:", hog_img.shape)
        hog_vector = hog_img.flatten()
        return classify_knn(hog_vector,hog_dict, k=7)

    def add_salt_noise(img, param):
        return ImageAugmentation.augment_images_salt_pepper(img, **{param: True})

    def add_pepper_noise(img, param):
        return ImageAugmentation.augment_images_salt_pepper(img, **{param: True})

    def add_salt_pepper_noise(img, param):
        return ImageAugmentation.augment_images_salt_pepper(img, **{param: True})

    def add_sine_wave_noise(img, n_cycles=5, angle_deg=45, opacity=0.2):
        return ImageAugmentation.augment_images_wave(img, point=(0, 7))

    def add_contrast_noise(img, param):
        return ImageAugmentation.augment_images_contrast(img, **{param: True})

    def add_blur_noise(img, param):
        return ImageAugmentation.augment_images_blurring(img, **{param: True})

    default_levels = {
        'blurs': ['blur_5x5', 'blur_15x15', 'blur_25x25'],
        'salt_peppers': ['salt_only', 'pepper_only', 'salt_and_pepper'],
        'contrasts': ['contrast_0.1',
                      'contrast_0.5',
                      'contrast_0.75',
                      'contrast_1',
                      'contrast_1.5',
                      'contrast_2',
                      'contrast_2.5',
                      'contrast_3']
    }

    noise_to_defaults = {
        'salt': default_levels['salt_peppers'],
        'pepper': default_levels['salt_peppers'],
        'salt_pepper': default_levels['salt_peppers'],
        'contrast': default_levels['contrasts'],
        'blur': default_levels['blurs']
    }

    noise_order = ['salt', 'pepper', 'salt_pepper', 'sine', 'contrast', 'blur']
    active_noises = {}
    instructions = [
        "Keys: s-Salt, p-Pepper, a-S&P, f-Sine, c-Contrast, b-Blur, ",
        "n-Clear, + Next, d-Classify, q-Quit"
    ]
    classification_result = "USB Type: Not classified"
    last_frame = None

    # window setup
    window_name = "Camera with Noise"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    # Fixed rectangle size
    rect_width, rect_height = 308, 233

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        last_frame = frame.copy()
        display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        for noise in noise_order:
            if noise in active_noises:
                param = active_noises[noise]
                if noise == 'salt':
                    display_frame = add_salt_noise(display_frame, param)
                elif noise == 'pepper':
                    display_frame = add_pepper_noise(display_frame, param)
                elif noise == 'salt_pepper':
                    display_frame = add_salt_pepper_noise(display_frame, param)
                elif noise == 'sine':
                    display_frame = add_sine_wave_noise(display_frame)
                elif noise == 'contrast':
                    display_frame = add_contrast_noise(display_frame, param)
                elif noise == 'blur':
                    display_frame = add_blur_noise(display_frame, param)

        norm = cv2.normalize(display_frame, None, 0, 255, cv2.NORM_MINMAX)
        norm_uint8 = norm.astype(np.uint8)
        display_frame_bgr = cv2.cvtColor(norm_uint8, cv2.COLOR_GRAY2BGR)

        # Calculate centered rectangle coordinates
        frame_height, frame_width = display_frame_bgr.shape[:2]
        top_left = ((frame_width - rect_width) // 2, (frame_height - rect_height) // 2)
        bottom_right = (top_left[0] + rect_width, top_left[1] + rect_height)

        # Draw rectangle overlay (blue, thickness 2)
        cv2.rectangle(display_frame_bgr, top_left, bottom_right, (255, 0, 0), 2)

        cv2.putText(display_frame_bgr, instructions[0], (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(display_frame_bgr, instructions[1], (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        active_text = "Active: " + ", ".join(f"{n}:{active_noises[n]}" for n in active_noises)
        cv2.putText(display_frame_bgr, active_text, (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(display_frame_bgr, classification_result, (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow(window_name, display_frame_bgr)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('+'):
            for noise in list(active_noises):
                if noise in noise_to_defaults:
                    defaults = noise_to_defaults[noise]
                    try:
                        idx = defaults.index(active_noises[noise])
                    except ValueError:
                        idx = 0
                    next_idx = (idx + 1) % len(defaults)
                    active_noises[noise] = defaults[next_idx]
            continue

        if key == ord('d'):  # classify only the rectangle region
            x1, y1 = top_left
            x2, y2 = bottom_right
            # Extract region from last_frame and classify it
            region = last_frame[y1:y2, x1:x2]
            region = cv2.resize(region, (617, 466))
            print("Region shape:", region.shape)
            result = my_classification_function(region)
            classification_result = "USB Type: " + result
            continue

        if key in {ord('s'), ord('p'), ord('a'), ord('f'), ord('c'), ord('b')}:
            mapping = {ord('s'): 'salt', ord('p'): 'pepper', ord('a'): 'salt_pepper',
                       ord('f'): 'sine', ord('c'): 'contrast', ord('b'): 'blur'}
            noise = mapping[key]
            if noise in noise_to_defaults:
                active_noises[noise] = noise_to_defaults[noise][0]
            else:
                active_noises[noise] = None
        elif key == ord('n'):
            active_noises.clear()
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def read_hog_dict_real_time(csv_path):
    hog_dict = {}
    with open(csv_path, 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header
        # Assume features start at column index 4
        for row in reader:
            class_label = row[0]
            filename = row[1]
            n_rows = int(row[2])
            n_cols = int(row[3])
            # Convert all hog feature values (columns 4 onward) to floats
            hog_vector = np.array([float(val) for val in row[4:]], dtype=np.float32)
            record = {
                "filename": filename,
                "rows": n_rows,
                "cols": n_cols,
                "hog_vector": hog_vector
            }
            if class_label not in hog_dict:
                hog_dict[class_label] = []
            hog_dict[class_label].append(record)
    return hog_dict



if __name__ == "__main__":
    cfg= Config.load()
    csv_path=os.path.join('data', 'hog_features.csv')
    hog_dict= read_hog_dict_real_time(csv_path)
    main(hog_dict)
