import os
import cv2
import numpy as np

class ImageAugmentation:
    """
    A class for performing various image augmentation techniques such as flipping, rotation,
    cropping, blurring, brightness/contrast adjustment, noise addition, and wave pattern overlays.

    Attributes:
        classes_dir (list): List of directories containing class-wise images.
        save_dir (str): Output directory where augmented images will be saved.
    """
    def __init__(self, data_dir, save_dir):
        """
        Initialize the ImageAugmentation instance.

        Args:
            data_dir (list): List of class directories with images to augment.
            save_dir (str): Directory to save the augmented images.
        """
        self.classes_dir = data_dir
        self.save_dir = save_dir


    def augment_images_flipping(self, **kwargs):
        """
        Augment images by flipping them horizontally, vertically, or both.

        Keyword Args:
            horizontal_flip (bool): If True, perform horizontal flip.
            vertical_flip (bool): If True, perform vertical flip.
            (default): If neither is specified, perform both flips.
        """
        key = None
        for images in self.classes_dir:
            class_base_name = os.path.basename(images)
            image_files = os.listdir(images)
            for image in image_files:
                img_path = os.path.join(images, image)
                img = cv2.imread(img_path)

                if kwargs.get('horizontal_flip', False):
                    flipped_img = cv2.flip(img, 1)
                    key = 'horizontal_flip'
                elif kwargs.get('vertical_flip', False):
                    flipped_img = cv2.flip(img, 0)
                    key = 'vertical_flip'
                else:
                    flipped_img = cv2.flip(img, -1)
                    key = 'both_flip'
              
                flipped_img_path = os.path.join(self.save_dir, class_base_name, f"flipped_{key}_{image}")
                os.makedirs(os.path.dirname(flipped_img_path), exist_ok=True)
                cv2.imwrite(flipped_img_path, flipped_img)

    def augment_images_rotation(self, **kwargs):
        """
        Augment images by rotating them by specified angles.

        Keyword Args:
            rotate_45 (bool)
            rotate_90 (bool)
            rotate_135 (bool)
            rotate_225 (bool)
            rotate_270 (bool)
            rotate_315 (bool)
        """
        key = None
        for images in self.classes_dir:
            class_base_name = os.path.basename(images)
            image_files = os.listdir(images)
            for image in image_files:
                img_path = os.path.join(images, image)
                img = cv2.imread(img_path)

                if kwargs.get('rotate_45', False):
                    M = cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), 45, 1)
                    rotated_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
                    key="rotate_45"
                elif kwargs.get('rotate_90', False):
                    M = cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), 90, 1)
                    rotated_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
                    key="rotate_90"
                elif kwargs.get('rotate_135', False):
                    M = cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), 135, 1)
                    rotated_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
                    key="rotate_135"
                elif kwargs.get('rotate_225', False):
                    M = cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), 225, 1)
                    rotated_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
                    key="rotate_225"
                elif kwargs.get('rotate_270', False):
                    M = cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), 270, 1)
                    rotated_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
                    key="rotate_270"
                elif kwargs.get('rotate_315', False):
                    M = cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), 315, 1)
                    rotated_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
                    key="rotate_315"

                rotated_img_path = os.path.join(self.save_dir, class_base_name, f"rotated_{key}_{image}")
                os.makedirs(os.path.dirname(rotated_img_path), exist_ok=True)
                cv2.imwrite(rotated_img_path, rotated_img)

    def augment_images_cropping(self, **kwargs):
        """
        Augment images by cropping the center of the image to various sizes.

        Keyword Args:
            crop_50_percent (bool): Crop central 50% of the image.
            crop_25_percent (bool): Crop central 25% of the image.
            crop_10_percent (bool): Crop central 10% of the image.
        """
        key = None
      
        for images in self.classes_dir:
            class_base_name = os.path.basename(images)
            image_files = os.listdir(images)
            for image in image_files:
                img_path = os.path.join(images, image)
                img = cv2.imread(img_path)

                if kwargs.get('crop_50_percent', False):
                    h, w = img.shape[:2]
                    cropped_img = img[int(h*0.25):int(h*0.75), int(w*0.25):int(w*0.75)]
                    key = 'crop_50_percent'
                elif kwargs.get('crop_25_percent', False):
                    h, w = img.shape[:2]
                    cropped_img = img[int(h*0.375):int(h*0.625), int(w*0.375):int(w*0.625)]
                    key = 'crop_25_percent'
                elif kwargs.get('crop_10_percent', False):
                    h, w = img.shape[:2]
                    cropped_img = img[int(h*0.45):int(h*0.55), int(w*0.45):int(w*0.55)]
                    key = 'crop_10_percent'

                cropped_img_path = os.path.join(self.save_dir, class_base_name, f"cropped_{key}_{image}")
                os.makedirs(os.path.dirname(cropped_img_path), exist_ok=True)
                cv2.imwrite(cropped_img_path, cropped_img)

    def augment_images_blurring(self, **kwargs):
        """
        Apply Gaussian blur to images with different kernel sizes.

        Keyword Args:
            blur_5x5 (bool): Apply blur with 5x5 kernel.
            blur_15x15 (bool): Apply blur with 15x15 kernel.
            blur_25x25 (bool): Apply blur with 25x25 kernel.
        """
        key = None
        
        for images in self.classes_dir:
            class_base_name = os.path.basename(images)
            image_files = os.listdir(images)
            for image in image_files:
                img_path = os.path.join(images, image)
                img = cv2.imread(img_path)

                if kwargs.get('blur_5x5', False):
                    blurred_img = cv2.GaussianBlur(img, (5, 5), 0)
                    key = 'blur_5x5'
      
                elif kwargs.get('blur_15x15', False):
                    blurred_img = cv2.GaussianBlur(img, (15, 15), 0)
                    key = 'blur_15x15'
      
                elif kwargs.get('blur_25x25', False):
                    blurred_img = cv2.GaussianBlur(img, (25, 25), 0)
                    key = 'blur_25x25'

                blurred_img_path = os.path.join(self.save_dir, class_base_name, f"blurred_{key}_{image}")
                os.makedirs(os.path.dirname(blurred_img_path), exist_ok=True)
                cv2.imwrite(blurred_img_path, blurred_img)
    
    def augment_images_salt_pepper(self, **kwargs):
        """
        Add salt, pepper, or both types of noise to the images.

        Keyword Args:
            salt_only (bool): Apply only white pixels as noise.
            pepper_only (bool): Apply only black pixels as noise.
            salt_and_pepper (bool): Apply both salt and pepper noise.
        
        Notes:
            The amount of noise and ratio is fixed at 4% total with 50-50 salt-pepper split.
        """
        key = None
        for images in self.classes_dir:
            class_base_name = os.path.basename(images)
            image_files = os.listdir(images)
            for image in image_files:
                img_path = os.path.join(images, image)
                img = cv2.imread(img_path)

                if kwargs.get('salt_only', False):
                    s_vs_p = 0.5
                    amount = 0.04
                    out = np.copy(img)
                    num_salt = np.ceil(amount * img.size * s_vs_p)
                    coords = [np.random.randint(0, i, int(num_salt)) for i in img.shape[:2]]
                    out[tuple(coords)] = 255
                    key = 'salt_only'

                elif kwargs.get('pepper_only', False):
                    s_vs_p = 0.5
                    amount = 0.04
                    out = np.copy(img)
                    num_pepper = np.ceil(amount * img.size * (1. - s_vs_p))
                    coords = [np.random.randint(0, i, int(num_pepper)) for i in img.shape[:2]]
                    out[tuple(coords)] = 0
                    key = 'pepper_only'

                elif kwargs.get('salt_and_pepper', False):
                    s_vs_p = 0.5
                    amount = 0.04
                    out = np.copy(img)
                    num_salt = np.ceil(amount * img.size * s_vs_p)
                    coords = [np.random.randint(0, i, int(num_salt)) for i in img.shape[:2]]
                    out[tuple(coords)] = 255
                    
                    
                    num_pepper = np.ceil(amount * img.size * (1. - s_vs_p))
                    coords = [np.random.randint(0, i, int(num_pepper)) for i in img.shape[:2]]
                    out[tuple(coords)] = 0
                    key = 'salt_and_pepper'

                salt_pepper_img_path = os.path.join(self.save_dir, class_base_name, f"salt_pepper_{key}_{image}")
                os.makedirs(os.path.dirname(salt_pepper_img_path), exist_ok=True)
                cv2.imwrite(salt_pepper_img_path, out)


    def augment_images_contrast(self, **kwargs):
        """
        Adjust image contrast by scaling pixel intensity values.

        Keyword Args:
            contrast_0.1 to contrast_3 (bool): Contrast scaling factors from 0.1 to 3.0
        """
        key = None
        for images in self.classes_dir:
            class_base_name = os.path.basename(images)
            image_files = os.listdir(images)
            for image in image_files:
                img_path = os.path.join(images, image)
                img = cv2.imread(img_path)

                if kwargs.get('contrast_0.1', False):
                    alpha = 0.1
                    beta = 0
                    contrast_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
                    key = 'contrast_0.1'
                elif kwargs.get('contrast_0.5', False):
                    alpha = 0.5
                    beta = 0
                    contrast_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
                    key = 'contrast_0.5'
                elif kwargs.get('contrast_0.75', False):
                    alpha = 0.75
                    beta = 0
                    contrast_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
                    key = 'contrast_0.75'
                elif kwargs.get('contrast_1', False):
                    alpha = 1.0
                    beta = 0
                    contrast_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
                    key = 'contrast_1'
                elif kwargs.get('contrast_1.5', False):
                    alpha = 1.5
                    beta = 0
                    contrast_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
                elif kwargs.get('contrast_2', False):
                    alpha = 2.0
                    beta = 0
                    contrast_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
                    key = 'contrast_2'
                elif kwargs.get('contrast_2.5', False):
                    alpha = 2.5
                    beta = 0
                    contrast_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
                    key = 'contrast_2.5'
                elif kwargs.get('contrast_3', False):
                    alpha = 3.0
                    beta = 0
                    contrast_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
                    key = 'contrast_3'

                contrast_img_path = os.path.join(self.save_dir, class_base_name, f"contrast_{key}_{image}")
                os.makedirs(os.path.dirname(contrast_img_path), exist_ok=True)
                cv2.imwrite(contrast_img_path, contrast_img)
                

    def augment_images_overlay(self, **kwargs):
        """
        Overlay a solid color layer (black or white) over the image with a given opacity.

        Keyword Args:
            mode (str): 'darken' or 'lighten'
            opacity (float): Opacity value between 0.0 and 1.0
        """
        mode = kwargs.get('mode', 'darken')
        opacity = kwargs.get('opacity', 0.7)

        if mode not in ('darken', 'lighten'):
            raise ValueError("mode must be 'darken' or 'lighten'")
        if not (0.0 <= opacity <= 1.0):
            raise ValueError("opacity must be between 0.0 and 1.0")

        for class_dir in self.classes_dir:
            class_name = os.path.basename(class_dir)
            out_class_dir = os.path.join(self.save_dir, class_name)
            os.makedirs(out_class_dir, exist_ok=True)

            for filename in os.listdir(class_dir):
                img_path = os.path.join(class_dir, filename)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                if mode == 'darken':
                    overlay = np.zeros_like(img)
                else:  # lighten
                    overlay = np.full_like(img, 255)

                out = cv2.addWeighted(img, 1.0 - opacity, overlay, opacity, 0)

                key = f"{mode}_{opacity:.2f}"
                out_name = f"{key}_{filename}"
                cv2.imwrite(os.path.join(out_class_dir, out_name), out)


    def augment_images_brightness(self, **kwargs):
        """
        Adjust image brightness by adding a constant to pixel values.

        Keyword Args:
            brightness_-100 to brightness_150 (bool): Brightness shifts from -100 to +150
        """
        key = None
        for images in self.classes_dir:
            class_base_name = os.path.basename(images)
            image_files = os.listdir(images)
            for image in image_files:
                img_path = os.path.join(images, image)
                img = cv2.imread(img_path)
                
                if kwargs.get('brightness_-100', False):
                    alpha = 1.0
                    beta = -100
                    brightness_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
                    key = 'brightness_-100'
                elif kwargs.get('brightness_-50', False):
                    alpha = 1.0
                    beta = -50
                    brightness_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
                    key = 'brightness_-50'
                elif kwargs.get('brightness_-30', False):
                    alpha = 1.0
                    beta = -30
                    brightness_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
                    key = 'brightness_-30'
                elif kwargs.get('brightness_-20', False):
                    alpha = 1.0
                    beta = -20
                    brightness_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
                    key = 'brightness_-20'
                elif kwargs.get('brightness_-10', False):
                    alpha = 1.0
                    beta = -10
                    brightness_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
                    key = 'brightness_-10'
                elif kwargs.get('brightness_0', False):
                    alpha = 1.0
                    beta = 0
                    brightness_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
                    key = 'brightness_0'
                elif kwargs.get('brightness_10', False):
                    alpha = 1.0
                    beta = 10
                    brightness_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
                    key = 'brightness_10'
                elif kwargs.get('brightness_20', False):
                    alpha = 1.0
                    beta = 20
                    brightness_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
                    key = 'brightness_20'
                elif kwargs.get('brightness_30', False):
                    alpha = 1.0
                    beta = 30
                    brightness_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
                    key = 'brightness_30'
                elif kwargs.get('brightness_50', False): 
                    alpha = 1.0
                    beta = 50
                    brightness_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
                    key = 'brightness_50'
                elif kwargs.get('brightness_100', False): 
                    alpha = 1.0
                    beta = 100
                    brightness_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
                    key = 'brightness_100'
                elif kwargs.get('brightness_150', False):
                    alpha = 1.0
                    beta = 150
                    brightness_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
                    key = 'brightness_150'

                brightness_img_path = os.path.join(self.save_dir, class_base_name, f"brightness_{key}_{image}")
                os.makedirs(os.path.dirname(brightness_img_path), exist_ok=True)
                cv2.imwrite(brightness_img_path, brightness_img)
                

    def augment_images_wave(self, n_cycles=5, angle_deg=45, opacity=0.2):
        """
        Apply a sine wave pattern overlay on grayscale version of the image.

        Args:
            n_cycles (int): Number of sine cycles across the diagonal of the image.
            angle_deg (float): Direction angle (in degrees) of the sine wave.
            opacity (float): Opacity of the sine pattern, from 0.0 to 1.0.
        """
        angle = np.deg2rad(angle_deg)
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)

        for images in self.classes_dir:
            class_base_name = os.path.basename(images)
            image_files = os.listdir(images)

            for image in image_files:
                if not image.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
                    continue

                img_path = os.path.join(images, image)
                img = cv2.imread(img_path)

                if img is None:
                    print(f"Warning: Couldn't read {img_path}")
                    continue

                # Convert to grayscale
                if len(img.shape) == 3:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                else:
                    gray = img.copy()

                rows, cols = gray.shape
                x = np.arange(cols)
                y = np.arange(rows)
                X, Y = np.meshgrid(x, y)

                # Frequency from wave count
                diag_length = np.sqrt(rows**2 + cols**2)
                frequency = n_cycles / diag_length

                # Generate sine wave in range [-1, 1]
                sine_wave = np.cos(2 * np.pi * frequency * (X * cos_angle + Y * sin_angle))

                # Convert to visible pattern: [0, 255], but keep only the "lines" (edges)
                sine_lines = ((1 - sine_wave) * 255 * opacity).astype(np.uint8)  # Invert so lines are brighter

                # Add faint sine lines to the original image
                modulated = cv2.add(gray, sine_lines)
                modulated = np.clip(modulated, 0, 255).astype(np.uint8)

                # Save result
                wave_img_path = os.path.join(
                    self.save_dir,
                    class_base_name,
                    f"wave_overlay_{n_cycles}cycles_angle{angle_deg}_opacity{opacity}_{image}"
                )
                os.makedirs(os.path.dirname(wave_img_path), exist_ok=True)
                cv2.imwrite(wave_img_path, modulated)
