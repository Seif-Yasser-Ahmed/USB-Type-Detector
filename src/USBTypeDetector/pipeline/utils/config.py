from easydict import EasyDict as edict
import numpy as np
__C = edict()
cfg = __C


__C.USB_Detector = edict()

__C.USB_Detector.data_dir= "../data/"

# params.yaml


__C.USB_Detector.classes_dir= "../data/classes/"
__C.USB_Detector.classes_greyscale_dir= "../data/classes_greyscale/"
__C.USB_Detector.classes_greyscale_augmented_dir= "../data/classes_greyscale_augmented/"
__C.USB_Detector.classes_preprocessed_dir= "../data/classes_preprocessed/"
__C.USB_Detector.output_dir= "../data/output/"

# diff_threshold: 50
# min_fraction: 0.01
# kernel_size_salt_pepper: 3

# brightness_block_size: 16

# brightness_threshold: 30

# delta_hsv_fix: 30

# fix_gamma_low_contrast: 0.7
# fix_gamma_high_contrast: 0.2


# augmentations:
#   flips:
#     - horizontal_flip
#     - vertical_flip
#     - both_flip

#   rotations:
#     - rotate_45
#     - rotate_90
#     - rotate_135
#     - rotate_225
#     - rotate_270
#     - rotate_315

#   crops:
#     - crop_50_percent
#     - crop_25_percent
#     - crop_10_percent

#   blurs:
#     - blur_5x5
#     - blur_15x15
#     - blur_25x25

#   salt_peppers:
#     - salt_only
#     - pepper_only
#     - salt_and_pepper

#   contrasts:
#     - contrast_0.1
#     - contrast_0.5
#     - contrast_0.75
#     - contrast_1
#     - contrast_1.5
#     - contrast_2
#     - contrast_2.5
#     - contrast_3

#   brightnesses:
#     - brightness_-100
#     - brightness_-50
#     - brightness_-30
#     - brightness_-20
#     - brightness_-10
#     - brightness_0
#     - brightness_10
#     - brightness_20
#     - brightness_30
#     - brightness_50
#     - brightness_100
#     - brightness_150
  
#   overlays:
#     - overlay_0.1
#     - overlay_0.2
#     - overlay_0.3
#     - overlay_0.4
#     - overlay_0.5
#     - overlay_0.6
#     - overlay_0.7
#     - overlay_0.8
#     - overlay_0.9
