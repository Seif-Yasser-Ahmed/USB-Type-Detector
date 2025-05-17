# USB Type Detector

## Project Overview

The USB Type Detector is a modular computer vision pipeline designed to identify various USB connector types (Type-A, Type-B, Type-C, Micro-B, and Micro-USB) using classical image processing techniques. The system preprocesses input images, extracts relevant features (e.g., HOG descriptors, geometric properties), and classifies connectors in real-time with KNN-based or rule-based classifiers. Optional noise augmentations support robustness testing under different perturbations.

## Team Members

- [Seif Yasser](https://github.com/Seif-Yasser-Ahmed)
- [AbdulRahman Hesham](https://github.com/AHKSASE2002)
- [Ahmed Nezar](https://github.com/Ahmed-Nezar)
- [Kirollos Ehab](https://github.com/KirollosEMH)

## Table of Contents

1. [Project Overview](#project-overview)
2. [Team Members](#team-members)
3. [Directory Structure](#directory-structure)
4. [Project Features](#project-features)
5. [Package Information](#package-information)
6. [Installation](#installation)
7. [Usage and Inference](#usage-and-inference)
8. [Future Enhancements](#future-enhancements)
9. [License](#license)

## Directory Structure

```plaintext
USB-Type-Detector/
├── README.md              # Project documentation (this file)
├── LICENSE                # MIT License
├── run.py                 # Real-time camera script with noise controls
├── config/
│   └── params.yaml        # Global parameters and paths
├── data/
│   ├── README.md          # Data directory documentation
│   └── classes/           # Raw image classes for each USB type
│       ├── Micro-B/
│       ├── micro-usb/
│       ├── Type-A/
│       ├── Type-B/
│       └── Type-C/
├── notebooks/
│   └── main.ipynb         # Jupyter notebook for experimentation
└── src/
    ├── README.md          # Source folder overview
    ├── requirements.txt   # Python dependencies
    ├── setup.py           # Package installer
    └── USBTypeDetector/   # Core Python package
        ├── usage.py       # Public API entrypoint
        ├── pipeline/      # Pipeline modules
        │   ├── preprocessor/   # Noise detection, filtering, ROI extraction
        │   ├── extractor/      # Edge, morphology, HOG feature extraction
        │   ├── augmenter/      # Image augmentation utilities
        │   └── classifier/     # KNN and geometry-based classifiers
        └── utils/         # Configuration loader and helpers
```

## Project Features

* **Preprocessing**: Noise detection & removal (salt-and-pepper, blur, contrast), ROI extraction, adaptive binarization.
* **Feature Extraction**: Edge detection, morphological operations, HOG descriptor generation, corner analysis.
* **Classification**: KNN-based based on HOG vector similarity; placeholder geometry-based rule classifier.
* **Augmentation**: Salt & pepper, blur, contrast, brightness, overlay, rotation, cropping, and sine wave noise to test robustness.
* **Real-Time Inference**: `run.py` provides an interactive camera window with live noise control keys and on-demand classification.
* **Modular Design**: YAML-configurable parameters; abstraction layers for each pipeline stage.

## Package Information

* **Name**: `USBTypeDetector`
* **Version**: 0.6
* **Authors**: Seif Yasser, AbdulRahman Hesham, Ahmed Nezar, Kirollos Ehab
* **Dependencies**: See `src/requirements.txt` and `config/params.yaml` for full list.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Seif-Yasser-Ahmed/USB-Type-Detector.git
   cd USB-Type-Detector
   ```
2. (Optional) Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Windows
   ```
3. Install dependencies:

   ```bash
   pip install -r src/requirements.txt
   ```
4. Install the package:

   ```bash
   pip install ./src
   ```

## Usage and Inference

### Real-Time Camera Demo

Run the main script to launch a camera window with interactive noise controls and classification:

```bash
python run.py
```

* **Keys**:

  * `s`/`p`/`a`/`f`/`c`/`b`: Toggle salt, pepper, salt+pepper, sine, contrast, blur noise.
  * `+`: Next noise level preset.
  * `d`: Classify center ROI.
  * `n`: Clear all noise.
  * `q`: Quit.


## Future Enhancements

* Implement geometry-based classifier methods (`classify_by_aspect_ratio`, `classify_by_number_pins`, etc.).
* Improve ROI extraction with deep learning-based segmentation.
* Add support for female connector detection and orientation correction.
* Integrate a Streamlit/Gradio web interface for user-friendly demos.
* Extend augmentations with color jitter, perspective transforms, and lighting adjustments.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
