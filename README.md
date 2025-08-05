# Fingerprint Blood Group Prediction

This project predicts a person's blood group from fingerprint images using deep learning. It leverages a trained neural network model to classify fingerprint images into one of the major blood groups (A+, A-, B+, B-, AB+, AB-, O+, O-).

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model](#model)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results & Visualizations](#results--visualizations)
- [License](#license)

## Overview

The goal of this project is to explore the possibility of predicting blood groups from fingerprint patterns using machine learning. The project includes:

- A dataset of fingerprint images labeled by blood group.
- A deep learning model (Keras/TensorFlow) for classification.
- Scripts for training, prediction, and visualization.

## Dataset

The dataset is organized by blood group, with each folder containing fingerprint images in BMP format:

```
dataset_blood_group/
  ├── A+/
  ├── A-/
  ├── B+/
  ├── B-/
  ├── AB+/
  ├── AB-/
  ├── O+/
  └── O-/
```

Each subfolder contains fingerprint images for the respective blood group.

## Model

- The model architecture is defined and trained using Keras.
- The trained model is saved as `blood_group_model.h5`.
- Model architecture and training history visualizations are provided.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/Fingerprint-BloodGroup-Prediction.git
   cd Fingerprint-BloodGroup-Prediction-main
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **(Optional) Extract the dataset if not already extracted:**
   - The dataset is provided in both `.zip` and `.rar` formats.

## Usage

### Training

To train the model (if you want to retrain):

```bash
python main.py
```

### Prediction

To predict the blood group of a fingerprint image:

```bash
python predict.py --image path_to_image.BMP
```

Replace `path_to_image.BMP` with the path to your fingerprint image.

### Visualization

- `visualization.py` contains code for visualizing data distribution, training history, and more.
- Pre-generated visualizations are available as PNG files in the project root.

## Project Structure

```
.
├── blood_group_model.h5           # Trained model
├── dataset_blood_group/           # Fingerprint image dataset
├── main.py                        # Model training script
├── predict.py                     # Prediction script
├── visualization.py               # Visualization utilities
├── requirements.txt               # Python dependencies
├── *.png                          # Visualizations and results
└── README.md                      # Project documentation
```

## Results & Visualizations

- **Model Architecture:** `model_architecture.png`
- **Training History:** `training_history.png`
- **Data Distribution:** `data_distribution.png`, `dataset_distribution_detailed.png`
- **Sample Images:** `sample_images.png`
- **Prediction Example:** `prediction_result.png`
- **Image Quality Analysis:** `image_quality_analysis.png`

## License

This project is for academic and research purposes. Please check individual files for license information.

---

**Note:** This project is a proof of concept and should not be used for medical or forensic purposes without further validation.