# RSNA Breast Cancer Detection

This repository contains my solution to the RSNA 2024 Breast Cancer Detection competition. The goal of the competition is to detect and classify breast cancer from mammography images. I have implemented state-of-the-art models and preprocessing techniques to achieve high accuracy in breast cancer identification from this challenging dataset.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Installation](#installation)
- [Usage](#usage)
- [Contributions](#contributions)
- [License](#license)

---

## Overview

The RSNA 2024 Breast Cancer Detection competition focuses on accurately identifying breast cancer in mammography images. This repository demonstrates how to leverage deep learning to build robust classification models, utilizing advanced techniques such as data augmentation, transfer learning, and model fine-tuning.

---

## Dataset

The dataset consists of mammography images provided by the RSNA, divided into training and validation sets. Each image is categorized based on the presence or absence of breast cancer. Patient-specific subfolders are used for organizing the images, and TensorFlow records are generated for efficient data processing.

- **Source**: [RSNA Mammography Dataset](https://www.kaggle.com/competitions/rsna-breast-cancer-detection)

---

## Preprocessing

Key preprocessing steps applied to the dataset include:
- **Image Windowing**: Applied to enhance contrast and focus on regions of interest.
- **Cropping and Resizing**: Performed to standardize image sizes for model input.
- **Normalization and Augmentation**: Implemented using Albumentations to handle class imbalance and improve model generalization.

---

## Model Architecture

The solution implements multiple architectures to address breast cancer detection. The primary models used are:

- **ConvNeXt-Tiny**: A high-performance convolutional network used for transfer learning.
- **EfficientNet**: Optimized for both speed and accuracy in handling large-scale mammography images.

### Custom Preprocessing and Layers
- Custom image preprocessing is handled before feeding data into the neural networks.
- Residual and convolutional layers are added to fine-tune the model for mammography data.

---

## Training

The model is trained using the following configurations:
- **Loss Function**: Binary Cross-Entropy and Binary Focal Cross-Entropy.
- **Optimizers**: Adam and AdamW for optimization.
- **Augmentations**: Applied using Albumentations to increase data diversity and robustness.

---

## Evaluation

The model's performance is evaluated using key metrics such as:
- **F1 Score**: Focused on maximizing the balance between precision and recall.
- **ROC-AUC Score**: To assess the model's ability to distinguish between cancerous and non-cancerous images.
- **Accuracy**: For overall performance evaluation.

---

## Installation

### Requirements

- Python 3.8+
- TensorFlow
- Keras
- Albumentations
- Other dependencies can be found in the `requirements.txt`.

### Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/rsna-breast-cancer-detection.git
cd rsna-breast-cancer-detection
