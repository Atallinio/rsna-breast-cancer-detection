# RSNA Breast Cancer Detection

This repository contains my solutions to the [**RSNA Breast Cancer Detection**](https://www.kaggle.com/competitions/rsna-breast-cancer-detection/) competition. The goal of the competition is to detect and classify breast cancer from mammography images. In this repository, I have implemented two distinct approaches to tackle this challenge:

1. A **CVAE (Conditional Variational Autoencoder)** anomaly detection model designed to identify abnormal patterns in the mammograms.
2. A **Transfer Learning** classification model solution.

Both models leverage advanced preprocessing techniques and strive to achieve high accuracy in breast cancer identification from this complex dataset.


---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Generative Learning Model](#generative-learning-model-(cvae))
- [Transfer Learning Model](#transfer-learning-model)

---

## Overview

The RSNA Breast Cancer Detection competition focuses on accurately identifying breast cancer in mammography images. Early and accurate detection is essential for effective treatment, and automating this process could improve the quality and safety of patient care, while reducing costs and unnecessary medical procedures. By applying deep learning techniques, this competition contributes to advancing automated breast cancer detection.

---

## Dataset

### Key Features:

- **Images**: The dataset contains radiographic breast images, in DICOM format, for about 11,000 patients, with approximately 8,000 patients in the hidden test set. Each patient may have multiple images, usually around four.
- **Patient Metadata**: Accompanying metadata includes essential information such as:
  - **site_id**: ID code for the source hospital.
  - **patient_id**: Unique identifier for each patient.
  - **image_id**: Unique identifier for each image.
  - **laterality**: Indicates whether the image is of the left or right breast.
  - **view**: Orientation of the image, typically two views per breast for screening exams.
  - **age**: Patient's age in years.
  - **implant**: Indicates if the patient has breast implants.
  - **density**: A rating of breast tissue density (A to D), with D being the most dense, affecting diagnostic challenges.
  - **cancer**: Target variable indicating the presence of malignant cancer (provided only for training).
  - **biopsy**: Indicates if a follow-up biopsy was performed.
  - **invasive**: Specifies if the diagnosed cancer was invasive.
  - **BIRADS**: Assessment category for follow-up necessity, ranging from 0 (needs follow-up) to 2 (normal).
  - **prediction_id**: ID for matching submission rows in the test set.
  - **difficult_negative_case**: Flag for unusually difficult cases (only provided in training).

- **Source**: [RSNA Mammography Dataset](https://www.kaggle.com/competitions/rsna-breast-cancer-detection/data)

---

## Preprocessing

For this project, I used preprocessing code from Paul Bacher's [MammographyPreprocessor](https://www.kaggle.com/code/paulbacher/custom-preprocessor-rsna-breast-cancer#About-resizing-parameter) class, with some modifications. The preprocessing steps aim to prepare the RSNA Breast Cancer Detection dataset for deep learning models. Below, I describe the original steps as well as the final format of the dataset I created.

### **Preprocessing Steps:**

- **Windowing**:  
  Improve the contrast of the mammography images, making them more clinically accurate for viewing.

- **Fix Photometric Interpretation**:  
  Ensure that all image backgrounds are set to zero, which standardizes the pixel intensity values across the dataset.

- **Rescale with Slope and Intercept**:  
  Even though this might not be strictly necessary, it helps to ensure image consistency by applying the correct scaling to pixel values.

- **Normalize Between 0 and 255**:  
  Convert pixel values to an 8-bit range, reducing the bit-depth of the images to grayscale for easier processing.

- **Flip the Breasts**:  
  To ensure consistency, all images are flipped so that the breasts are oriented in the same direction.

- **Crop the Background**:  
  Remove extra background regions to focus only on the breast tissue, which is the region of interest for cancer detection.

- **Resize**:  
  After cropping, resize the images to a standard size to fit the input dimensions of the model.

- **Save the Image**:  
  Save the preprocessed images, with the option of either PNG (default) or JPEG formats.


![Screenshot 2024-10-13 033521](https://github.com/user-attachments/assets/6f5c38a4-4f81-4a00-97c1-e14c1afd7afc)



### **Aspect Ratio Consideration for Resizing**

I created a dataset of **728x1456** PNG images, which adheres to a **2:1 aspect ratio**. This choice was based on the following observations:

- When cropped, a sample of 300 mammography images had a median aspect ratio of 2.1, making them naturally rectangular in shape.
- For computational purposes, resizing the images is essential. However, resizing to square dimensions (e.g., 1024x1024) causes an uneven loss of information, particularly compressing the vertical axis more than the horizontal axis.

---

## Generative Learning Model (CVAE)

I will frame the problem as an anomaly detection problem and will train a Convolutional Variational Autoencoder. 
The intuition behind the model is as follows:

- Train a Convolutional Variational Autoencoder (CVAE) on the non-cancer images only
- Calculate a threshold for the error
- A higher error should be expected when generating cancer images

### **The Model Architecture**:
- **Base Model**: Built upon a VAE from a TensorFlow tutorial² 
- **Encoder Architecture**: Implemented Pre-activation residual blocks in the encoder for improved feature extraction
- **Decoder Architecture (5 Layers)**: Utilizes Transposed Convolutions to reconstruct the image from the latent space
- **Residual Blocks**: allow for deeper architectures while mitigating the vanishing gradient problem



![autoencoder_architecture](https://github.com/user-attachments/assets/7a424bf0-11c3-4a96-897b-486b5e10675e)




---

## Training 

- **Loss Function**: A Variational Autoencoder uses a special loss function consisting of two terms: **Kullback-Leibler divergence** & **Reconstruction Loss**

![image](https://github.com/user-attachments/assets/fae1ec1f-2659-456c-b11c-e0337c97b585)


---
## Transfer Learning Model

The Idea is to use an EfficientNet model as a backbone to classify the existence of cancer in mammography images

![image](https://github.com/user-attachments/assets/f49d8c12-a0d3-44cd-99ea-3fcc3f263c20)

---

## Training

The model is trained and evaluated using the following configurations:
- **Loss Function**: Vanilla Binary Cross-Entropy.
- **Optimizer**: Adam optimizer.

Augmentation Applied to Train data
- Random brightness
- Random contrast
- Random cropping


The model's performance is evaluated using key metrics such as:
- **Probabilistic F1 Score**: Focused on maximizing the balance between precision and recall.
- **AUC Score**: To assess the model's ability to distinguish between cancerous and non-cancerous images.
- **Binary Accuracy**: For overall performance evaluation.

---

### Clone the repository

```bash
git clone https://github.com/Atallinio/rsna-breast-cancer-detection.git
cd rsna-breast-cancer-detection
