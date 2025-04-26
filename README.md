# 🖼️ Image Processing and Classification App

This project implements a simple image processing application with object-oriented design principles (classes, inheritance, and exception handling). It processes RGB images, builds a customizable image editing system, and uses a K-Nearest Neighbors (KNN) classifier to predict image labels.

---

## 📦 Project Structure

- `project.py` — Core functionality (image processing classes and methods)
- `image_viewer.py` — Utility for viewing processed images
- `img/` — Folder with input and expected output images
- `knn_data/` — Sample labeled images for classification tasks

---

## ✨ Core Features

### RGBImage Class
Handles the basic structure and validation of RGB images:
- **Constructor** to initialize image data and dimensions
- **Getter/Setter methods** to retrieve or modify pixels
- **Deep copying** to safely duplicate images
- **Validation and exception handling** for input types and values

_Example: (Insert image showing raw RGB image structure)_

---

### ImageProcessingTemplate Class
Implements basic image processing operations:
- **Negate**: Invert colors to create a photo negative

  
  ![ezgif com-animated-gif-maker](https://github.com/user-attachments/assets/7af611b6-5211-4b28-ad58-7b62d42516a6)
---
- **Grayscale**: Convert images to grayscale by averaging RGB channels
- **Rotate 180°**: Rotate an image upside down
- **Get Average Brightness**: Calculate mean pixel brightness
- **Adjust Brightness**: Increase or decrease overall brightness
- **Blur**: Smooth images by averaging pixel neighborhoods



---

### StandardImageProcessing Class
A basic monetization version of the template:
- Inherits from `ImageProcessingTemplate`
- Adds **cost tracking** for each processing operation
- **Coupon system** to redeem free edits

_Example: (Insert image showing cost tracking in action)_

---

### PremiumImageProcessing Class
A premium app version with extra features:
- Fixed upfront **membership cost**
- **Tile**: Repeat an image to fill larger dimensions
- **Sticker**: Overlay an image onto another at specified coordinates
- **Edge Highlight**: Perform edge detection using convolution with a kernel

_Example: (Insert image showing tiling, stickers, and edge highlighting)_

---

### ImageKNNClassifier Class
Implements a simple K-Nearest Neighbors classifier for images:
- **Fit**: Load labeled training images
- **Distance**: Compute Euclidean distance between images
- **Vote**: Determine most common label among nearest neighbors
- **Predict**: Classify new images based on training data

_Example: (Insert diagram showing KNN neighbor selection and prediction)_

---

## 🚀 Running the Project

### Setup
Install required packages:

```bash
pip install numpy Pillow
