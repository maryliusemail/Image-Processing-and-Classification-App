# 🖼️ RGB Image Processing and K-Nearest Neighbors Classification System

This project implements an extensible image processing framework using object-oriented programming (OOP), and a basic machine learning image classification system. It focuses on handling low-level image representations, applying transformation pipelines, managing images through inheritance, and using a distance-based supervised learning algorithm for classification.

---

## 📦 Repository Structure

- `project.py` — Core source code (image processing and KNN classifier classes)
- `image_viewer.py` — Visualization utility to preview processed images
- `img/` — Sample input and expected output images
- `knn_data/` — Training datasets for KNN model experiments

---

## ✨ Core Modules and Functionalities

### RGBImage Class
Defines the fundamental RGB image object using nested lists:
- Initializes 3D matrices representing pixels in `(row, column, [R,G,B])` format.
- Enforces strict type and shape validation with runtime exception handling.
- Supports controlled access to image data through getters/setters.
- Implements deep copying to maintain image immutability across transformations.

_Example:_  
*(Insert visual showing RGB matrix structure and pixel values)*

---

### ImageProcessingTemplate Class
Provides a library of stateless, computationally efficient image transformations:

- **Negate**: Invert pixel intensities (`255 - intensity`) across all color channels.

_Example:_  

![ezgif com-animated-gif-maker](https://github.com/user-attachments/assets/5003de94-ac7d-4caa-adf9-d3d1dc61be02)

- **Grayscale**: Average RGB channels per pixel using floor division.

_Example:_  
![ezgif com-crop](https://github.com/user-attachments/assets/5cf32b05-fbe5-4b72-8504-5dd6a4ae32c4)


- **Rotate 180°**: Flip the image along both horizontal and vertical axes.

_Example:_  
![ezgif com-animated-gif-maker (4)](https://github.com/user-attachments/assets/c6058491-32cb-432f-81a3-cc54c740b2a4)


- **Get Average Brightness**: Calculate mean brightness value across all pixels.

_Example:_  
*(Insert brightness heatmap or show brightness comparison value)*

- **Adjust Brightness**: Add/subtract uniform intensity with clipping at `[0, 255]`.

_Example:_  
*(Insert side-by-side images: darkened version, brightened version)*

- **Blur**: Smooth image using local neighborhood averaging.

_Example:_  
![ezgif com-animated-gif-maker (3)](https://github.com/user-attachments/assets/5f060c91-2a51-448d-9403-6da417032321)



---

### StandardImageProcessing Class
Extends `ImageProcessingTemplate` with minor usage tracking:
- Inherits all transformation operations.
- Tracks the number of processing operations performed.
- Provides a coupon system to allow free operations temporarily.

Uses method overriding and `super()` to extend base functionality cleanly.

_Example:_  
*(Insert visual showing sequence of transformations applied on an image)*

---

### PremiumImageProcessing Class
Extends the base functionality with additional advanced image manipulation operations:

- **Tile**: Repeats an image to fill larger dimensions using modular indexing.

_Example:_  
*(Insert image showing a smaller image tiled across a larger canvas)*

- **Sticker**: Overlays a smaller image onto a background at a specified (x, y) coordinate.

_Example:_  
![ezgif com-animated-gif-maker (1)](https://github.com/user-attachments/assets/b1c271e4-f3b9-468d-b1cb-a9fb19c879a3)


- **Edge Highlight**: Applies a 3x3 Laplacian convolution filter to detect and highlight edges.

_Example:_  
*(Insert image showing edge-highlighted output using Laplacian filter)*

---

### ImageKNNClassifier Class
Implements a simple instance-based supervised learning algorithm for image classification:
- **fit()**: Stores (image, label) training data pairs.
- **distance()**: Computes flattened pixel-wise Euclidean distance between two images.
- **vote()**: Determines the most common label among nearest neighbors.
- **predict()**: Predicts the label for new images based on `k` nearest training examples.

Uses purely manual distance calculation and list operations for classification logic.

_Example:_  
*(Insert diagram showing a test image and its k-nearest neighbors with majority voting)*

---

## 🚀 How to Run

### Setup
Install required packages for testing utilities:

```bash
pip install numpy Pillow
