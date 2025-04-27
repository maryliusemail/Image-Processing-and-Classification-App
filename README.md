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


---

### ImageProcessingTemplate Class
Provides a library of stateless, computationally efficient image transformations:

- **Negate**: Invert pixel intensities (`255 - intensity`) across all color channels.

_Example:_  
![ezgif com-animated-gif-maker (6)](https://github.com/user-attachments/assets/2f9b3532-4384-4976-bd6e-47c04c745429)



- **Grayscale**: Average RGB channels per pixel using floor division.

_Example:_  
![ezgif com-crop](https://github.com/user-attachments/assets/5cf32b05-fbe5-4b72-8504-5dd6a4ae32c4)


- **Rotate 180°**: Flip the image along both horizontal and vertical axes.

_Example:_  
![ezgif com-animated-gif-maker (5)](https://github.com/user-attachments/assets/623618e6-1b3f-4093-b6a7-adef07021e63)



- **Adjust Brightness**: Add/subtract uniform intensity with clipping at `[0, 255]`.

_Example:_  
![ezgif com-resize](https://github.com/user-attachments/assets/e23acb5a-3278-45bd-9e39-8379ad4c433b)


- **Blur**: Smooth image using local neighborhood averaging.

_Example:_  
![ezgif com-animated-gif-maker (3)](https://github.com/user-attachments/assets/5f060c91-2a51-448d-9403-6da417032321)



---

### StandardImageProcessing Class
Extends `ImageProcessingTemplate` with minor usage tracking:
- Inherits all transformation operations.
- Tracks the number of processing operations performed.
- Provides a coupon system to allow free operations temporarily.


---

### PremiumImageProcessing Class
Extends the base functionality with additional advanced image manipulation operations:

- **Tile**: Repeats an image to fill larger dimensions using modular indexing.

_Example:_  
![ezgif com-animated-gif-maker (7)](https://github.com/user-attachments/assets/3e5ad8d7-cd87-453d-80e8-eaab2044a559)


- **Sticker**: Overlays a smaller image onto a background at a specified (x, y) coordinate.

_Example:_  
![ezgif com-animated-gif-maker (1)](https://github.com/user-attachments/assets/b1c271e4-f3b9-468d-b1cb-a9fb19c879a3)


- **Edge Highlight**: Applies a 3x3 Laplacian convolution filter to detect and highlight edges.

_Example:_  
![ezgif com-animated-gif-maker](https://github.com/user-attachments/assets/5003de94-ac7d-4caa-adf9-d3d1dc61be02)

---

### ImageKNNClassifier Class
Implements a basic instance-based learning model for image classification based on nearest neighbor search.

# **fit(data)**:  
  Stores labeled training data into the classifier for future use.  
  - Input: List of `(image, label)` tuples where each image is an `RGBImage` object and label is a string.
  - Behavior: Saves the training dataset internally.  
  - Exception Handling: Raises `ValueError` if the number of training samples is fewer than `k_neighbors`.


---

# **distance(image1, image2)**:  
 Measures how similar two `RGBImage` instances are by calculating the **Euclidean distance** between them. 

_Image 1:_  
![steve](https://github.com/user-attachments/assets/adafec77-93c3-47c0-a937-837a1d8b9503)

_Image 2:_  
![knn_test_img](https://github.com/user-attachments/assets/64814854-c958-4145-ab78-c28fd05bbc79)


```python
img1 = img_read_helper('img/steve.png')
img2 = img_read_helper('img/knn_test_img.png')
knn = ImageKNNClassifier(3)
knn.distance(img1, img2)
```
- Output: 15946.312896716909
---

# **vote(candidates)**:  
  Determines the most common label from a list of candidate labels.  
  - Input: List of labels (strings) corresponding to the nearest neighbors.
  - Output: The label with the highest frequency among candidates.
  - Tie-breaking: In the event of a tie, any majority label may be returned (implementation-dependent).

_Example:_  
*(Insert diagram showing candidate neighbor labels and selected majority label)*

---

# **predict(image)**:  
  Predicts the label of a new `RGBImage` by finding its `k_neighbors` nearest training examples.
  - For a given test image:
    1. Computes distance to all stored training images.
    2. Sorts training data by ascending distance.
    3. Selects the top `k_neighbors`.
    4. Applies the `vote()` function to determine the most common label among neighbors.
  - Exception Handling: Raises `ValueError` if `fit()` has not been called prior to prediction.

_Example:_  
*(Insert diagram showing a new image surrounded by its 3/5/7 nearest neighbors and final predicted label)*

---


