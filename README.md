# 🖼️ RGB Image Processing and K-Nearest Neighbors Classification System


This project builds a basic image processing system and a simple machine learning classifier from scratch.  
It allows you to apply common image transformations (like negation, grayscale, blur, and rotation) and classify images based on pixel similarity using the K-Nearest Neighbors (KNN) algorithm.



---

## 🚀 How to Run the Project

### 1. Download the Project Files
Download or clone the full project folder, which includes:

- `project.py` (core code)
- `image_viewer.py` (image viewer tool)
- `img/` folder with sample images
- `img/out/` folder (output results will be saved here)

Make sure the `out/` folder exists inside `img/` to store your processed images.

---

### 2. Install Required Python Packages

If you don't already have the required libraries installed, run:

```bash
pip install numpy Pillow
```

### 3. Run and Save Results

- Use the classes and functions in `project.py` to process images.
- Save any outputs to `img/out/` for easier viewing.

**Example to Test (on Terminal):**

```python
img_proc = ImageProcessingTemplate()
img = img_read_helper('img/steve.png')
img_blur = img_proc.blur(img)
img_save_helper('img/out/test_image_32x32_blur.png', img_blur)
```

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

## 🧠 Image KNN Classifier Overview

K-Nearest Neighbors (KNN) is a classic machine learning algorithm commonly used for **classification** tasks.  
It works under the principle that similar data points exist close together in feature space.

In this project, we apply KNN to classify images based on their raw pixel values.

---

### 🌞🌙 Real-World Example: Classifying Day vs Night Images

This project follows a typical KNN workflow.  
Imagine building a model to determine whether an image shows **daytime** or **nighttime**:

1. **Collect a dataset**:
   - Images labeled `"daytime"`
   - Images labeled `"nighttime"`

2. **Classify a new image**:
   - Measure how **similar** the new image is to the labeled examples.
   - Find the **k** closest images (nearest neighbors).
   - **Vote** among them to predict the label.

This approach generalizes to **any kind of image classification** based on visual similarity.


## 🛠️ How the KNN Classifier Works in This Project

### **Step 1: Fitting the Model — `fit(data)`**

- **Purpose**: Save labeled training data (images and labels) for future use.
- **Input**: List of `(image, label)` pairs.


---

### **Step 2: Measuring Distance — `distance(image1, image2)`**

- **Purpose**: Quantify how visually similar two images are.
- **Method**:
  - Flatten each 3D RGB image matrix into a 1D list.
  - Compute the **Euclidean distance** between corresponding pixel intensities:
  
    $$d(a, b) = \sqrt{(a_1-b_1)^2 + (a_2-b_2)^2 + \dots + (a_n-b_n)^2}$$

  
  - A **smaller distance** indicates higher similarity.

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

### **Step 3: Voting — `vote(candidates)`**

- **Purpose**: Choose the most common label among the nearest neighbors.
- **Input**: A list of candidate labels (strings).
- **Behavior**:
  - Returns the most frequent label.
  - If there’s a tie, any of the majority labels may be selected.


---

### **Step 4: Predicting — `predict(image)`**

- **Purpose**: Predict the label of a new image using the KNN method.
- **Workflow**:
  1. Compute distances to all stored training images.
  2. Sort the images by ascending distance.
  3. Select the top `k_neighbors`.
  4. Apply `vote()` to predict the label based on the neighbors' labels.

- **Training Data**:
  - Training images are loaded from the `knn_data/` folder.
  - Each image is labeled (e.g., `"daytime"`, `"nighttime"`) and used for nearest neighbor comparisons.


---



_Example:_  
![knn_test_img](https://github.com/user-attachments/assets/1b3183e6-5e11-48db-9c0d-c0e6190404f2)

```python
knn_tests('img/knn_test_img.png')
```
- Output: nighttime

- ✅ **Result**: The model correctly predicted the image as nighttime!



---


