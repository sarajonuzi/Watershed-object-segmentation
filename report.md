# Watershed-Based Object Segmentation and Counting Using Classical Image Processing

**Student:** Sara Jonuzanj  
**Student ID:** C210302801  

## 1. Introduction

Object segmentation and counting is an important task in image processing. It is used in applications such as counting coins, seeds, pills, cells, and other small objects. A common difficulty appears when objects touch or overlap. Simple thresholding can detect the foreground, but it often merges touching objects into one connected region. This project solves that problem using marker-controlled watershed segmentation.

## 2. Aim of the Project

The aim of this project is to build a classical image processing system that can:

- segment objects from the background,
- separate touching or overlapping objects,
- count the detected objects,
- measure object areas,
- save visual results for analysis.

## 3. Methodology

The proposed method follows a classical image processing pipeline.

### 3.1 Preprocessing

The input image is converted to grayscale. Gaussian filtering or median filtering is used to reduce noise and make the image easier to threshold. Noise reduction is important because small bright or dark pixels can create false objects.

### 3.2 Thresholding

After preprocessing, thresholding is applied to obtain a binary foreground mask. Otsu thresholding is used when the object and background intensities are clearly separable. Adaptive thresholding can be used when lighting is uneven.

### 3.3 Morphological Cleaning

Morphological opening removes small noise, while closing fills small gaps in object regions. Hole filling and small component removal improve the binary mask before watershed segmentation.

### 3.4 Distance Transform

The distance transform computes the distance of each foreground pixel from the nearest background pixel. Peaks in the distance map usually correspond to object centers. These peaks are used to generate foreground markers.

### 3.5 Marker-Controlled Watershed

The watershed algorithm treats the image as a topographic surface. Markers guide the algorithm so that touching objects are separated into different regions. This reduces over-segmentation compared with an uncontrolled watershed.

### 3.6 Connected Component Analysis

After watershed segmentation, each labeled region is treated as one detected object. Connected component analysis is used to count the objects and calculate their areas.

## 4. Implementation

The project is implemented in Python using OpenCV and NumPy. The main script is:

```text
src/watershed_counter.py
```

It accepts either a single image or a folder of images. For every image, it saves:

- original image,
- binary mask,
- distance transform visualization,
- watershed result with boundaries,
- final labeled result,
- summary CSV file with object counts and areas.

## 5. Evaluation

The system can be evaluated using count error:

```text
count_error = abs(predicted_count - true_count)
```

and count accuracy:

```text
count_accuracy = 1 - count_error / true_count
```

If ground-truth segmentation masks are available, IoU and Dice coefficient can also be used:

```text
IoU = intersection / union
Dice = 2 * intersection / (predicted_area + ground_truth_area)
```

## 6. Expected Results

The expected result is that isolated objects are segmented correctly and touching objects are separated by watershed boundaries. The result quality depends on image contrast, object overlap, noise level, and parameter selection.

## 7. Conclusion

This project demonstrates how classical image processing can be used for object segmentation and counting without deep learning. The marker-controlled watershed method is especially useful when objects are touching, because it uses object-center markers from the distance transform to split merged regions.

