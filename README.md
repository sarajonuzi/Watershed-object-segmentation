# Watershed-Based Object Segmentation and Counting Using Classical Image Processing

Student: Sara Jonuzanj  
Student ID: C210302801

## Project Summary

This project segments and counts overlapping objects such as coins, seeds, pills, or cells using a classical image processing pipeline. It uses preprocessing, thresholding, distance transform, marker-controlled watershed segmentation, morphology, and connected component analysis.

## Pipeline

1. Load input image.
2. Convert to grayscale.
3. Apply Gaussian or median filtering.
4. Segment foreground using Otsu or adaptive thresholding.
5. Clean the binary mask with morphological opening and closing.
6. Compute distance transform.
7. Detect sure foreground markers.
8. Use marker-controlled watershed to separate touching objects.
9. Remove small artifacts.
10. Count objects and measure region areas.

## Folder Structure

```text
.
├── data/
│   ├── input/              # Put your real images here
│   └── output/             # Results are saved here
├── src/
│   ├── watershed_counter.py
│   └── generate_synthetic_dataset.py
├── report.md
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Test With Synthetic Images

Generate test images:

```bash
python src/generate_synthetic_dataset.py
```

Run segmentation and counting:

```bash
python src/watershed_counter.py --input data/input --output data/output
```

## Run On One Image

```bash
python src/watershed_counter.py --input data/input/example.png --output data/output
```

## Useful Parameters

```bash
python src/watershed_counter.py --input data/input --output data/output --min-area 80 --distance-ratio 0.42 --threshold otsu
```

- `--min-area`: removes tiny false objects.
- `--distance-ratio`: controls how strict marker extraction is. Lower values can detect more objects; higher values detect fewer but cleaner objects.
- `--threshold`: choose `otsu` or `adaptive`.

## Evaluation

If true object counts are known, compare them with the predicted counts printed by the program and saved in `summary.csv`.

For each image:

```text
count_error = abs(predicted_count - true_count)
count_accuracy = 1 - count_error / true_count
```

If ground-truth masks are available, IoU and Dice can be computed:

```text
IoU = intersection / union
Dice = 2 * intersection / (predicted_area + ground_truth_area)
```

