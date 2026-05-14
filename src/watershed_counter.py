from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class SegmentationResult:
    count: int
    areas: list[int]
    binary_mask: np.ndarray
    distance_view: np.ndarray
    labels: np.ndarray
    overlay: np.ndarray


def list_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(path for path in input_path.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


def fill_holes(binary: np.ndarray) -> np.ndarray:
    flood = binary.copy()
    mask = np.zeros((binary.shape[0] + 2, binary.shape[1] + 2), np.uint8)
    cv2.floodFill(flood, mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    return cv2.bitwise_or(binary, holes)


def build_binary_mask(gray: np.ndarray, threshold_method: str) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    if threshold_method == "adaptive":
        binary = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            41,
            3,
        )
    else:
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    return fill_holes(binary)


def segment_objects(
    image: np.ndarray,
    threshold_method: str = "otsu",
    min_area: int = 100,
    distance_ratio: float = 0.40,
) -> SegmentationResult:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = build_binary_mask(gray, threshold_method)

    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    distance_view = cv2.normalize(distance, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    _, sure_foreground = cv2.threshold(distance, distance_ratio * distance.max(), 255, 0)
    sure_foreground = sure_foreground.astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)
    sure_background = cv2.dilate(binary, kernel, iterations=3)
    unknown = cv2.subtract(sure_background, sure_foreground)

    marker_count, markers = cv2.connectedComponents(sure_foreground)
    markers = markers + 1
    markers[unknown == 255] = 0

    watershed_labels = cv2.watershed(image.copy(), markers)

    final_labels = np.zeros_like(watershed_labels, dtype=np.int32)
    areas: list[int] = []
    next_label = 1

    for label in range(2, marker_count + 1):
        region = (watershed_labels == label).astype(np.uint8)
        area = int(cv2.countNonZero(region))
        if area < min_area:
            continue
        final_labels[region == 1] = next_label
        areas.append(area)
        next_label += 1

    overlay = image.copy()
    overlay[watershed_labels == -1] = (0, 0, 255)

    colored_labels = colorize_labels(final_labels)
    overlay = cv2.addWeighted(overlay, 0.65, colored_labels, 0.35, 0)

    return SegmentationResult(
        count=len(areas),
        areas=areas,
        binary_mask=binary,
        distance_view=distance_view,
        labels=final_labels,
        overlay=overlay,
    )


def colorize_labels(labels: np.ndarray) -> np.ndarray:
    colored = np.zeros((*labels.shape, 3), dtype=np.uint8)
    rng = np.random.default_rng(42)

    for label in np.unique(labels):
        if label == 0:
            continue
        color = rng.integers(40, 230, size=3, dtype=np.uint8)
        colored[labels == label] = color

    return colored


def save_result(image_path: Path, output_dir: Path, result: SegmentationResult) -> None:
    stem = image_path.stem
    image_output = output_dir / stem
    image_output.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(image_output / "binary_mask.png"), result.binary_mask)
    cv2.imwrite(str(image_output / "distance_transform.png"), result.distance_view)
    cv2.imwrite(str(image_output / "watershed_overlay.png"), result.overlay)
    cv2.imwrite(str(image_output / "labeled_objects.png"), colorize_labels(result.labels))


def run(input_path: Path, output_dir: Path, threshold_method: str, min_area: int, distance_ratio: float) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []

    for image_path in list_images(input_path):
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Skipping unreadable image: {image_path}")
            continue

        result = segment_objects(
            image,
            threshold_method=threshold_method,
            min_area=min_area,
            distance_ratio=distance_ratio,
        )
        save_result(image_path, output_dir, result)

        records.append(
            {
                "image": image_path.name,
                "predicted_count": result.count,
                "areas": ";".join(str(area) for area in result.areas),
            }
        )
        print(f"{image_path.name}: {result.count} objects")

    summary = pd.DataFrame(records)
    summary.to_csv(output_dir / "summary.csv", index=False)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watershed-based object segmentation and counting.")
    parser.add_argument("--input", type=Path, help="Input image or directory.")
    parser.add_argument("--output", type=Path, help="Output directory.")
    parser.add_argument("--threshold", choices=["otsu", "adaptive"], default="otsu")
    parser.add_argument("--min-area", type=int, default=100)
    parser.add_argument("--distance-ratio", type=float, default=0.40)
    args, _ = parser.parse_known_args()

    project_root = Path(__file__).resolve().parents[1]
    if args.input is None:
        args.input = project_root / "data" / "input"
    if args.output is None:
        args.output = project_root / "data" / "output"

    return args


def main() -> None:
    args = parse_args()
    run(args.input, args.output, args.threshold, args.min_area, args.distance_ratio)


if __name__ == "__main__":
    main()
