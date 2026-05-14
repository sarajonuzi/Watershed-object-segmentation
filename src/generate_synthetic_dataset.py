from pathlib import Path
import random

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "data" / "input"


def draw_synthetic_objects(path: Path, count: int, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

    height, width = 520, 700
    image = np.full((height, width, 3), 235, dtype=np.uint8)

    centers = []
    attempts = 0
    while len(centers) < count and attempts < count * 100:
        attempts += 1
        radius = random.randint(28, 48)
        x = random.randint(radius + 20, width - radius - 20)
        y = random.randint(radius + 20, height - radius - 20)
        if all(np.hypot(x - cx, y - cy) > 0.95 * (radius + cr) for cx, cy, cr in centers):
            centers.append((x, y, radius))

    for x, y, radius in centers:
        color = (
            random.randint(80, 160),
            random.randint(90, 170),
            random.randint(120, 210),
        )
        cv2.circle(image, (x, y), radius, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(image, (x - radius // 4, y - radius // 4), radius // 5, (255, 255, 255), -1, lineType=cv2.LINE_AA)

    noise = np.random.normal(0, 6, image.shape).astype(np.int16)
    noisy = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), noisy)


def main() -> None:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)

    examples = [
        ("synthetic_08_objects.png", 8, 10),
        ("synthetic_12_objects.png", 12, 20),
        ("synthetic_16_objects.png", 16, 30),
    ]

    for filename, count, seed in examples:
        draw_synthetic_objects(INPUT_DIR / filename, count, seed)

    truth_path = INPUT_DIR / "ground_truth_counts.csv"
    truth_path.write_text(
        "image,true_count\n"
        + "\n".join(f"{filename},{count}" for filename, count, _ in examples)
        + "\n",
        encoding="utf-8",
    )

    print(f"Generated {len(examples)} images in {INPUT_DIR}")


if __name__ == "__main__":
    main()
