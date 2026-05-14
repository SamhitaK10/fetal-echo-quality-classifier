"""
Quality labeling pipeline for ultrasound images.

For each image this script computes:
  - sharpness  (Laplacian variance — low = blurry / poor probe contact)
  - brightness (mean pixel value — low = too dark, high = over-exposed)
  - contrast   (pixel std — low = low contrast / no gel / wrong tissue)
  - noise      (difference between image and a blurred version of itself)

It then assigns:
  - quality_label : good | blurry | too_dark | low_contrast | over_exposed
  - guidance_text : natural-language instruction for the operator
  - abnormal_flag : 1 if the image should be reviewed by a specialist
                    (heuristic: correct view but low brightness = poor window)

Output: labels.csv  (one row per image, all splits combined)
"""

import csv
import os
from pathlib import Path

import cv2
import numpy as np

# ──────────────────────────────────────────────
# Thresholds  (tuned to ultrasound's dark palette)
# ──────────────────────────────────────────────
BLUR_THRESH       = 40    # Laplacian variance below this → blurry
DARK_THRESH       = 20    # mean brightness below this   → too dark
BRIGHT_THRESH     = 180   # mean brightness above this   → over-exposed
CONTRAST_THRESH   = 25    # pixel std below this         → low contrast
NOISE_THRESH      = 15    # noise score above this       → noisy

# ──────────────────────────────────────────────
# Guidance text  (what the operator should do)
# ──────────────────────────────────────────────
GUIDANCE = {
    "good":          "Image quality is good. Proceed with assessment.",
    "blurry":        "Image is blurry. Press the probe firmly against the skin and apply more ultrasound gel. Hold the probe still and ask the patient to hold their breath.",
    "too_dark":      "Image is too dark. Increase the gain setting on the machine, or apply more gel and press the probe more firmly.",
    "over_exposed":  "Image is too bright. Reduce the gain setting on the machine slightly.",
    "low_contrast":  "Low contrast detected. Re-position the probe — try a slightly different angle. Ensure the patient has a full bladder if scanning the abdomen.",
    "noisy":         "Image is noisy. Wipe off old gel, apply fresh gel, and reduce probe pressure slightly. Ask the patient to remain still.",
}

# View types that exist in this dataset
VIEW_LABELS = {"Aorta", "Flows", "V sign", "X sign", "Other"}

# Views that warrant specialist review if image quality is poor
CRITICAL_VIEWS = {"Aorta", "Flows", "V sign", "X sign"}


def laplacian_variance(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def noise_score(gray: np.ndarray) -> float:
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return float(np.abs(gray.astype(float) - blurred.astype(float)).mean())


def classify_quality(gray: np.ndarray) -> str:
    blur  = laplacian_variance(gray)
    mean  = float(gray.mean())
    std   = float(gray.std())
    noise = noise_score(gray)

    if mean < DARK_THRESH:
        return "too_dark"
    if mean > BRIGHT_THRESH:
        return "over_exposed"
    if blur < BLUR_THRESH:
        return "blurry"
    if std < CONTRAST_THRESH:
        return "low_contrast"
    if noise > NOISE_THRESH:
        return "noisy"
    return "good"


def abnormal_flag(view: str, quality: str) -> int:
    """
    Simple heuristic: flag for specialist review when
    a critical cardiac view has poor image quality
    (could be masking a real finding).
    """
    return int(view in CRITICAL_VIEWS and quality != "good")


def process_dataset(root: Path, output_csv: Path):
    rows = []
    splits = ["train", "valid", "test"]

    for split in splits:
        split_dir = root / split
        if not split_dir.exists():
            continue
        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            view = class_dir.name
            for img_path in sorted(class_dir.glob("*.jpg")):
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                blur   = round(laplacian_variance(gray), 2)
                mean   = round(float(gray.mean()), 2)
                std    = round(float(gray.std()), 2)
                noise  = round(noise_score(gray), 2)
                qlabel = classify_quality(gray)
                flag   = abnormal_flag(view, qlabel)

                rows.append({
                    "split":          split,
                    "view":           view,
                    "filename":       img_path.name,
                    "filepath":       str(img_path.relative_to(root)),
                    "sharpness":      blur,
                    "brightness":     mean,
                    "contrast_std":   std,
                    "noise":          noise,
                    "quality_label":  qlabel,
                    "guidance_text":  GUIDANCE[qlabel],
                    "abnormal_flag":  flag,
                })

    fieldnames = [
        "split", "view", "filename", "filepath",
        "sharpness", "brightness", "contrast_std", "noise",
        "quality_label", "guidance_text", "abnormal_flag",
    ]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Labeled {len(rows)} images -> {output_csv}")

    # Summary
    from collections import Counter
    qlabels = Counter(r["quality_label"] for r in rows)
    flags   = sum(r["abnormal_flag"] for r in rows)
    print("\nQuality label distribution:")
    for label, count in sorted(qlabels.items(), key=lambda x: -x[1]):
        pct = 100 * count / len(rows)
        print(f"  {label:<15} {count:>5}  ({pct:.1f}%)")
    print(f"\nAbnormal flags: {flags} ({100*flags/len(rows):.1f}%)")


if __name__ == "__main__":
    ROOT = Path(__file__).parent / "Fetal Echocardiography First Trimester"
    OUT  = Path(__file__).parent / "labels.csv"
    process_dataset(ROOT, OUT)
