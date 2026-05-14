"""
Reorganizes ALL data (fetal echo + new domains) into a clean quality-label
folder structure ready for PyTorch ImageFolder training.

Output structure
----------------
dataset/
    train/
        good/           <- original clean images
        blurry/         <- poor probe contact / patient movement
        too_dark/       <- insufficient gain
        low_contrast/   <- no gel / wrong tissue window
        noisy/          <- electrical interference
        angled/         <- wrong probe tilt
    valid/
        <same labels>
    test/
        <same labels>

Sources
-------
  1. Fetal echo originals (train/valid/test)           -> good/
  2. Augmented fetal echo (synthetic/fetal_echo_augmented) -> blurry/ too_dark/ etc.
  3. New domain placeholders (synthetic/lung, fast, second_trimester)
       - placeholder "good" images -> good/
       - augmented versions of placeholders -> blurry/ too_dark/ etc.
       (replace placeholders with real diffusion images later — structure stays the same)
"""

import re
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

BASE   = Path(__file__).parent
SOURCE = BASE / "Fetal Echocardiography First Trimester"
AUG    = BASE / "synthetic" / "fetal_echo_augmented"
SYNTH  = BASE / "synthetic"
OUT    = BASE / "dataset"

NEW_DOMAINS = ["lung", "fast", "second_trimester"]

AUG_SUFFIX_MAP = {
    "aug_blur":         "blurry",
    "aug_dark":         "too_dark",
    "aug_low_contrast": "low_contrast",
    "aug_noisy":        "noisy",
    "aug_angled":       "angled",
}

# Augmentation functions (applied to new domain placeholders)
def aug_blur(img):
    import random
    k = random.choice([15, 21, 27])
    return cv2.GaussianBlur(img, (k, k), 0)

def aug_dark(img):
    import random
    return np.clip(img.astype(float) * random.uniform(0.2, 0.45), 0, 255).astype(np.uint8)

def aug_low_contrast(img):
    import random
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    flat = np.full_like(img, int(gray.mean()))
    return cv2.addWeighted(img, random.uniform(0.15, 0.35), flat, random.uniform(0.65, 0.85), 0)

def aug_noisy(img):
    import random
    noise = np.random.normal(0, random.uniform(20, 45), img.shape).astype(np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def aug_angled(img):
    import random
    h, w = img.shape[:2]
    shear = random.uniform(0.1, 0.25) * random.choice([-1, 1])
    M = np.float32([[1, shear, 0], [0, 1, 0]])
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

AUGMENTATIONS = {
    "blurry":       aug_blur,
    "too_dark":     aug_dark,
    "low_contrast": aug_low_contrast,
    "noisy":        aug_noisy,
    "angled":       aug_angled,
}


def safe_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def sanitize(name: str) -> str:
    return re.sub(r"[\s\(\)]+", "_", name).strip("_")


# ─────────────────────────────────────────────
# 1. Original fetal echo images -> good/
# ─────────────────────────────────────────────
def copy_good_images(counts: dict):
    for split in ["train", "valid", "test"]:
        split_dir = SOURCE / split
        if not split_dir.exists():
            continue
        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir():
                continue
            view = class_dir.name
            for img in class_dir.glob("*.jpg"):
                new_name = f"fetal__{sanitize(view)}__{sanitize(img.stem)}.jpg"
                safe_copy(img, OUT / split / "good" / new_name)
                counts[(split, "good")] += 1


# ─────────────────────────────────────────────
# 2. Augmented fetal echo -> blurry/ too_dark/ etc.
# ─────────────────────────────────────────────
def copy_augmented_fetal(counts: dict):
    for view_dir in AUG.iterdir():
        if not view_dir.is_dir():
            continue
        view = view_dir.name

        for img in view_dir.glob("*.jpg"):
            stem = img.stem

            qlabel = None
            for suffix, label in AUG_SUFFIX_MAP.items():
                if stem.endswith(suffix):
                    qlabel = label
                    break
            if qlabel is None:
                continue

            name_lower = stem.lower()
            split = "train" if "train" in name_lower else ("valid" if "valid" in name_lower else "test")

            new_name = f"fetal__{sanitize(view)}__{sanitize(stem)}.jpg"
            safe_copy(img, OUT / split / qlabel / new_name)
            counts[(split, qlabel)] += 1


# ─────────────────────────────────────────────
# 3. New domain placeholders -> good/  +  augmented versions -> bad labels
#    Split 70% train / 15% valid / 15% test
# ─────────────────────────────────────────────
def copy_new_domains(counts: dict):
    np.random.seed(42)

    for domain in NEW_DOMAINS:
        domain_dir = SYNTH / domain
        if not domain_dir.exists():
            continue

        for view_dir in sorted(domain_dir.iterdir()):
            if not view_dir.is_dir():
                continue
            view = view_dir.name

            images = sorted(view_dir.glob("*.jpg"))
            n = len(images)
            indices = np.random.permutation(n)
            train_end = int(0.70 * n)
            valid_end = int(0.85 * n)

            split_map = {}
            for i, idx in enumerate(indices):
                if i < train_end:
                    split_map[images[idx].name] = "train"
                elif i < valid_end:
                    split_map[images[idx].name] = "valid"
                else:
                    split_map[images[idx].name] = "test"

            for img_path in images:
                split = split_map[img_path.name]
                stem  = img_path.stem

                # Copy original as "good"
                new_name = f"{domain}__{sanitize(view)}__{sanitize(stem)}.jpg"
                safe_copy(img_path, OUT / split / "good" / new_name)
                counts[(split, "good")] += 1

                # Generate and save a bad-quality version for each label
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                for qlabel, aug_fn in AUGMENTATIONS.items():
                    aug_img = aug_fn(img)
                    bad_name = f"{domain}__{sanitize(view)}__{sanitize(stem)}__{qlabel}.jpg"
                    dst = OUT / split / qlabel / bad_name
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(dst), aug_img)
                    counts[(split, qlabel)] += 1


# ─────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────
QUALITY_LABELS = ["good", "blurry", "too_dark", "low_contrast", "noisy", "angled"]

def print_summary(counts: dict):
    splits = ["train", "valid", "test"]
    col_w  = 14
    header = f"{'':20}" + "".join(f"{s:>{col_w}}" for s in splits) + f"{'TOTAL':>{col_w}}"
    print("\n" + header)
    print("-" * len(header))

    totals = defaultdict(int)
    grand  = 0
    for label in QUALITY_LABELS:
        row = f"{label:<20}"
        row_total = 0
        for split in splits:
            n = counts[(split, label)]
            row += f"{n:>{col_w}}"
            totals[split] += n
            row_total += n
        row += f"{row_total:>{col_w}}"
        grand += row_total
        print(row)

    print("-" * len(header))
    total_row = f"{'TOTAL':<20}" + "".join(f"{totals[s]:>{col_w}}" for s in splits) + f"{grand:>{col_w}}"
    print(total_row + "\n")


def main():
    if OUT.exists():
        print(f"Removing existing {OUT} ...")
        shutil.rmtree(OUT)

    counts = defaultdict(int)

    print("[1/3] Copying original fetal echo images -> good/ ...")
    copy_good_images(counts)

    print("[2/3] Copying augmented fetal echo images -> quality label folders ...")
    copy_augmented_fetal(counts)

    print("[3/3] Copying new domain images (lung / FAST / 2nd trimester) + augmenting ...")
    copy_new_domains(counts)

    print(f"\nDataset organized at: {OUT}")
    print_summary(counts)

    print("Quality label -> guidance text:")
    print("  good          -> Image quality is good. Proceed with assessment.")
    print("  blurry        -> Apply more gel and hold the probe still.")
    print("  too_dark      -> Increase the gain setting on the machine.")
    print("  low_contrast  -> Re-position the probe at a slightly different angle.")
    print("  noisy         -> Apply fresh gel and reduce probe pressure.")
    print("  angled        -> Rotate the probe until the structure is centred on screen.")


if __name__ == "__main__":
    main()
