"""
generate_video_dataset.py
Converts labeled image dataset → synthetic video clips for video-CNN training.
Each image produces one 16-frame MP4 with realistic probe-motion augmentation.

Output: video_dataset/{train,valid,test}/{class}/clip_XXXX.mp4

Usage:
    python generate_video_dataset.py
    python generate_video_dataset.py --src dataset_224 --dst video_dataset --frames 16 --fps 10
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# ── defaults ──────────────────────────────────────────────────────────────────
SRC_ROOT = Path("dataset_224")
DST_ROOT = Path("video_dataset")
N_FRAMES = 16
FPS      = 10

# Per-class augmentation overrides
CLASS_CFG = {
    "blurry":       {"blur_prob": 0.9, "jitter": 8},
    "angled":       {"angle":     3.5, "jitter": 4},
    "too_dark":     {"bright":    0.14},
    "noisy":        {"noise":     12},
    "low_contrast": {"bright":    0.06},
    "good":         {},
}


def augment_clip(img_bgr: np.ndarray, n_frames: int, cls: str) -> list:
    h, w = img_bgr.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    rng = np.random.default_rng()

    # Pull per-class config (with defaults)
    cfg        = CLASS_CFG.get(cls, {})
    max_jitter = cfg.get("jitter",    5)
    max_angle  = cfg.get("angle",     1.5)
    max_bright = cfg.get("bright",    0.06)
    noise_std  = cfg.get("noise",     0)
    blur_prob  = cfg.get("blur_prob", 0.15)

    # Smooth random walks (probe hand tremor)
    dx = np.clip(np.cumsum(rng.normal(0, 1.0, n_frames)), -max_jitter, max_jitter)
    dy = np.clip(np.cumsum(rng.normal(0, 1.0, n_frames)), -max_jitter, max_jitter)
    da = np.clip(np.cumsum(rng.normal(0, 0.12, n_frames)), -max_angle,  max_angle)
    db = np.clip(np.cumsum(rng.normal(0, 0.006, n_frames)), -max_bright, max_bright)

    # Optional blur event (motion artefact burst)
    blur_frames = set()
    if rng.random() < blur_prob:
        start = int(rng.integers(0, max(1, n_frames - 4)))
        blur_frames = set(range(start, min(start + 4, n_frames)))

    frames = []
    for i in range(n_frames):
        # Affine: rotation + translation
        M = cv2.getRotationMatrix2D((cx, cy), float(da[i]), 1.0)
        M[0, 2] += dx[i]
        M[1, 2] += dy[i]
        frame = cv2.warpAffine(img_bgr, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        # Brightness / contrast drift
        frame = np.clip(frame.astype(np.float32) * (1.0 + db[i]), 0, 255).astype(np.uint8)

        # Gaussian noise (for noisy class)
        if noise_std > 0:
            noise = rng.normal(0, noise_std, frame.shape).astype(np.float32)
            frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # Motion blur burst
        if i in blur_frames:
            k = int(rng.choice([5, 7, 9]))
            kernel = np.zeros((k, k), dtype=np.float32)
            kernel[k // 2, :] = 1.0 / k
            frame = cv2.filter2D(frame, -1, kernel)

        frames.append(frame)

    return frames


def process_split(split: str, src_root: Path, dst_root: Path, n_frames: int, fps: int):
    src_split = src_root / split
    if not src_split.exists():
        print(f"  [skip] {src_split} not found")
        return 0

    total = 0
    for cls_dir in sorted(src_split.iterdir()):
        if not cls_dir.is_dir():
            continue
        cls_name = cls_dir.name
        out_dir  = dst_root / split / cls_name
        out_dir.mkdir(parents=True, exist_ok=True)

        images = (
            sorted(cls_dir.glob("*.jpg")) +
            sorted(cls_dir.glob("*.jpeg")) +
            sorted(cls_dir.glob("*.png")) +
            sorted(cls_dir.glob("*.bmp"))
        )

        for idx, img_path in enumerate(tqdm(images, desc=f"{split}/{cls_name}", leave=False)):
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            frames   = augment_clip(img, n_frames, cls_name)
            out_path = out_dir / f"clip_{idx:04d}.mp4"
            h, w     = frames[0].shape[:2]

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
            for f in frames:
                writer.write(f)
            writer.release()
            total += 1

        print(f"    {split}/{cls_name}: {len(images)} clips written")

    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src",    default=str(SRC_ROOT))
    ap.add_argument("--dst",    default=str(DST_ROOT))
    ap.add_argument("--frames", type=int, default=N_FRAMES)
    ap.add_argument("--fps",    type=int, default=FPS)
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)

    print(f"Source : {src}")
    print(f"Output : {dst}")
    print(f"Frames : {args.frames} @ {args.fps} fps ({args.frames / args.fps:.1f}s per clip)")
    print()

    total = 0
    for split in ["train", "valid", "test"]:
        print(f"[{split}]")
        total += process_split(split, src, dst, args.frames, args.fps)

    print(f"\nDone — {total} clips saved to {dst}/")


if __name__ == "__main__":
    main()
