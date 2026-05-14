"""
Synthetic ultrasound data generator for the Real-Time Guidance System.

Strategy
--------
We cannot run a full diffusion model locally, so this script does two things:

1. AUGMENTATION-BASED SYNTHETIC DATA
   Takes existing fetal echo images and produces realistic corruptions that
   simulate the quality problems an untrained operator causes:
     - blur        (poor probe contact / patient movement)
     - dark        (insufficient gain)
     - low_contrast (wrong tissue window / no gel)
     - noisy        (electrical interference / cheap machine)
     - angled       (wrong probe tilt)

   Each augmented image gets a quality label + guidance text automatically,
   giving us labelled training pairs without any manual annotation.

2. METADATA TEMPLATES FOR NEW DOMAINS
   Creates the folder structure + a CSV of synthetic metadata for three new
   ultrasound domains relevant to rural/low-resource settings.
   When you run a real diffusion model (e.g. MedSyn, SonoSim, or fine-tuned
   Stable Diffusion) you drop generated images into these folders and the
   metadata CSV is already waiting.

New domains
-----------
  A. Lung ultrasound      -- diagnose pneumonia / pleural effusion (LMIC #1 killer)
  B. FAST exam (trauma)   -- detect internal bleeding in emergencies
  C. 2nd/3rd trimester    -- extend fetal coverage beyond 1st trimester

Usage
-----
  python generate_synthetic.py

Outputs
-------
  synthetic/
    fetal_echo_augmented/   <- augmented versions of existing images
    lung/                   <- placeholder folders for diffusion output
    fast/                   <- placeholder folders for diffusion output
    second_trimester/       <- placeholder folders for diffusion output
  synthetic_metadata.csv    <- labels + guidance for all synthetic rows
"""

import csv
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

random.seed(42)
np.random.seed(42)

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
BASE   = Path(__file__).parent
SOURCE = BASE / "Fetal Echocardiography First Trimester"
OUT    = BASE / "synthetic"
OUT.mkdir(exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# Guidance text (mirrors label_quality.py so both CSVs are consistent)
# ──────────────────────────────────────────────────────────────────────────────
GUIDANCE = {
    "good":         "Image quality is good. Proceed with assessment.",
    "blurry":       "Image is blurry. Press the probe firmly against the skin and apply more ultrasound gel. Hold the probe still and ask the patient to hold their breath.",
    "too_dark":     "Image is too dark. Increase the gain setting on the machine, or apply more gel and press the probe more firmly.",
    "low_contrast": "Low contrast detected. Re-position the probe -- try a slightly different angle. Ensure the patient has a full bladder if scanning the abdomen.",
    "noisy":        "Image is noisy. Wipe off old gel, apply fresh gel, and reduce probe pressure slightly. Ask the patient to remain still.",
    "angled":       "Probe angle is off. Rotate the probe slightly until the structure appears centred on screen. Small adjustments work best.",
}

# ──────────────────────────────────────────────────────────────────────────────
# Image augmentations that simulate operator errors
# ──────────────────────────────────────────────────────────────────────────────

def aug_blur(img):
    """
    Realistic poor probe contact / patient movement.
    Combines motion blur (directional smear) + spatially uneven blur
    (stronger on one side, simulating partial probe lift) + slight
    Rayleigh speckle degradation.
    """
    h, w = img.shape[:2]

    # Random motion blur direction
    angle  = random.uniform(0, 180)
    length = random.randint(15, 35)
    kernel = np.zeros((length, length))
    kernel[length // 2, :] = 1
    M = cv2.getRotationMatrix2D((length // 2, length // 2), angle, 1)
    kernel = cv2.warpAffine(kernel, M, (length, length))
    kernel = kernel / kernel.sum()
    blurred = cv2.filter2D(img, -1, kernel)

    # Spatially uneven blur — stronger on a random edge (partial probe lift)
    side = random.choice(["left", "right", "top", "bottom"])
    mask = np.ones((h, w), dtype=np.float32)
    if side == "left":
        fade = random.randint(w // 4, w // 2)
        mask[:, :fade] = np.linspace(0, 1, fade)
    elif side == "right":
        fade = random.randint(w // 4, w // 2)
        mask[:, w - fade:] = np.linspace(1, 0, fade)
    elif side == "top":
        fade = random.randint(h // 4, h // 2)
        mask[:fade, :] = np.linspace(0, 1, fade).reshape(-1, 1)
    else:
        fade = random.randint(h // 4, h // 2)
        mask[h - fade:, :] = np.linspace(1, 0, fade).reshape(-1, 1)

    mask3 = np.stack([mask] * 3, axis=-1)
    heavy = cv2.GaussianBlur(img, (31, 31), 0)
    out   = (blurred * mask3 + heavy * (1 - mask3)).astype(np.uint8)

    # Add mild speckle degradation
    speckle = np.random.rayleigh(8, out.shape).astype(np.int16)
    out = np.clip(out.astype(np.int16) + speckle - 4, 0, 255).astype(np.uint8)
    return out, "blurry"


def aug_dark(img):
    """
    Realistic insufficient gain.
    Non-uniform darkening (deeper = darker, simulating real attenuation),
    plus slight contrast crush in shadows.
    """
    h, w  = img.shape[:2]
    base_factor = random.uniform(0.15, 0.40)

    # Depth-dependent attenuation gradient (top brighter, bottom darker)
    gradient = np.linspace(base_factor + 0.15, base_factor, h,
                           dtype=np.float32).reshape(h, 1, 1)
    darkened = np.clip(img.astype(np.float32) * gradient, 0, 255)

    # Crush shadows slightly (make dark areas even darker)
    darkened = np.power(darkened / 255.0, random.uniform(1.2, 1.8)) * 255
    return darkened.astype(np.uint8), "too_dark"


def aug_low_contrast(img):
    """
    Realistic low contrast — wrong tissue plane / insufficient gel.
    Compresses the dynamic range toward mid-grey and adds a diffuse
    haze layer simulating acoustic reverberation near the probe.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mid  = int(gray.mean())

    # Compress dynamic range
    alpha   = random.uniform(0.12, 0.30)
    flat    = np.full_like(img, mid)
    blended = cv2.addWeighted(img, alpha, flat, 1 - alpha, 0).astype(np.float32)

    # Add reverberation haze in top third (near-field artefact)
    h = img.shape[0]
    haze_depth = random.randint(h // 5, h // 3)
    haze_strength = random.uniform(0.15, 0.35)
    haze = np.full_like(blended, mid * 0.6, dtype=np.float32)
    fade = np.linspace(haze_strength, 0, haze_depth).reshape(-1, 1, 1)
    blended[:haze_depth] = blended[:haze_depth] * (1 - fade) + haze[:haze_depth] * fade

    return blended.astype(np.uint8), "low_contrast"

def aug_noisy(img):
    """
    Realistic electrical/acoustic noise from a cheap machine.
    Combines Rayleigh speckle (ultrasound-specific), horizontal
    scan-line banding (electrical interference), and random
    dropout pixels (dead transducer elements).
    """
    h, w  = img.shape[:2]
    out   = img.astype(np.float32)

    # Rayleigh speckle (multiplicative — brighter areas get more noise)
    speckle = np.random.rayleigh(random.uniform(18, 35), (h, w, 3)).astype(np.float32)
    out     = out + speckle * (out / 255.0)

    # Horizontal scan-line banding (electrical interference)
    n_bands  = random.randint(3, 10)
    band_ys  = np.random.randint(0, h, n_bands)
    strength = random.uniform(15, 40)
    for y in band_ys:
        thickness = random.randint(1, 4)
        y2 = min(y + thickness, h)
        out[y:y2] += strength * random.choice([-1, 1])

    # Dead transducer elements (vertical dropout lines)
    n_lines = random.randint(1, 4)
    for _ in range(n_lines):
        x = random.randint(0, w - 1)
        out[:, x] = random.uniform(0, 30)

    return np.clip(out, 0, 255).astype(np.uint8), "noisy"


def aug_angled(img):
    """
    Realistic wrong probe tilt.
    Combines perspective warp (simulating probe rocking) with
    one-sided signal dropout (structures fall out of the scan plane).
    """
    h, w = img.shape[:2]

    # Perspective warp — probe rocked to one side
    tilt = random.uniform(0.08, 0.20)
    side = random.choice(["left", "right"])
    if side == "left":
        src = np.float32([[0,0],[w,0],[w,h],[0,h]])
        dst = np.float32([[w*tilt,h*tilt],[w,0],[w,h],[0,h]])
    else:
        src = np.float32([[0,0],[w,0],[w,h],[0,h]])
        dst = np.float32([[0,0],[w-w*tilt,h*tilt],[w,h],[0,h]])

    M   = cv2.getPerspectiveTransform(src, dst)
    out = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    # One-sided signal loss (structure dropped off scan plane)
    dropout_width = random.randint(w // 6, w // 3)
    fade = np.linspace(1, 0, dropout_width, dtype=np.float32)
    if side == "left":
        out[:, :dropout_width] = (out[:, :dropout_width] *
                                   fade.reshape(1, -1, 1)).astype(np.uint8)
    else:
        out[:, w-dropout_width:] = (out[:, w-dropout_width:] *
                                     fade[::-1].reshape(1, -1, 1)).astype(np.uint8)

    return out, "angled"

AUGMENTATIONS = [aug_blur, aug_dark, aug_low_contrast, aug_noisy, aug_angled]


def augment_existing() -> list[dict]:
    """
    Apply every augmentation to every image in the dataset.
    This produces 6,720 images per quality label (one per original),
    perfectly balancing the 6,720 good originals.
    """
    aug_dir = OUT / "fetal_echo_augmented"
    aug_dir.mkdir(exist_ok=True)

    rows = []
    splits = ["train", "valid", "test"]

    for split in splits:
        for class_dir in sorted((SOURCE / split).iterdir()):
            if not class_dir.is_dir():
                continue
            view   = class_dir.name
            images = sorted(class_dir.glob("*.jpg"))
            sample = images  # use ALL images, no sampling

            view_out = aug_dir / view
            view_out.mkdir(exist_ok=True)

            for aug_fn in AUGMENTATIONS:
                for src_path in sample:
                    img = cv2.imread(str(src_path))
                    if img is None:
                        continue
                    aug_img, qlabel = aug_fn(img)
                    stem     = src_path.stem
                    out_name = f"{stem}__{aug_fn.__name__}.jpg"
                    out_path = view_out / out_name
                    cv2.imwrite(str(out_path), aug_img)

                    rows.append({
                        "domain":        "fetal_echo_1st_trimester",
                        "split":         split,
                        "view":          view,
                        "filename":      out_name,
                        "filepath":      str(out_path.relative_to(BASE)),
                        "source":        "augmentation",
                        "quality_label": qlabel,
                        "guidance_text": GUIDANCE[qlabel],
                        "abnormal_flag": 0,
                        "notes":         f"Augmented from {src_path.name} via {aug_fn.__name__}",
                    })

    print(f"  Augmented fetal echo: {len(rows)} images")
    return rows


# ──────────────────────────────────────────────────────────────────────────────
# New domain metadata templates
# (fill these folders with diffusion-generated images later)
# ──────────────────────────────────────────────────────────────────────────────

LUNG_VIEWS = {
    "A_lines": {
        "description": "Normal lung. Horizontal reverberation artefacts parallel to pleural line.",
        "guidance_good": "Normal lung pattern visible. A-lines present. No urgent referral needed.",
        "diffusion_prompt": "ultrasound image of lung, A-lines artifact, normal lung, horizontal reverberation, pleural line visible, grayscale",
    },
    "B_lines": {
        "description": "Interstitial syndrome / pulmonary oedema / early pneumonia. Vertical comet-tail artefacts.",
        "guidance_good": "B-lines detected. This may indicate fluid in the lungs. Refer to clinician urgently.",
        "diffusion_prompt": "ultrasound image of lung, B-lines, comet tail artifact, vertical bright lines from pleura, grayscale, medical",
    },
    "consolidation": {
        "description": "Pneumonia or atelectasis. Tissue-like echogenicity replacing air pattern.",
        "guidance_good": "Consolidation pattern detected. Urgent specialist referral required.",
        "diffusion_prompt": "ultrasound image of lung consolidation, hepatisation, tissue-like echotexture, air bronchograms, grayscale",
    },
    "pleural_effusion": {
        "description": "Free fluid above diaphragm. Anechoic (black) collection.",
        "guidance_good": "Fluid collection detected above diaphragm. Refer urgently.",
        "diffusion_prompt": "ultrasound image of pleural effusion, anechoic fluid collection, above diaphragm, grayscale, medical ultrasound",
    },
}

FAST_VIEWS = {
    "RUQ_normal": {
        "description": "Right upper quadrant (Morrison's pouch). No free fluid.",
        "guidance_good": "Right upper quadrant: no free fluid seen. Continue assessment.",
        "diffusion_prompt": "FAST ultrasound RUQ Morrison's pouch normal, liver kidney interface, no free fluid, grayscale",
    },
    "RUQ_fluid": {
        "description": "Right upper quadrant with free fluid (anechoic stripe at liver-kidney interface).",
        "guidance_good": "Free fluid detected in right upper quadrant. Urgent surgical referral required.",
        "diffusion_prompt": "FAST ultrasound RUQ Morrison's pouch, anechoic free fluid between liver and kidney, hemorrhage, grayscale",
    },
    "LUQ_normal": {
        "description": "Left upper quadrant (splenorenal). No free fluid.",
        "guidance_good": "Left upper quadrant: no free fluid seen. Continue assessment.",
        "diffusion_prompt": "FAST ultrasound LUQ splenorenal recess normal, spleen kidney interface, no free fluid, grayscale",
    },
    "pericardial_normal": {
        "description": "Subxiphoid cardiac view. No pericardial effusion.",
        "guidance_good": "Heart view: no pericardial fluid detected.",
        "diffusion_prompt": "FAST ultrasound subxiphoid cardiac view, heart, no pericardial effusion, grayscale",
    },
    "pericardial_effusion": {
        "description": "Pericardial effusion. Anechoic collection around heart.",
        "guidance_good": "Fluid around the heart detected. Urgent emergency referral required.",
        "diffusion_prompt": "FAST ultrasound pericardial effusion, anechoic fluid surrounding heart, tamponade risk, grayscale",
    },
    "pelvic_normal": {
        "description": "Pelvic view. No free fluid in pelvis.",
        "guidance_good": "Pelvis: no free fluid detected.",
        "diffusion_prompt": "FAST ultrasound pelvic view, bladder, no free fluid, grayscale",
    },
}

SECOND_TRIMESTER_VIEWS = {
    "four_chamber": {
        "description": "Fetal four-chamber cardiac view (18-22 weeks). Both ventricles and atria visible.",
        "guidance_good": "Four-chamber cardiac view obtained. Image quality adequate for assessment.",
        "diffusion_prompt": "obstetric ultrasound fetal heart four chamber view 20 weeks, ventricles atria, grayscale, 2nd trimester",
    },
    "biparietal_diameter": {
        "description": "BPD measurement plane. Symmetric oval skull at level of thalami.",
        "guidance_good": "BPD view obtained. Measure between outer and inner skull edges.",
        "diffusion_prompt": "obstetric ultrasound biparietal diameter BPD fetal head 20 weeks, thalami, cavum septum pellucidum, grayscale",
    },
    "femur_length": {
        "description": "Femur length measurement. Long axis of fetal femur shaft.",
        "guidance_good": "Femur view obtained. Measure the full length of the bone shaft.",
        "diffusion_prompt": "obstetric ultrasound fetal femur length measurement 20 weeks, long axis bone, grayscale",
    },
    "abdominal_circumference": {
        "description": "AC measurement plane. Round cross-section at level of stomach and portal vein.",
        "guidance_good": "Abdominal view obtained. Ensure the section is round and shows stomach bubble.",
        "diffusion_prompt": "obstetric ultrasound fetal abdominal circumference AC 20 weeks, stomach, portal vein, round cross section, grayscale",
    },
    "spine": {
        "description": "Fetal spine longitudinal. Two parallel echogenic lines (laminae).",
        "guidance_good": "Spine view obtained. Check for continuity of both lines along the full length.",
        "diffusion_prompt": "obstetric ultrasound fetal spine longitudinal 20 weeks, two parallel echogenic lines, neural tube, grayscale",
    },
    "placenta_location": {
        "description": "Placental position relative to internal os.",
        "guidance_good": "Placenta visible. Note position -- if near or covering cervix, refer for specialist review.",
        "diffusion_prompt": "obstetric ultrasound placenta position anterior posterior fundal 20 weeks, uterus, grayscale",
    },
    "third_trimester_presentation": {
        "description": "Fetal lie and presentation (32-36 weeks). Cephalic vs breech.",
        "guidance_good": "Fetal position identified. Note presentation for delivery planning.",
        "diffusion_prompt": "obstetric ultrasound fetal presentation 32 weeks, cephalic breech, third trimester, fetal head pelvis, grayscale",
    },
}


def create_domain_structure(domain_name: str, views: dict, n_placeholders: int = 50) -> list[dict]:
    """
    Create folder structure and metadata rows for a new domain.
    Placeholder images are grey squares (224x224) -- replace with
    diffusion-generated images.
    """
    domain_dir = OUT / domain_name
    domain_dir.mkdir(exist_ok=True)
    rows = []

    for view_name, meta in views.items():
        view_dir = domain_dir / view_name
        view_dir.mkdir(exist_ok=True)

        # Write a README in each view folder
        readme = view_dir / "README.txt"
        readme.write_text(
            f"View: {view_name}\n"
            f"Description: {meta['description']}\n\n"
            f"Diffusion prompt to use:\n{meta['diffusion_prompt']}\n\n"
            f"Guidance text (good quality):\n{meta['guidance_good']}\n\n"
            f"Drop {n_placeholders}+ generated images here, then run label_quality.py."
        )

        # Create grey placeholder images so the folder is not empty
        for i in range(n_placeholders):
            ph = np.full((224, 224, 3), 128, dtype=np.uint8)
            ph_name = f"{view_name}_placeholder_{i:03d}.jpg"
            cv2.imwrite(str(view_dir / ph_name), ph)

            rows.append({
                "domain":        domain_name,
                "split":         "train" if i < 35 else ("valid" if i < 43 else "test"),
                "view":          view_name,
                "filename":      ph_name,
                "filepath":      str((view_dir / ph_name).relative_to(BASE)),
                "source":        "placeholder_replace_with_diffusion",
                "quality_label": "good",
                "guidance_text": meta["guidance_good"],
                "abnormal_flag": 0,
                "notes":         meta["diffusion_prompt"],
            })

    print(f"  {domain_name}: {len(rows)} placeholder rows ({len(views)} views)")
    return rows


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("Generating synthetic data...\n")
    all_rows = []

    print("[1/1] Augmenting existing fetal echo images (all images, all augmentations)...")
    all_rows += augment_existing()

    # New domains (lung, fast, second_trimester) already have real procedurally
    # generated images from generate_ultrasound_images.py -- do NOT create placeholders.

    # Write combined metadata CSV
    out_csv = BASE / "synthetic_metadata.csv"
    fieldnames = [
        "domain", "split", "view", "filename", "filepath",
        "source", "quality_label", "guidance_text", "abnormal_flag", "notes",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nDone. {len(all_rows)} augmented rows -> synthetic_metadata.csv")
    print(f"Output folder: {OUT}")

if __name__ == "__main__":
    main()
