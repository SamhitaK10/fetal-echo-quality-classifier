"""
Procedural synthetic ultrasound image generator.

Generates realistic-looking ultrasound images for three domains:
  - Lung ultrasound  (A-lines, B-lines, consolidation, pleural effusion)
  - FAST exam        (RUQ/LUQ normal+fluid, pericardial normal+effusion, pelvic)
  - 2nd/3rd trimester obstetric (four-chamber, BPD, femur, AC, spine, placenta, presentation)

Technique
---------
Real ultrasound images have a distinct visual signature that we can simulate:
  1. Fan-shaped (sector) scan area on a black background
  2. Multiplicative Rayleigh speckle noise -- the "grainy" texture
  3. Depth / gain gradient (brighter near probe, darker deep)
  4. Domain-specific structures: bright hyperechoic lines, dark anechoic fluid,
     mid-grey tissue, reverberation artefacts, posterior enhancement
  5. Machine UI overlay: depth scale, probe marker, measurement callipers

These images are good enough for training a quality classifier because they
share the same noise characteristics and visual artefacts as real ultrasound.
Replace them with real diffusion-generated images later for production use.

Usage
-----
  python generate_ultrasound_images.py
"""

import random
from pathlib import Path

import cv2
import numpy as np

random.seed(42)
np.random.seed(42)

BASE  = Path(__file__).parent
SYNTH = BASE / "synthetic"

IMG_W, IMG_H = 640, 480   # output resolution
N_PER_VIEW   = 150        # images per view (50 train-ish, 50 valid-ish, 50 test-ish)


# ──────────────────────────────────────────────────────────────────────────────
# Core rendering primitives
# ──────────────────────────────────────────────────────────────────────────────

def make_canvas() -> np.ndarray:
    """Black canvas matching the machine UI background."""
    return np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)


def speckle_noise(shape, intensity=0.5) -> np.ndarray:
    """Rayleigh-distributed multiplicative speckle -- the signature of ultrasound."""
    rayleigh = np.random.rayleigh(scale=intensity, size=shape)
    return np.clip(rayleigh, 0, 1)


def draw_sector(canvas, apex_x, apex_y, angle_spread=70, depth=380,
                base_brightness=90, deep_brightness=30) -> np.ndarray:
    """
    Draw the fan-shaped scan area.
    Returns a mask (1 inside sector, 0 outside) and fills canvas with tissue texture.
    """
    mask = np.zeros((IMG_H, IMG_W), dtype=np.float32)
    half = angle_spread / 2

    for y in range(IMG_H):
        for x in range(IMG_W):
            dx = x - apex_x
            dy = y - apex_y
            dist = np.sqrt(dx**2 + dy**2)
            if dist == 0 or dist > depth:
                continue
            angle = np.degrees(np.arctan2(dx, dy))
            if abs(angle) <= half:
                mask[y, x] = 1.0

    # Depth-dependent brightness
    brightness_map = np.zeros((IMG_H, IMG_W), dtype=np.float32)
    for y in range(IMG_H):
        for x in range(IMG_W):
            if mask[y, x]:
                dy = y - apex_y
                dx = x - apex_x
                dist = np.sqrt(dx**2 + dy**2)
                t = dist / depth
                brightness_map[y, x] = base_brightness * (1 - t) + deep_brightness * t

    # Speckle texture
    sp = speckle_noise((IMG_H, IMG_W), intensity=0.6)
    texture = (brightness_map * sp).astype(np.uint8)
    canvas[:, :, 0] = texture
    canvas[:, :, 1] = texture
    canvas[:, :, 2] = texture

    return mask


def draw_depth_scale(canvas, apex_y=40, depth=380, n_marks=8):
    """Add depth markers on the right edge (typical machine UI)."""
    x = IMG_W - 18
    for i in range(n_marks + 1):
        y = int(apex_y + (depth / n_marks) * i)
        if 0 <= y < IMG_H:
            cv2.line(canvas, (x - 6, y), (x + 6, y), (180, 180, 180), 1)
            label = f"{i * 2}cm"
            cv2.putText(canvas, label, (x - 35, y + 4),
                        cv2.FONT_HERSHEY_PLAIN, 0.6, (150, 150, 150), 1)


def add_ui_chrome(canvas):
    """Minimal machine UI: probe marker at top, depth scale."""
    cv2.circle(canvas, (IMG_W // 2, 10), 6, (220, 220, 220), -1)
    draw_depth_scale(canvas)


def draw_ellipse(canvas, cx, cy, rx, ry, angle, brightness, thickness=-1):
    cv2.ellipse(canvas, (cx, cy), (rx, ry), angle, 0, 360,
                (brightness, brightness, brightness), thickness)


def draw_line(canvas, x1, y1, x2, y2, brightness, thickness=1):
    cv2.line(canvas, (x1, y1), (x2, y2),
             (brightness, brightness, brightness), thickness)


def gaussian_blob(canvas, cx, cy, rx, ry, brightness, add=True):
    """Smooth blob (tissue, organ)."""
    Y, X = np.ogrid[:IMG_H, :IMG_W]
    mask = ((X - cx)**2 / rx**2 + (Y - cy)**2 / ry**2) <= 1
    if add:
        canvas[mask] = np.clip(canvas[mask].astype(int) + brightness, 0, 255)
    else:
        canvas[mask] = brightness


def add_posterior_enhancement(canvas, mask, below_y, cx, width=60, strength=30):
    """Bright stripe below fluid (posterior acoustic enhancement artefact)."""
    for y in range(below_y, min(below_y + 60, IMG_H)):
        for x in range(cx - width // 2, cx + width // 2):
            if 0 <= x < IMG_W and mask[y, x]:
                canvas[y, x, :] = np.clip(canvas[y, x, :].astype(int) + strength, 0, 255)


def jitter(val, delta):
    return int(val + random.randint(-delta, delta))


# ──────────────────────────────────────────────────────────────────────────────
# View generators — each returns a completed (H, W, 3) uint8 image
# ──────────────────────────────────────────────────────────────────────────────

# ── LUNG ──────────────────────────────────────────────────────────────────────

def gen_lung_a_lines(_i):
    """Normal lung: horizontal reverberation lines."""
    c = make_canvas()
    apex_x, apex_y = IMG_W // 2, 20
    mask = draw_sector(c, apex_x, apex_y, angle_spread=65, depth=370,
                       base_brightness=70, deep_brightness=20)
    # Bright pleural line
    pleura_y = jitter(110, 8)
    draw_line(c, apex_x - 160, pleura_y, apex_x + 160, pleura_y, 200, 2)
    # A-lines (reverberations at regular intervals below pleura)
    spacing = jitter(55, 5)
    for k in range(1, 5):
        y = pleura_y + k * spacing
        brightness = max(40, 160 - k * 35)
        width = 120 + k * 15
        if y < IMG_H:
            draw_line(c, apex_x - width, y, apex_x + width, y, brightness, 1)
    add_ui_chrome(c)
    return c


def gen_lung_b_lines(_i):
    """B-lines: vertical comet-tail artefacts from pleura."""
    c = make_canvas()
    apex_x, apex_y = IMG_W // 2, 20
    mask = draw_sector(c, apex_x, apex_y, angle_spread=65, depth=370,
                       base_brightness=65, deep_brightness=18)
    pleura_y = jitter(110, 8)
    draw_line(c, apex_x - 160, pleura_y, apex_x + 160, pleura_y, 190, 2)
    n_blines = random.randint(2, 5)
    for _ in range(n_blines):
        bx = jitter(apex_x, 100)
        for y in range(pleura_y, min(IMG_H, pleura_y + 280)):
            fade = max(0, 200 - (y - pleura_y) // 2)
            if mask[y, bx]:
                c[y, bx] = min(255, int(c[y, bx, 0]) + fade)
                if bx > 0 and mask[y, bx - 1]:
                    c[y, bx - 1] = min(255, int(c[y, bx - 1, 0]) + fade // 2)
                if bx < IMG_W - 1 and mask[y, bx + 1]:
                    c[y, bx + 1] = min(255, int(c[y, bx + 1, 0]) + fade // 2)
    add_ui_chrome(c)
    return c


def gen_lung_consolidation(_i):
    """Consolidation: tissue-like echogenicity replacing air."""
    c = make_canvas()
    apex_x, apex_y = IMG_W // 2, 20
    mask = draw_sector(c, apex_x, apex_y, angle_spread=65, depth=380,
                       base_brightness=60, deep_brightness=15)
    pleura_y = jitter(110, 8)
    draw_line(c, apex_x - 160, pleura_y, apex_x + 160, pleura_y, 180, 2)
    # Large tissue-density region below pleura
    cy = jitter(220, 30)
    rx, ry = jitter(110, 20), jitter(80, 15)
    # Fill with liver-like texture
    Y, X = np.ogrid[:IMG_H, :IMG_W]
    region = ((X - apex_x)**2 / rx**2 + (Y - cy)**2 / ry**2) <= 1
    sp = speckle_noise((IMG_H, IMG_W), 0.7)
    tissue = (sp * 110).astype(np.uint8)
    for ch in range(3):
        c[:, :, ch] = np.where(region & mask.astype(bool), tissue, c[:, :, ch])
    # Air bronchograms (bright dots inside consolidation)
    for _ in range(random.randint(4, 8)):
        bx = jitter(apex_x, rx - 20)
        by = jitter(cy, ry - 20)
        cv2.circle(c, (bx, by), random.randint(2, 5), (200, 200, 200), -1)
    add_ui_chrome(c)
    return c


def gen_lung_pleural_effusion(_i):
    """Pleural effusion: anechoic (black) collection above diaphragm."""
    c = make_canvas()
    apex_x, apex_y = IMG_W // 2, 20
    mask = draw_sector(c, apex_x, apex_y, angle_spread=65, depth=390,
                       base_brightness=60, deep_brightness=18)
    # Diaphragm line
    diaphragm_y = jitter(310, 20)
    draw_line(c, apex_x - 180, diaphragm_y, apex_x + 180, diaphragm_y, 200, 3)
    # Anechoic effusion above diaphragm
    eff_top = jitter(180, 20)
    eff_cx  = jitter(apex_x, 20)
    eff_rx, eff_ry = jitter(120, 15), (diaphragm_y - eff_top) // 2
    eff_cy = eff_top + eff_ry
    Y, X = np.ogrid[:IMG_H, :IMG_W]
    fluid = ((X - eff_cx)**2 / eff_rx**2 + (Y - eff_cy)**2 / eff_ry**2) <= 1
    for ch in range(3):
        c[:, :, ch] = np.where(fluid & mask.astype(bool), 8, c[:, :, ch])
    # Posterior enhancement
    add_posterior_enhancement(c, mask, diaphragm_y, eff_cx, width=eff_rx * 2, strength=25)
    add_ui_chrome(c)
    return c


# ── FAST ──────────────────────────────────────────────────────────────────────

def _fast_liver_kidney(c, mask, with_fluid=False):
    """RUQ: liver (grey) + kidney (oval outline) +/- free fluid."""
    apex_x = IMG_W // 2
    # Liver
    lx, ly, lrx, lry = jitter(apex_x - 10, 15), jitter(200, 20), jitter(160, 15), jitter(90, 10)
    Y, X = np.ogrid[:IMG_H, :IMG_W]
    liver = ((X - lx)**2 / lrx**2 + (Y - ly)**2 / lry**2) <= 1
    sp = speckle_noise((IMG_H, IMG_W), 0.55)
    liver_tex = (sp * 95).astype(np.uint8)
    for ch in range(3):
        c[:, :, ch] = np.where(liver & mask.astype(bool), liver_tex, c[:, :, ch])
    cv2.ellipse(c, (lx, ly), (lrx, lry), 0, 0, 360, (160, 160, 160), 2)

    # Kidney
    kx, ky = jitter(apex_x + 80, 20), jitter(260, 20)
    cv2.ellipse(c, (kx, ky), (45, 65), 0, 0, 360, (140, 140, 140), 2)
    cv2.ellipse(c, (kx, ky), (25, 42), 0, 0, 360, (80, 80, 80), -1)
    cv2.ellipse(c, (kx, ky), (12, 22), 0, 0, 360, (170, 170, 170), -1)  # sinus

    if with_fluid:
        # Anechoic stripe at liver-kidney interface
        fluid_cx = (lx + kx) // 2
        fluid_cy = jitter(230, 10)
        cv2.ellipse(c, (fluid_cx, fluid_cy), (40, 18), 0, 0, 360, (10, 10, 10), -1)
        add_posterior_enhancement(c, mask, fluid_cy + 20, fluid_cx, width=60, strength=20)


def gen_fast_ruq_normal(_i):
    c = make_canvas()
    mask = draw_sector(c, IMG_W // 2, 20, angle_spread=70, depth=380)
    _fast_liver_kidney(c, mask, with_fluid=False)
    add_ui_chrome(c)
    return c


def gen_fast_ruq_fluid(_i):
    c = make_canvas()
    mask = draw_sector(c, IMG_W // 2, 20, angle_spread=70, depth=380)
    _fast_liver_kidney(c, mask, with_fluid=True)
    add_ui_chrome(c)
    return c


def gen_fast_luq_normal(_i):
    c = make_canvas()
    apex_x = IMG_W // 2
    mask = draw_sector(c, apex_x, 20, angle_spread=70, depth=380)
    # Spleen
    sx, sy = jitter(apex_x, 20), jitter(190, 20)
    cv2.ellipse(c, (sx, sy), (jitter(90, 10), jitter(70, 8)), 20, 0, 360, (120, 120, 120), 2)
    sp = speckle_noise((IMG_H, IMG_W), 0.5)
    Y, X = np.ogrid[:IMG_H, :IMG_W]
    spleen = ((X - sx)**2 / 90**2 + (Y - sy)**2 / 70**2) <= 1
    spleen_tex = (sp * 100).astype(np.uint8)
    for ch in range(3):
        c[:, :, ch] = np.where(spleen & mask.astype(bool), spleen_tex, c[:, :, ch])
    # Kidney below
    kx, ky = jitter(apex_x + 60, 15), jitter(290, 15)
    cv2.ellipse(c, (kx, ky), (40, 60), 0, 0, 360, (130, 130, 130), 2)
    add_ui_chrome(c)
    return c


def gen_fast_pericardial_normal(_i):
    c = make_canvas()
    mask = draw_sector(c, IMG_W // 2, 20, angle_spread=60, depth=360)
    cx, cy = IMG_W // 2, jitter(230, 20)
    # Pericardium
    cv2.ellipse(c, (cx, cy), (jitter(100, 10), jitter(80, 10)), 0, 0, 360, (170, 170, 170), 2)
    # Ventricles (two chambers)
    cv2.ellipse(c, (jitter(cx - 30, 10), cy), (40, 55), 0, 0, 360, (40, 40, 40), -1)
    cv2.ellipse(c, (jitter(cx + 30, 10), cy), (35, 50), 0, 0, 360, (40, 40, 40), -1)
    draw_line(c, cx, cy - 55, cx, cy + 55, 150, 2)
    add_ui_chrome(c)
    return c


def gen_fast_pericardial_effusion(_i):
    c = make_canvas()
    mask = draw_sector(c, IMG_W // 2, 20, angle_spread=60, depth=360)
    cx, cy = IMG_W // 2, jitter(230, 20)
    # Pericardium
    cv2.ellipse(c, (cx, cy), (jitter(115, 10), jitter(95, 10)), 0, 0, 360, (170, 170, 170), 2)
    # Anechoic effusion ring
    cv2.ellipse(c, (cx, cy), (110, 90), 0, 0, 360, (12, 12, 12), -1)
    # Heart inside effusion
    cv2.ellipse(c, (cx, cy), (85, 68), 0, 0, 360, (100, 100, 100), 2)
    cv2.ellipse(c, (jitter(cx - 25, 8), cy), (35, 48), 0, 0, 360, (38, 38, 38), -1)
    cv2.ellipse(c, (jitter(cx + 25, 8), cy), (30, 42), 0, 0, 360, (38, 38, 38), -1)
    add_ui_chrome(c)
    return c


def gen_fast_pelvic_normal(_i):
    c = make_canvas()
    mask = draw_sector(c, IMG_W // 2, 20, angle_spread=75, depth=385)
    # Bladder (large anechoic)
    bx, by = jitter(IMG_W // 2, 20), jitter(200, 20)
    cv2.ellipse(c, (bx, by), (jitter(100, 15), jitter(75, 10)), 0, 0, 360, (10, 10, 10), -1)
    cv2.ellipse(c, (bx, by), (jitter(100, 15), jitter(75, 10)), 0, 0, 360, (160, 160, 160), 2)
    add_posterior_enhancement(c, mask, by + 80, bx, width=160, strength=30)
    add_ui_chrome(c)
    return c


# ── 2nd / 3rd TRIMESTER ───────────────────────────────────────────────────────

def gen_four_chamber(_i):
    c = make_canvas()
    mask = draw_sector(c, IMG_W // 2, 20, angle_spread=75, depth=370)
    cx, cy = jitter(IMG_W // 2, 20), jitter(220, 20)
    # Pericardium
    cv2.ellipse(c, (cx, cy), (jitter(115, 12), jitter(90, 10)), jitter(15, 10),
                0, 360, (170, 170, 170), 2)
    # Four chambers
    offsets = [(-45, -25), (45, -25), (-40, 30), (40, 30)]
    sizes   = [(50, 40), (45, 38), (48, 38), (42, 35)]
    bright  = [35, 38, 32, 36]
    for (ox, oy), (rx, ry), br in zip(offsets, sizes, bright):
        cv2.ellipse(c, (cx + ox, cy + oy), (rx, ry), 0, 0, 360, (br, br, br), -1)
    # Interventricular septum
    draw_line(c, cx, cy - 30, cx, cy + 35, 160, 2)
    # Atrioventricular septum
    draw_line(c, cx - 80, cy, cx + 80, cy, 150, 2)
    add_ui_chrome(c)
    return c


def gen_bpd(_i):
    """Biparietal diameter: oval skull at level of thalami."""
    c = make_canvas()
    mask = draw_sector(c, IMG_W // 2, 20, angle_spread=70, depth=370)
    cx, cy = jitter(IMG_W // 2, 20), jitter(220, 20)
    # Skull (bright ring)
    rx, ry = jitter(130, 12), jitter(105, 10)
    cv2.ellipse(c, (cx, cy), (rx, ry), jitter(10, 8), 0, 360, (190, 190, 190), 6)
    # Brain tissue inside
    sp = speckle_noise((IMG_H, IMG_W), 0.45)
    brain_tex = (sp * 80).astype(np.uint8)
    Y, X = np.ogrid[:IMG_H, :IMG_W]
    brain = ((X - cx)**2 / (rx - 6)**2 + (Y - cy)**2 / (ry - 6)**2) <= 1
    for ch in range(3):
        c[:, :, ch] = np.where(brain & mask.astype(bool), brain_tex, c[:, :, ch])
    # Thalami (two oval hypoechoic structures)
    for ox in [-22, 22]:
        cv2.ellipse(c, (cx + ox, cy + jitter(5, 5)), (18, 25), 0, 0, 360, (55, 55, 55), -1)
    # Cavum septum pellucidum (midline anechoic)
    cv2.rectangle(c, (cx - 8, cy - 40), (cx + 8, cy - 20), (15, 15, 15), -1)
    # BPD calliper markers
    cv2.line(c, (cx - rx, cy), (cx - rx - 10, cy), (0, 255, 0), 2)
    cv2.line(c, (cx + rx, cy), (cx + rx + 10, cy), (0, 255, 0), 2)
    add_ui_chrome(c)
    return c


def gen_femur(_i):
    """Femur length: long bone shaft."""
    c = make_canvas()
    mask = draw_sector(c, IMG_W // 2, 20, angle_spread=65, depth=360)
    # Femur shaft (bright hyperechoic line with acoustic shadow below)
    angle = jitter(30, 15)
    cx, cy = jitter(IMG_W // 2, 30), jitter(210, 30)
    length = jitter(140, 20)
    rad = np.radians(angle)
    dx, dy = int(length * np.cos(rad)), int(length * np.sin(rad))
    x1, y1 = cx - dx // 2, cy - dy // 2
    x2, y2 = cx + dx // 2, cy + dy // 2
    # Bone shadow
    for offset in range(1, 25):
        alpha = max(0, 200 - offset * 8)
        cv2.line(c, (x1, y1 + offset), (x2, y2 + offset), (alpha // 5, alpha // 5, alpha // 5), 1)
    # Bone itself
    cv2.line(c, (x1, y1), (x2, y2), (220, 220, 220), 4)
    # Calliper
    cv2.line(c, (x1 - 5, y1 - 5), (x1 + 5, y1 + 5), (0, 255, 0), 2)
    cv2.line(c, (x2 - 5, y2 - 5), (x2 + 5, y2 + 5), (0, 255, 0), 2)
    add_ui_chrome(c)
    return c


def gen_abdominal_circumference(_i):
    """AC: round cross-section with stomach bubble."""
    c = make_canvas()
    mask = draw_sector(c, IMG_W // 2, 20, angle_spread=70, depth=375)
    cx, cy = jitter(IMG_W // 2, 20), jitter(215, 20)
    rx, ry = jitter(105, 12), jitter(95, 10)
    # Skin (outer ring)
    cv2.ellipse(c, (cx, cy), (rx, ry), 0, 0, 360, (160, 160, 160), 4)
    # Abdominal tissue
    sp = speckle_noise((IMG_H, IMG_W), 0.48)
    abd_tex = (sp * 85).astype(np.uint8)
    Y, X = np.ogrid[:IMG_H, :IMG_W]
    abd = ((X - cx)**2 / (rx - 4)**2 + (Y - cy)**2 / (ry - 4)**2) <= 1
    for ch in range(3):
        c[:, :, ch] = np.where(abd & mask.astype(bool), abd_tex, c[:, :, ch])
    # Stomach bubble (anechoic)
    sx, sy = jitter(cx - 25, 20), jitter(cy + 10, 15)
    cv2.ellipse(c, (sx, sy), (jitter(22, 5), jitter(18, 4)), 0, 0, 360, (12, 12, 12), -1)
    cv2.ellipse(c, (sx, sy), (jitter(22, 5), jitter(18, 4)), 0, 0, 360, (140, 140, 140), 1)
    # Portal vein (hyperechoic walls)
    pvx, pvy = jitter(cx + 15, 15), jitter(cy - 10, 10)
    cv2.ellipse(c, (pvx, pvy), (14, 10), 0, 0, 360, (170, 170, 170), 2)
    add_ui_chrome(c)
    return c


def gen_spine(_i):
    """Fetal spine: two parallel echogenic lines (laminae)."""
    c = make_canvas()
    mask = draw_sector(c, IMG_W // 2, 20, angle_spread=60, depth=355)
    angle = jitter(10, 20)
    cx, cy = IMG_W // 2, jitter(215, 20)
    length = jitter(200, 20)
    gap = jitter(14, 3)
    rad = np.radians(angle)
    dx, dy = int(length * np.cos(rad)), int(length * np.sin(rad))
    # Two laminae
    for offset in [-gap // 2, gap // 2]:
        ox = int(offset * np.sin(rad))
        oy = int(offset * np.cos(rad))
        x1 = cx - dx // 2 + ox
        y1 = cy - dy // 2 + oy
        x2 = cx + dx // 2 + ox
        y2 = cy + dy // 2 + oy
        cv2.line(c, (x1, y1), (x2, y2), (200, 200, 200), 3)
        # Vertebral bodies between lines (short transverse echoes)
    for i in range(-4, 5):
        t = i / 5
        mid_x = int(cx + t * dx)
        mid_y = int(cy + t * dy)
        nrm_x = int(-np.sin(rad) * gap)
        nrm_y = int(np.cos(rad) * gap)
        cv2.line(c, (mid_x - nrm_x, mid_y - nrm_y),
                 (mid_x + nrm_x, mid_y + nrm_y), (160, 160, 160), 2)
    add_ui_chrome(c)
    return c


def gen_placenta(_i):
    """Placenta position: homogeneous granular texture on uterine wall."""
    c = make_canvas()
    mask = draw_sector(c, IMG_W // 2, 20, angle_spread=75, depth=390)
    side = random.choice(["anterior", "posterior", "fundal"])
    cx, cy = IMG_W // 2, 220
    # Uterine wall
    cv2.ellipse(c, (cx, cy), (160, 130), 0, 0, 360, (140, 140, 140), 3)
    sp = speckle_noise((IMG_H, IMG_W), 0.52)
    # Placental tissue (grainy, medium echogenicity)
    placenta_tex = np.clip((sp * 115 + 20), 0, 255).astype(np.uint8)
    Y, X = np.ogrid[:IMG_H, :IMG_W]
    if side == "anterior":
        plac = ((X - cx)**2 / 150**2 + (Y - (cy - 90))**2 / 35**2) <= 1
    elif side == "posterior":
        plac = ((X - cx)**2 / 150**2 + (Y - (cy + 95))**2 / 35**2) <= 1
    else:
        plac = ((X - (cx - 130))**2 / 35**2 + (Y - cy)**2 / 110**2) <= 1
    for ch in range(3):
        c[:, :, ch] = np.where(plac & mask.astype(bool), placenta_tex, c[:, :, ch])
    # Amniotic fluid (anechoic)
    amn = ((X - cx)**2 / 130**2 + (Y - cy)**2 / 105**2) <= 1
    fluid = amn & ~plac
    for ch in range(3):
        c[:, :, ch] = np.where(fluid & mask.astype(bool), 10, c[:, :, ch])
    add_ui_chrome(c)
    return c


def gen_presentation(_i):
    """Fetal presentation: head (cephalic) or buttocks (breech) in lower uterus."""
    c = make_canvas()
    mask = draw_sector(c, IMG_W // 2, 20, angle_spread=75, depth=390)
    presentation = random.choice(["cephalic", "breech"])
    cx, cy = jitter(IMG_W // 2, 20), jitter(230, 20)
    if presentation == "cephalic":
        # Round skull
        cv2.ellipse(c, (cx, cy), (jitter(90, 10), jitter(75, 8)), jitter(5, 10),
                    0, 360, (190, 190, 190), 5)
        sp = speckle_noise((IMG_H, IMG_W), 0.42)
        brain_tex = (sp * 75).astype(np.uint8)
        Y, X = np.ogrid[:IMG_H, :IMG_W]
        brain = ((X - cx)**2 / 84**2 + (Y - cy)**2 / 69**2) <= 1
        for ch in range(3):
            c[:, :, ch] = np.where(brain & mask.astype(bool), brain_tex, c[:, :, ch])
    else:
        # Breech: two legs / buttocks shape
        for ox in [-35, 35]:
            cv2.ellipse(c, (cx + ox, cy), (28, 45), 0, 0, 360, (130, 130, 130), -1)
        cv2.ellipse(c, (cx, cy - 30), (55, 35), 0, 0, 360, (120, 120, 120), -1)
    # Cervix indicator
    draw_line(c, cx - 20, cy + 120, cx + 20, cy + 120, 170, 2)
    add_ui_chrome(c)
    return c


# ──────────────────────────────────────────────────────────────────────────────
# View registry
# ──────────────────────────────────────────────────────────────────────────────

VIEW_GENERATORS = {
    "lung": {
        "A_lines":          gen_lung_a_lines,
        "B_lines":          gen_lung_b_lines,
        "consolidation":    gen_lung_consolidation,
        "pleural_effusion": gen_lung_pleural_effusion,
    },
    "fast": {
        "RUQ_normal":               gen_fast_ruq_normal,
        "RUQ_fluid":                gen_fast_ruq_fluid,
        "LUQ_normal":               gen_fast_luq_normal,
        "pericardial_normal":       gen_fast_pericardial_normal,
        "pericardial_effusion":     gen_fast_pericardial_effusion,
        "pelvic_normal":            gen_fast_pelvic_normal,
    },
    "second_trimester": {
        "four_chamber":              gen_four_chamber,
        "biparietal_diameter":       gen_bpd,
        "femur_length":              gen_femur,
        "abdominal_circumference":   gen_abdominal_circumference,
        "spine":                     gen_spine,
        "placenta_location":         gen_placenta,
        "third_trimester_presentation": gen_presentation,
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    total = 0
    for domain, views in VIEW_GENERATORS.items():
        domain_dir = SYNTH / domain
        print(f"\n[{domain}]")
        for view_name, gen_fn in views.items():
            view_dir = domain_dir / view_name
            view_dir.mkdir(parents=True, exist_ok=True)

            # Remove old placeholders
            for old in view_dir.glob("*placeholder*.jpg"):
                old.unlink()

            for i in range(N_PER_VIEW):
                img = gen_fn(i)
                out_path = view_dir / f"{view_name}_{i:04d}.jpg"
                cv2.imwrite(str(out_path), img)

            count = len(list(view_dir.glob("*.jpg")))
            total += count
            print(f"  {view_name}: {count} images")

    print(f"\nDone. {total} synthetic ultrasound images generated in {SYNTH}")
    print("Run reorganize_dataset.py to rebuild the training dataset.")


if __name__ == "__main__":
    main()
