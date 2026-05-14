# ============================================================
# Ultrasound Quality Classifier — EfficientNet-B3
# Local VS Code version (adapted from train_colab.py)
# ============================================================

import os
import time
import copy
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — saves plots to disk
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

from sklearn.metrics import classification_report, confusion_matrix

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ── Paths ────────────────────────────────────────────────────
BASE     = Path(__file__).parent
DATA_DIR = BASE / "dataset"
OUT_DIR  = BASE / "outputs"
OUT_DIR.mkdir(exist_ok=True)

# Verify dataset structure
for split in ["train", "valid", "test"]:
    split_dir = DATA_DIR / split
    classes   = sorted(p.name for p in split_dir.iterdir() if p.is_dir())
    counts    = [len(list((split_dir / c).glob("*.jpg"))) for c in classes]
    print(f"{split}: {dict(zip(classes, counts))}")

# ── Config ───────────────────────────────────────────────────
CFG = {
    "img_size":        300,
    "batch_size":      32,
    "num_workers":     0,    # 0 required on Windows without multiprocessing guard
    "phase1_epochs":   5,    # frozen backbone
    "phase2_epochs":   20,   # fine-tune
    "phase1_lr":       1e-3,
    "phase2_lr":       1e-4,
    "weight_decay":    1e-4,
    "label_smoothing": 0.1,
    "checkpoint_path": str(OUT_DIR / "ultrasound_efficientnet_b3.pth"),
}

GUIDANCE = {
    "good":         "Image quality is good. Proceed with assessment.",
    "blurry":       "Image is blurry. Apply more gel and hold the probe still. Ask the patient to hold their breath.",
    "too_dark":     "Image is too dark. Increase the gain setting on the machine.",
    "low_contrast": "Low contrast detected. Re-position the probe at a slightly different angle.",
    "noisy":        "Image is noisy. Apply fresh gel and reduce probe pressure slightly.",
    "angled":       "Probe angle is off. Rotate the probe until the structure is centred on screen.",
}

# ── Data loaders ─────────────────────────────────────────────
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.Resize((CFG["img_size"], CFG["img_size"])),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

val_tf = transforms.Compose([
    transforms.Resize((CFG["img_size"], CFG["img_size"])),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

train_ds = datasets.ImageFolder(DATA_DIR / "train", transform=train_tf)
valid_ds  = datasets.ImageFolder(DATA_DIR / "valid", transform=val_tf)
test_ds   = datasets.ImageFolder(DATA_DIR / "test",  transform=val_tf)

train_dl = DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True,
                      num_workers=CFG["num_workers"], pin_memory=False)
valid_dl  = DataLoader(valid_ds, batch_size=CFG["batch_size"], shuffle=False,
                      num_workers=CFG["num_workers"], pin_memory=False)
test_dl   = DataLoader(test_ds,  batch_size=CFG["batch_size"], shuffle=False,
                      num_workers=CFG["num_workers"], pin_memory=False)

CLASS_NAMES = train_ds.classes
NUM_CLASSES  = len(CLASS_NAMES)
print("Classes:", CLASS_NAMES)
print(f"Train: {len(train_ds)} | Valid: {len(valid_ds)} | Test: {len(test_ds)}")

# ── Model ────────────────────────────────────────────────────
def build_model(num_classes: int) -> nn.Module:
    model = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model


model = build_model(NUM_CLASSES).to(device)

with torch.no_grad():
    dummy = torch.randn(2, 3, CFG["img_size"], CFG["img_size"]).to(device)
    out   = model(dummy)
    print("Output shape:", out.shape)

# ── Training helpers ─────────────────────────────────────────
def freeze_backbone(model: nn.Module):
    for name, param in model.named_parameters():
        param.requires_grad = "classifier" in name
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params (head only): {trainable:,}")


def unfreeze_last_blocks(model: nn.Module, n_blocks: int = 3):
    total_blocks = 8
    unfreeze_from = total_blocks - n_blocks

    for param in model.parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        block_idx = None
        parts = name.split(".")
        if parts[0] == "features" and len(parts) > 1:
            try:
                block_idx = int(parts[1])
            except ValueError:
                pass
        if block_idx is not None and block_idx >= unfreeze_from:
            param.requires_grad = True
        if "classifier" in name:
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params (last {n_blocks} blocks + head): {trainable:,}")


def run_epoch(model, loader, criterion, optimizer=None, phase="train"):
    is_train = phase == "train"
    model.train() if is_train else model.eval()

    total_loss, total_correct, total_samples = 0.0, 0, 0

    with torch.set_grad_enabled(is_train):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss   = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            preds          = logits.argmax(dim=1)
            total_loss    += loss.item() * images.size(0)
            total_correct += (preds == labels).sum().item()
            total_samples += images.size(0)

    return total_loss / total_samples, total_correct / total_samples


def train_phase(model, train_dl, valid_dl, optimizer, scheduler, criterion,
                epochs: int, phase_name: str):
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_dl, criterion, optimizer, "train")
        va_loss, va_acc = run_epoch(model, valid_dl, criterion, phase="val")
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        print(f"[{phase_name}] Epoch {epoch:>3}/{epochs} | "
              f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f} | "
              f"{time.time()-t0:.0f}s")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_weights = copy.deepcopy(model.state_dict())
            print(f"  --> New best val acc: {best_val_acc:.4f}")

    model.load_state_dict(best_weights)
    print(f"\n{phase_name} complete. Best val acc: {best_val_acc:.4f}\n")
    return history


# ── Phase 1: train head only ─────────────────────────────────
criterion = nn.CrossEntropyLoss(label_smoothing=CFG["label_smoothing"])

freeze_backbone(model)
optimizer1 = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=CFG["phase1_lr"], weight_decay=CFG["weight_decay"]
)
scheduler1 = CosineAnnealingLR(optimizer1, T_max=CFG["phase1_epochs"], eta_min=1e-5)

history1 = train_phase(
    model, train_dl, valid_dl,
    optimizer1, scheduler1, criterion,
    epochs=CFG["phase1_epochs"],
    phase_name="Phase 1 (head)"
)

# ── Phase 2: fine-tune last blocks ───────────────────────────
unfreeze_last_blocks(model, n_blocks=3)
optimizer2 = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=CFG["phase2_lr"], weight_decay=CFG["weight_decay"]
)
scheduler2 = CosineAnnealingLR(optimizer2, T_max=CFG["phase2_epochs"], eta_min=1e-6)

history2 = train_phase(
    model, train_dl, valid_dl,
    optimizer2, scheduler2, criterion,
    epochs=CFG["phase2_epochs"],
    phase_name="Phase 2 (fine-tune)"
)

torch.save({
    "model_state_dict": model.state_dict(),
    "class_names":      CLASS_NAMES,
    "cfg":              CFG,
}, CFG["checkpoint_path"])
print("Model saved to:", CFG["checkpoint_path"])

# ── Plot training curves ──────────────────────────────────────
def plot_history(h1, h2):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    train_loss = h1["train_loss"] + h2["train_loss"]
    val_loss   = h1["val_loss"]   + h2["val_loss"]
    train_acc  = h1["train_acc"]  + h2["train_acc"]
    val_acc    = h1["val_acc"]    + h2["val_acc"]
    epochs     = range(1, len(train_loss) + 1)
    phase_split = len(h1["train_loss"])

    for ax, (tr, va, title) in zip(axes, [
        (train_loss, val_loss, "Loss"),
        (train_acc,  val_acc,  "Accuracy"),
    ]):
        ax.plot(epochs, tr, label="Train")
        ax.plot(epochs, va, label="Validation")
        ax.axvline(phase_split + 0.5, color="gray", linestyle="--",
                   alpha=0.6, label="Fine-tune starts")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path = OUT_DIR / "training_curves.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print("Saved:", out_path)


plot_history(history1, history2)

# ── Evaluate on test set ──────────────────────────────────────
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for images, labels in test_dl:
        images = images.to(device)
        preds  = model(images).argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)

print("=" * 60)
print("TEST SET RESULTS")
print("=" * 60)
print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

cm = confusion_matrix(all_labels, all_preds)
fig, ax = plt.subplots(figsize=(8, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("Confusion Matrix — Test Set")
plt.tight_layout()
cm_path = OUT_DIR / "confusion_matrix.png"
plt.savefig(cm_path, dpi=150)
plt.close()
print("Saved:", cm_path)

# ── Inference demo ────────────────────────────────────────────
def predict(image_path: str, model: nn.Module) -> dict:
    img    = Image.open(image_path).convert("RGB")
    tensor = val_tf(img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

    pred_idx   = probs.argmax()
    pred_label = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])

    return {
        "quality_label": pred_label,
        "confidence":    round(confidence, 3),
        "guidance_text": GUIDANCE[pred_label],
        "all_probs":     {cls: round(float(p), 3)
                          for cls, p in zip(CLASS_NAMES, probs)},
    }


sample_path = str(next((DATA_DIR / "test" / "blurry").glob("*.jpg")))
result = predict(sample_path, model)

print("\n--- Inference Demo ---")
print(f"Image:         {Path(sample_path).name}")
print(f"Quality label: {result['quality_label']}")
print(f"Confidence:    {result['confidence']:.1%}")
print(f"Guidance text: {result['guidance_text']}")
print("\nAll class probabilities:")
for cls, prob in sorted(result["all_probs"].items(), key=lambda x: -x[1]):
    bar = "#" * int(prob * 30)
    print(f"  {cls:<15} {prob:.3f}  {bar}")
