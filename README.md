# 🫀 Fetal Echo Quality Classifier

A clinical ultrasound quality assessment system for first trimester fetal echocardiography scans using EfficientNet-B3, EigenCAM explainability, FastAPI, and synthetic ultrasound augmentation.

The system classifies fetal ultrasound scan quality into six categories and generates explainability heatmaps alongside clinical quality guidance.

---

# ✨ Core Features

- 6 class fetal ultrasound quality classification
- EfficientNet-B3 fine tuned on ultrasound data
- EigenCAM explainability overlays
- Synthetic ultrasound augmentation pipeline
- FastAPI inference backend
- Clinical terminology normalization
- Dockerized deployment pipeline

---

# 📚 Dataset & Preprocessing

## Ultrasound Quality Classes

```text
good
blurry
too_dark
low_contrast
noisy
angled
```

These labels correspond to clinically meaningful scan quality issues:

| Internal Label | Clinical Label |
|---|---|
| blurry | Motion / Focus Artefact |
| too_dark | Insufficient Gain |
| low_contrast | Low Tissue Contrast |
| noisy | Noise Artefact |
| angled | Incorrect Probe Angle |
| good | Diagnostic Quality |

---

## Dataset Utilities

### Manual Annotation Pipeline
```bash
label_quality.py
```

Interactive annotation tool used for manually labeling fetal ultrasound scans.

---

### Dataset Organization
```bash
reorganize_dataset.py
```

Automatically organizes labeled images into:
- train/
- valid/
- test/

directory structures for model training.

---

### Synthetic Ultrasound Augmentation
```bash
generate_synthetic.py
generate_ultrasound_images.py
```

Generated synthetic ultrasound images to augment underrepresented classes and simulate realistic acquisition defects.

Synthetic transformations included:
- Gaussian blur
- Contrast degradation
- Brightness reduction
- Noise injection
- Probe angle distortion

---

## Metadata Files

```text
labels.csv
synthetic_metadata.csv
```

- `labels.csv` stores scan level annotations
- `synthetic_metadata.csv` stores augmentation generation metadata

---

# 🤖 Model Architecture

## Base Architecture

EfficientNet-B3 pretrained on ImageNet and fine tuned for fetal ultrasound quality classification.

Selected because of:
- Strong performance on medical imaging tasks
- Efficient parameter scaling
- Effective feature extraction for texture heavy ultrasound scans
- Lower inference cost compared to larger CNN architectures

---

## Training Workflow

### Phase 1
Train classification head while freezing pretrained backbone weights.

### Phase 2
Unfreeze final EfficientNet feature blocks and fine tune the network end to end.

---

## Training Scripts

```bash
train_local.py
train_colab.py
train_colab.ipynb
```

Supports:
- Local GPU training
- Google Colab GPU training
- Validation tracking
- Checkpoint saving

---

## Model Output

```text
model_weights.pth
```

Each inference generates:
- Predicted quality classification
- Confidence score
- Full probability distribution across all six classes
- EigenCAM activation heatmap
- Clinical guidance text

---

# 🔥 Explainability with EigenCAM

EigenCAM was integrated to provide visual explainability for ultrasound quality predictions.

The activation map highlights regions of the fetal ultrasound scan that contributed most strongly to the model's classification decision.

---

## Hook Layer

```python
model.features[-2]
```

This captures high level spatial feature representations before pooling and classification.

---

## Explainability Output

Each prediction generates:
- Original ultrasound scan
- EigenCAM activation heatmap
- Overlay visualization highlighting activated regions
- Confidence distribution
- Clinical interpretation guidance

Heatmaps are returned as base64 encoded images through the API pipeline.

---

# ⚡ FastAPI Backend

Backend located in:

```text
hf_space/
```

---

# 🩺 API Endpoint

## Image Classification

```http
POST /predict
```

### Supported Formats
- PNG
- JPG
- BMP

### Returns
- Predicted quality class
- Confidence score
- Full class probabilities
- Original image
- EigenCAM heatmap
- Clinical recommendation text

---

# 🛠️ Backend Stack

- FastAPI
- PyTorch
- timm
- OpenCV
- NumPy
- EigenCAM
- Docker
- Hugging Face Spaces

---

# ⚙️ Deployment Files

```text
main.py
requirements.txt
Dockerfile
README.md
```

---

# 🔧 Engineering Fixes

## EigenCAM Compatibility

```python
inplace=False
```

Applied to Dropout layers to prevent backward hook conflicts during activation extraction.

---

## Cross Origin Support

```python
CORSMiddleware
```

Enabled frontend and backend communication across deployment origins.

---

# 📂 Repository Structure

```text
fetal-echo-quality-classifier/
│
├── hf_space/
│   ├── main.py
│   ├── requirements.txt
│   ├── Dockerfile
│   └── README.md
│
├── website/
│
├── train_local.py
├── train_colab.py
├── train_colab.ipynb
├── generate_synthetic.py
├── generate_ultrasound_images.py
├── reorganize_dataset.py
├── label_quality.py
├── labels.csv
├── synthetic_metadata.csv
└── README.md
```

---

# 🚀 Installation

## Clone Repository

```bash
git clone https://github.com/SamhitaK10/fetal-echo-quality-classifier.git
cd fetal-echo-quality-classifier
```

---

# ⚙️ Backend Setup

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run FastAPI Server

```bash
uvicorn main:app --reload
```

---

# 🧪 Training

## Local Training

```bash
python train_local.py
```

---

## Google Colab Training

Open:

```text
train_colab.ipynb
```

---

# 📈 Prediction Response Structure

Each API response contains both classification and explainability outputs for the uploaded ultrasound scan.

## Example Response

```json
{
  "label": "Insufficient Gain",
  "confidence": 0.97,
  "probabilities": {
    "good": 0.01,
    "too_dark": 0.97
  }
}
```

### Response Components

| Field | Description |
|---|---|
| label | Predicted ultrasound quality class |
| confidence | Model confidence for the predicted class |
| probabilities | Probability distribution across all classes |
| original_image | Base64 encoded uploaded scan |
| heatmap | Base64 encoded EigenCAM activation map |
| recommendation | Clinical guidance text associated with the prediction |

The backend returns both the original ultrasound scan and the corresponding EigenCAM visualization so activated anatomical regions can be inspected alongside the classification output.

---

# 🤖 AI Policy

The use of AI tools is allowed and encouraged on this project. The following AI tools were used:

| Tool | Where Used |
|---|---|
| **Claude Code (Claude Sonnet 4.6 — Anthropic)** | React website (all components), FastAPI backend refinements, EigenCAM integration, CORS setup, Dockerfile, synthetic video dataset generator, deployment wiring, GitHub setup |
| **Google Colab (Gemini suggestions)** | Notebook formatting and training loop guidance |

All AI-generated code was reviewed, tested, and validated by the author. Model architecture decisions, dataset design, class definitions, and clinical guidance text were authored by the project owner.

---

# 📜 License

MIT License
