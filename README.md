# 🫀 Fetal Echo Quality Classifier

A clinical ultrasound quality assessment system for first trimester fetal echocardiography scans using EfficientNet-B3, EigenCAM explainability, FastAPI, and synthetic ultrasound augmentation.

The system classifies fetal ultrasound scan quality into six categories and generates explainability heatmaps alongside clinical quality guidance.

This project is a prototype training and image quality control tool. It is not intended for diagnosis or clinical decision making.

---

## 📌 Project Overview

Fetal echocardiography is used to evaluate fetal heart structures during pregnancy. In first trimester imaging, scan quality is especially important because the fetal heart is small, moving, and difficult to capture clearly.

Poor scan quality makes clinical review harder and reduces the reliability of downstream AI systems. This project focuses on the step before diagnosis: checking whether a scan is clear enough for review, training, or further analysis.

Fetal Echo Quality Classifier helps classify ultrasound scan quality into six categories:

- Diagnostic Quality
- Motion / Focus Artifact
- Insufficient Gain
- Low Tissue Contrast
- Noise Artifact
- Incorrect Probe Angle

The system returns:

- Predicted quality class
- Confidence score
- Full probability distribution
- EigenCAM heatmap
- Clinical guidance text
- Downloadable report
- Annotated video output for ultrasound clips

---

## 🎯 Motivation

The main bottleneck this project addresses is image quality assessment in fetal echocardiography.

In fetal ultrasound, a scan may look usable at first glance but still contain quality issues such as blur, low contrast, poor gain, acoustic noise, or incorrect probe angle. These problems matter because they affect whether the image is useful for training, review, or downstream AI analysis.

This project was inspired by three needs:

- Students and early career ultrasound learners need feedback on scan quality.
- Rural or lower resource clinics may have limited access to specialized fetal echo expertise.
- AI systems in medicine depend on input image quality before any diagnostic model becomes useful.

The goal is not to diagnose fetal heart disease. The goal is to help users assess whether the image quality is strong enough for meaningful review.

---

## ✨ Core Features

- 6 class fetal ultrasound quality classification
- EfficientNet-B3 fine tuned on ultrasound data
- EigenCAM explainability overlays
- Synthetic ultrasound augmentation pipeline
- Manual annotation workflow
- FastAPI inference backend
- Clinical terminology normalization
- Image upload and prediction
- Video upload with frame level quality assessment
- Downloadable report with prediction, confidence, probabilities, guidance, and heatmap
- Dockerized deployment pipeline
- Reproducible training and inference scripts

---

## 🧭 Project Track

This project fits the following tracks:

- Domain specific idea, healthcare AI
- Application / Product
- Research component through model training, dataset creation, evaluation, and explainability

---

## 📚 Dataset and Preprocessing

### Dataset Source

The dataset was built from publicly available and open source fetal echocardiography ultrasound images, with a focus on first trimester fetal echo examples.

The dataset was used for prototype development and model experimentation. It was not used for clinical validation.

After collecting images, I standardized the dataset by:

- Filtering unusable or irrelevant samples
- Resizing images for model input
- Organizing images into labeled quality classes
- Splitting images into training, validation, and testing folders
- Tracking scan level labels in metadata files
- Adding synthetic ultrasound style examples for underrepresented classes

---

## Ultrasound Quality Classes

The model uses internal labels for training and maps them to clearer clinical labels in the interface.

```text
good
blurry
too_dark
low_contrast
noisy
angled
```

| Internal Label | Clinical Label | Meaning |
|---|---|---|
| good | Diagnostic Quality | The fetal cardiac structures are clear enough for review |
| blurry | Motion / Focus Artifact | The image is blurred because of fetal motion, probe motion, or poor focus |
| too_dark | Insufficient Gain | The image is too dark, so important structures are harder to see |
| low_contrast | Low Tissue Contrast | Tissue boundaries are not separated clearly enough |
| noisy | Noise Artifact | Speckle, shadowing, or random interference reduces image clarity |
| angled | Incorrect Probe Angle | The scan view is not aligned well, so target heart structures are missing, distorted, or harder to evaluate |

---

## Dataset Utilities

### Manual Annotation Pipeline

```bash
label_quality.py
```

Interactive annotation tool used for manually labeling fetal ultrasound scans into six quality categories.

The annotation workflow helped create scan level labels for training and validation.

### Dataset Organization

```bash
reorganize_dataset.py
```

Automatically organizes labeled images into:

```text
train/
valid/
test/
```

directory structures for model training and evaluation.

### Synthetic Ultrasound Augmentation

```bash
generate_synthetic.py
generate_ultrasound_images.py
```

Generated synthetic ultrasound images to augment underrepresented classes and simulate realistic acquisition defects.

Synthetic transformations included:

- Gaussian blur for motion or focus artifact
- Contrast degradation for low tissue contrast
- Brightness reduction for insufficient gain
- Noise injection for noise artifact
- Probe angle distortion for incorrect probe angle

This helped improve class balance and gave the model more examples of low quality scan patterns.

### Metadata Files

```text
labels.csv
synthetic_metadata.csv
```

- `labels.csv` stores scan level annotations
- `synthetic_metadata.csv` stores augmentation generation metadata

---

## 🤖 Model Architecture

### Base Architecture

EfficientNet-B3 pretrained on ImageNet and fine tuned for fetal ultrasound quality classification.

Selected because of:

- Strong performance on image classification tasks
- Efficient parameter scaling
- Effective feature extraction for texture heavy ultrasound scans
- Lower inference cost compared to larger CNN architectures

### Training Workflow

#### Phase 1

Train the classification head while freezing pretrained backbone weights.

#### Phase 2

Unfreeze final EfficientNet feature blocks and fine tune the network end to end.

### Training Scripts

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

## 📈 Evaluation and Evidence

On the validation set, the model reached approximately 90% accuracy.

I also reviewed class probabilities and confusion matrix behavior to check whether the model distinguished between the six ultrasound quality categories instead of only predicting the most common class.

The evaluation focused on:

- Overall validation accuracy
- Per class probability outputs
- Confusion matrix behavior
- Whether low quality classes were being confused with each other
- Whether the model overpredicted the majority class
- Qualitative review of EigenCAM heatmaps

Because the dataset is limited and includes synthetic augmentation, this result should be interpreted as prototype validation, not clinical validation.

Future evaluation should compare model predictions against expert clinician quality ratings on real hospital fetal echocardiography scans.

---

## Model Output

```text
model_weights.pth
```

Each image inference generates:

- Predicted quality classification
- Confidence score
- Full probability distribution across all six classes
- EigenCAM activation heatmap
- Clinical guidance text

---

## 🔥 Explainability with EigenCAM

EigenCAM was integrated to provide visual explainability for ultrasound quality predictions.

The activation map highlights regions of the fetal ultrasound scan that contributed most strongly to the model's classification decision.

This helps users inspect whether the model focused on the fetal heart region or on irrelevant artifacts.

### Hook Layer

```python
model.features[-2]
```

This layer was used for EigenCAM because it preserves spatial image features before the model compresses them for final classification. This lets the system generate a heatmap showing which scan regions influenced the quality prediction.

### Explainability Output

Each prediction generates:

- Original ultrasound scan
- EigenCAM activation heatmap
- Overlay visualization highlighting activated regions
- Confidence distribution
- Clinical interpretation guidance

Heatmaps are returned as base64 encoded images through the API pipeline.

---

## ⚡ Product Architecture

The product has three main layers:

1. Frontend interface for uploading scans and viewing results
2. FastAPI backend for preprocessing, inference, and response formatting
3. PyTorch model pipeline for classification and EigenCAM heatmap generation

The user uploads an ultrasound image or video clip. The backend preprocesses the input, runs the trained EfficientNet-B3 model, generates probabilities and explainability outputs, and returns the results to the frontend.

---

## ⚡ FastAPI Backend

Backend located in:

```text
hf_space/
```

The backend handles:

- Image preprocessing
- Model inference
- Probability generation
- EigenCAM heatmap creation
- Clinical guidance formatting
- Base64 image encoding
- Frontend API responses

---

## 🩺 API Endpoints

### Image Classification

```http
POST /predict
```

### Supported Formats

- PNG
- JPG
- JPEG
- BMP

### Returns

- Predicted quality class
- Confidence score
- Full class probabilities
- Original image
- EigenCAM heatmap
- Clinical recommendation text

### Video Classification

The system also supports ultrasound video upload.

The backend samples frames from the uploaded ultrasound clip, runs quality classification on each sampled frame, overlays the predicted quality label and EigenCAM heatmap, and returns an annotated video with frame level quality feedback.

This helps users inspect how image quality changes across an ultrasound clip.

---

## 📄 Report Download

The frontend allows users to download a report summarizing the scan assessment.

Each report includes:

- Uploaded scan
- Predicted quality class
- Confidence score
- Probability distribution across all six classes
- Clinical guidance text
- EigenCAM heatmap

This makes the output easier to save, review, and share for training or documentation.

---

## 🛠️ Backend Stack

- FastAPI
- PyTorch
- timm
- OpenCV
- NumPy
- EigenCAM
- Docker
- Hugging Face Spaces

---

## ⚙️ Deployment Files

```text
main.py
requirements.txt
Dockerfile
README.md
```

---

## 🔧 Engineering Fixes

### EigenCAM Compatibility

```python
inplace=False
```

Applied to Dropout layers to prevent backward hook conflicts during activation extraction.

### Cross Origin Support

```python
CORSMiddleware
```

Enabled frontend and backend communication across deployment origins.

---

## 📂 Repository Structure

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

## 🚀 Installation

### Clone Repository

```bash
git clone https://github.com/SamhitaK10/fetal-echo-quality-classifier.git
cd fetal-echo-quality-classifier
```

### Backend Setup

```bash
pip install -r requirements.txt
```

### Run FastAPI Server

```bash
uvicorn main:app --reload
```

---

## 🧪 Training

### Local Training

```bash
python train_local.py
```

### Google Colab Training

Open:

```text
train_colab.ipynb
```

---

## ▶️ Usage Instructions

### Image Prediction

1. Start the FastAPI backend.
2. Open the frontend interface.
3. Upload a fetal echocardiography ultrasound image.
4. Review the predicted quality class, confidence score, probability distribution, EigenCAM heatmap, and guidance text.
5. Download the report for later review or sharing.

### Video Prediction

1. Upload an ultrasound video clip.
2. The backend samples frames from the video.
3. The model predicts frame level quality labels.
4. The system returns an annotated video with quality labels and heatmap overlays.

---

## 📈 Prediction Response Structure

Each API response contains both classification and explainability outputs for the uploaded ultrasound scan.

### Example Response

```json
{
  "label": "Insufficient Gain",
  "confidence": 0.97,
  "probabilities": {
    "good": 0.01,
    "blurry": 0.00,
    "too_dark": 0.97,
    "low_contrast": 0.01,
    "noisy": 0.01,
    "angled": 0.00
  },
  "recommendation": "The scan appears under-gained. Increasing gain or improving acquisition settings may improve visibility."
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

The backend returns both the original ultrasound scan and the corresponding EigenCAM visualization so activated anatomical regions are inspected alongside the classification output.

---

## ✅ Use Cases and Impact

### Ultrasound Training

Students and early career ultrasound learners use the tool to learn what makes a fetal echo image usable or low quality.

Users receive feedback about image quality issues such as:

- Gain
- Contrast
- Noise
- Motion
- Focus
- Probe angle

### Image Quality Control

Clinics in rural or lower resource settings use the tool as a basic image quality check before review or downstream analysis.

### Downstream AI Reliability

Medical AI systems depend on input quality. This project helps flag low quality inputs before they move into downstream analysis pipelines.

---

## 🌍 Value to Society

Fetal Echo Quality Classifier supports earlier, clearer feedback on scan quality.

Potential impact includes:

- Helping students learn ultrasound quality patterns
- Supporting rural clinics with limited specialist access
- Reducing unusable scans before expert review
- Improving the quality of data used in downstream AI systems
- Encouraging safer AI workflows by separating image quality assessment from diagnosis

---

## ⚠️ Limitations

- This project is not a diagnostic tool.
- The model was trained on a limited prototype dataset.
- The dataset includes synthetic augmentation, which does not fully replace real clinical variation.
- Validation accuracy should not be interpreted as clinical performance.
- The system should be evaluated on clinician labeled hospital scans before any real clinical use.
- Performance may vary across ultrasound machines, gestational ages, probe types, and acquisition settings.
- Publicly available images may not represent the full range of clinical scan quality.

---

## 🔮 Future Work

Future improvements include:

- Collecting real low quality fetal echocardiography scans through a clinical partner
- Comparing model outputs against expert clinician quality ratings
- Expanding the dataset across machines, patients, gestational ages, and acquisition settings
- Improving video analysis by scoring scan quality over time
- Identifying the clearest frames within an ultrasound clip
- Flagging time ranges where scan quality drops
- Testing whether the tool improves training outcomes for students and early career ultrasound users
- Adding stronger model evaluation with external validation data

---

## 🤖 AI Usage

AI assisted development tools were used throughout the project for implementation support.

| Tool | Usage |
|---|---|
| Claude Code (Anthropic Sonnet) | Frontend development, FastAPI backend implementation, EigenCAM integration, CORS configuration, Docker setup, deployment support, and synthetic medical image generation |

---

## 📚 Citations and Acknowledgements

- PyTorch: https://pytorch.org/
- timm EfficientNet-B3: https://github.com/huggingface/pytorch-image-models
- FastAPI: https://fastapi.tiangolo.com/
- OpenCV: https://opencv.org/
- NumPy: https://numpy.org/
- Hugging Face Spaces: https://huggingface.co/spaces
- Fetal echocardiography datset: https://figshare.com/articles/figure/First_Trimester_Fetal_Echocardiography_Data_Set_for_Classification/21215492?file=37624184

External image sources, datasets, papers, repositories, and libraries used in the project should be cited here with links before final submission.

---

## 📜 License

MIT License
