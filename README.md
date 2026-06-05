# 🫀 Fetal Echo Quality Classifier

A clinical ultrasound quality assessment system for first trimester fetal echocardiography scans using EfficientNet-B3, EigenCAM explainability, FastAPI, and synthetic ultrasound augmentation.

The system classifies fetal ultrasound scan quality into six categories and generates explainability heatmaps alongside clinical quality guidance.

This project is a prototype training and image quality control tool. It is not intended for diagnosis or clinical decision making.

---

# 📌 Project Overview

Fetal echocardiography is used to evaluate fetal heart structures during pregnancy. In first trimester imaging, scan quality is especially important because the fetal heart is small, moving, and difficult to capture clearly.

Poor scan quality makes clinical review harder and reduces the reliability of downstream AI systems. This project focuses on the step before diagnosis: checking whether a scan is clear enough for review, training, or further analysis.

Fetal Echo Quality Classifier helps classify ultrasound scan quality into six categories:

- Diagnostic Quality
- Motion / Focus Artefact
- Insufficient Gain
- Low Tissue Contrast
- Noise Artefact
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

# 🎯 Motivation

The main bottleneck this project addresses is image quality assessment in fetal echocardiography.

In fetal ultrasound, a scan may look usable at first glance but still contain quality issues such as blur, low contrast, poor gain, acoustic noise, or incorrect probe angle. These problems matter because they affect whether the image is useful for training, review, or downstream AI analysis.

This project was inspired by three needs:

- Students and early career ultrasound learners need feedback on scan quality.
- Rural or lower resource clinics may have limited access to specialized fetal echo expertise.
- AI systems in medicine depend on input image quality before any diagnostic model becomes useful.

The goal is not to diagnose fetal heart disease. The goal is to help users assess whether the image quality is strong enough for meaningful review.

---

# ✨ Core Features

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

# 🧭 Project Track

This project fits the following tracks:

- Domain specific idea, healthcare AI
- Application / Product
- Research component through model training, dataset creation, evaluation, and explainability

---

# 📚 Dataset and Preprocessing

## Dataset Source

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

# Ultrasound Quality Classes

The model uses internal labels for training and maps them to clearer clinical labels in the interface.

```text
good
blurry
too_dark
low_contrast
noisy
angled
