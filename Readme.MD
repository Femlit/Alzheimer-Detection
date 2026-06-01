# 🧠 Alzheimer's Disease Detection from Brain MRI

> A deep learning web application that classifies Alzheimer's disease severity from brain MRI scans using a dual-branch EfficientNetB4 + ResNet50V2 attention fusion model.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://femi-alzheimer-detection.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)
![Accuracy](https://img.shields.io/badge/Test%20Accuracy-98.99%25-brightgreen)
![AUC](https://img.shields.io/badge/AUC-0.9998-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

##  Overview

Alzheimer's disease affects over 55 million people worldwide and is the leading cause of dementia. Early and accurate diagnosis is critical for effective intervention. This project builds a deep learning system that classifies brain MRI scans into four Alzheimer's severity stages with near-perfect accuracy, deployed as an interactive web application accessible to anyone with a browser.

The system uses a novel dual-branch architecture combining **EfficientNetB4** and **ResNet50V2** through an attention-based feature fusion mechanism, trained on the **OASIS MRI dataset** with a strict zero-leakage data split.

---

##  Classification Stages

| Stage | CDR Score | Description |
|---|---|---|
| **Non Demented** | 0 | No signs of cognitive impairment |
| **Very Mild Dementia** | 0.5 | Minimal memory lapses, early stage |
| **Mild Dementia** | 1 | Moderate cognitive decline |
| **Moderate Dementia** | 2 | Significant impairment, advanced stage |

---

## 📊 Model Performance

| Metric | Value |
|---|---|
| Test Accuracy | **98.99%** |
| AUC | **0.9998** |
| Weighted F1-Score | **98.99%** |
| Mild Dementia Recall | **1.00 (perfect)** |
| Moderate Dementia Recall | **1.00 (perfect)** |
| Test Set Size | 12,968 images |
| Data Leakage | None |

### Per-Class Results

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| Mild Dementia | 0.96 | 1.00 | 0.98 |
| Moderate Dementia | 0.97 | 1.00 | 0.99 |
| Non Demented | 1.00 | 0.99 | 0.99 |
| Very Mild Dementia | 0.96 | 0.99 | 0.98 |

---

## 🏗️ Architecture

```
Input MRI Image (224 × 224 × 3)
        │
        ├──────────────────┬──────────────────┐
        │                  │                  │
   Branch A           Branch B               │
 EfficientNetB4      ResNet50V2              │
 (last 30 layers     (last 30 layers         │
  fine-tuned)         fine-tuned)            │
        │                  │                  │
   Global Avg Pool    Global Avg Pool         │
        │                  │                  │
    Dense 512          Dense 512              │
   BatchNorm           BatchNorm              │
   Dropout(0.4)       Dropout(0.4)           │
        │                  │                  │
  Attention Gate      Attention Gate          │
   (Sigmoid)          (Sigmoid)              │
        │                  │                  │
        └──────── Concatenate (1024-dim) ─────┘
                           │
                      Dense 512
                     BatchNorm
                     Dropout(0.4)
                           │
                      Dense 256
                     Dropout(0.3)
                           │
                    Softmax (4 classes)
```

---

## 📁 Project Structure

```
Alzheimer-Detection/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── .streamlit/
│   └── config.toml           # Streamlit theme configuration
├── assets/
│   ├── training_history.png  # Training curves
│   ├── confusion_matrix.png  # Confusion matrix
│   ├── roc_auc_curves.png    # ROC AUC curves
│   ├── per_class_metrics.png # Per-class bar chart
│   └── gradcam_grid.png      # Grad-CAM examples
├── data/
│   └── final_metrics.json    # Evaluation metrics
└── utils/
    ├── __init__.py
    ├── model_loader.py       # Model download and caching
    └── gradcam.py            # Gradient visualisation
```

---

## 🗃️ Dataset

**OASIS Alzheimer's Detection** — [Kaggle](https://www.kaggle.com/datasets/ninadaithal/imagesoasis)

Derived from the Open Access Series of Imaging Studies (OASIS) longitudinal MRI dataset (Marcus et al., 2007).

| Class | Images | Percentage |
|---|---|---|
| Non Demented | 67,222 | 77.77% |
| Very Mild Dementia | 13,725 | 15.88% |
| Mild Dementia | 5,002 | 5.79% |
| Moderate Dementia | 488 | 0.56% |
| **Total** | **86,437** | **100%** |

**Split:** 70% train / 15% validation / 15% test — patient-level, zero data leakage

---

## 🚀 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/Femlit/Alzheimer-Detection.git
cd Alzheimer-Detection
```

### 2. Create a virtual environment

```bash
# Using conda
conda create -n alzheimer python=3.11
conda activate alzheimer

# Or using venv
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
# Mac
pip install -r requirements_mac.txt

# Linux / Windows
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 🌐 Deploy on Streamlit Cloud

1. Fork or clone this repository to your GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select this repository and set `app.py` as the main file
5. Click **Deploy**

The model (~330 MB) is hosted on Google Drive and downloaded automatically on first run.

---

## 🔬 Benchmark Comparison

| Study | Architecture | Dataset | Accuracy |
|---|---|---|---|
| Chandrasekaran et al. (2025) | Sequential CNN | Kaggle | 94.20% |
| Abd El-Latif et al. (2023) | Lightweight CNN | Kaggle | 95.93% |
| **This Work (2026)** | **EfficientNetB4 + ResNet50V2 Fusion** | **OASIS clean** | **98.99%** |

---

## 🖥️ App Pages

- **Diagnose** — Upload a brain MRI scan, get instant classification with confidence scores and gradient heatmap
- **Model Performance** — View training curves, confusion matrix, ROC curves, and per-class metrics
- **About** — Project background, architecture details, and disclaimer

---

## ⚙️ Technologies

- **Deep Learning:** TensorFlow 2.15, Keras
- **Architectures:** EfficientNetB4, ResNet50V2
- **Web App:** Streamlit
- **Visualisation:** Matplotlib, OpenCV, Seaborn
- **Training Platform:** Kaggle Notebooks (NVIDIA Tesla T4 x2)
- **Model Hosting:** Google Drive via gdown

---

## ⚠️ Disclaimer

This application is developed for **academic research and educational purposes** as a final year project at Olabisi Onabanjo University, Department of Computer Engineering. It is **not a certified medical device** and should not be used to make clinical diagnostic decisions without review by a qualified neurologist or radiologist.

---

## 👤 Author

**Olabode**
Department of Computer Engineering
Olabisi Onabanjo University, Nigeria
Supervised by Dr. Oyedeji — 2026

---

## 📄 Reference

Marcus, D. S., Wang, T. H., Parker, J., Csernansky, J. G., Morris, J. C., & Buckner, R. L. (2007). Open Access Series of Imaging Studies (OASIS): Cross-sectional MRI data in young, middle aged, nondemented, and demented older adults. *Journal of Cognitive Neuroscience*, 19(9), 1498–1507.
