# R.O.A.D. Barbados Historic Handwriting Challenge

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the complete, production-grade Optical Character Recognition (OCR) pipeline for the **R.O.A.D. Barbados Historic Handwriting Challenge**. The objective is to transcribe historical 18th–19th century Barbados archival record line images into digital text.

---

## 📁 Repository Structure

```text
Barbados-Historic-Handwriting-Challenge/
├── data/
│   ├── processed_images/     # All 5,472 CLAHE-enhanced, H=128px line images
│   ├── train_folds.csv       # Stratified 5-Fold dataset with length quantile bins
│   ├── processed_train.csv   # Normalized train labels
│   └── processed_test.csv    # Test set metadata
├── src/
│   ├── preprocessing.py      # Label normalizer & CLAHE image enhancer
│   ├── create_stratified_folds.py # 5-Fold length quantile split engine
│   ├── dataset.py            # PyTorch Dataset, Tokenizer & Dynamic Collate
│   ├── model.py              # CRNN (ResNet-18 4x downsampling + 2x BiLSTM + CTC)
│   ├── metrics.py            # Official Zindi Metric (0.5 CER + 0.5 WER)
│   ├── train.py              # Training loop with CosineAnnealingLR & Checkpoint
│   └── inference.py          # Test set prediction & submission.csv generator
├── CRNN.ipynb                # End-to-End SageMaker GPU Training Notebook
├── Train_Cleaned.csv         # Cleaned ground-truth training targets
├── Train_Folds.csv           # Primary 5-fold cross-validation CSV
├── Test.csv                  # Test image IDs (1,374 rows)
├── SampleSubmission.csv      # Submission format specification
├── Context.md                # Project documentation
└── README.md                 # Project guide
```

---

## ⚡ Quick Start on AWS SageMaker (`ml.g4dn.2xlarge`)

### 1. Clone Repository in SageMaker JupyterLab:
```bash
git clone https://github.com/AmanDeva/Barbados-Historic-Handwriting-Challenge.git
cd Barbados-Historic-Handwriting-Challenge
```

### 2. Run End-to-End Training Notebook:
- Open **`CRNN.ipynb`** in SageMaker JupyterLab.
- Select **PyTorch 2.0 (Python 3.10 / CUDA 11.8)** kernel.
- Run all cells to train the model on GPU and automatically generate `submission.csv`.

---

## 🎯 Architectural Summary

### 1. Data Preprocessing & Scaling (`src/preprocessing.py`)
- **CLAHE Enhancement**: Applied `clipLimit=2.0, tileGridSize=(8,8)` to sharpen faded iron gall ink loops without blowing out paper background.
- **Dimensional Standardization**: 100% of images normalized to $H=128\text{px}$ with proportional width scaling ($W = \text{round}(128 \times \text{Aspect Ratio})$), preserving natural handwriting stroke shapes.

### 2. Validation Strategy (`src/create_stratified_folds.py`)
- **Stratified 5-Fold Cross-Validation**: Binned text lengths into 5 quantile categories ($0-20\%$, $20-40\%$, $40-60\%$, $60-80\%$, $80-100\%$).
- **Mean Text Length Variance**: $\le 0.25$ characters across all 5 folds, preventing long-sentence validation skew.

### 3. Model Architecture (`src/model.py`)
- **Feature Backbone**: ResNet-18 with modified `(2, 1)` strides in `layer2..4` for exact **$4\times$ width downsampling**.
- **Sequence Modeler**: 2-layer Bidirectional LSTM (512 hidden size).
- **Output Head**: Linear projection to 76 vocabulary tokens + PyTorch `nn.CTCLoss`.

### 4. Official Zindi Metric Engine (`src/metrics.py`)
- **Metric Formula**:
  $$\text{Final Score} = 0.5 \times \text{Weighted CER} + 0.5 \times \text{Weighted WER}$$
- Errors and lengths are accumulated globally across all reference lines.

---

## 📜 License
This project is licensed under the MIT License.
