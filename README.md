# R.O.A.D. Barbados Historic Handwriting OCR Challenge

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![All-Time Best](https://img.shields.io/badge/Zindi%20Score-0.869784-brightgreen.svg)](https://zindi.africa/)

This repository contains the complete, production-grade Optical Character Recognition (OCR) pipeline for the **R.O.A.D. Barbados Historic Handwriting Challenge** (Zindi). The goal is to accurately transcribe 17th–19th century archival Barbados deed and notary register records into digital text.

---

## 🏆 Current Benchmark: `0.869784` (Submission 12)
- **WER Weighted**: `2.195916`
- **CER Weighted**: `4.259106`
- **Architecture**: 5-Fold Stratified TrOCR-Large (`microsoft/trocr-large-handwritten`) + Minimum Bayes Risk (MBR) Consensus Decoding.

---

## 📖 Essential Documentation
- **[`Context.md`](Context.md)**: Master technical context, paleographic domain insights, metric definitions, and complete empirical submission scorecard (Submissions 1 to 12).
- **[`AGENTS.md`](AGENTS.md)**: AI Agent and developer architecture guide with operational rules and key invariants.

---

## 📁 Repository Structure

```text
Barbados-Historic-Handwriting-Challenge/
├── data/
│   ├── processed_images/          # All 5,472 CIELAB-enhanced, deskewed images
│   ├── archaic_corpus.txt         # 22,648 authentic Barbados legal phrases
│   └── fonts/                     # 10 Historical Cursive TTF fonts
├── src/
│   ├── train_trocr_folds.py       # 5-Fold Stratified TrOCR-Large Trainer & MBR Consensus (Sub 12: 0.869784)
│   ├── train_trocr.py             # Single TrOCR-Large Trainer with max_length=256
│   ├── mbr_consensus.py           # Multi-Architecture Minimum Bayes Risk Decoder
│   ├── train_byt5_refiner.py      # Token-free byte-level post-OCR refiner
│   ├── clean_hard_mislabeled_samples.py # Active learning noise trimmer
│   ├── preprocessing.py           # 4-Pillar Paleography preprocessor
│   ├── dataset.py                 # PyTorch Dataset, Tokenizer & Dynamic Collate
│   ├── model.py                   # CRNN Baseline (ResNet34 + BiLSTM + CTC)
│   └── metrics.py                 # Official Zindi Metric calculation engine
├── Starters/
│   └── VLM/                       # Qwen2-VL LoRA training and inference
├── Train_Cleaned.csv              # 4,077 clean training lines
├── Train_UltraCleaned.csv         # 4,076 noise-trimmed training lines
├── Train_Folds.csv                # Primary 5-fold stratified cross-validation CSV
├── Test.csv                       # 1,374 test image IDs
├── SampleSubmission.csv           # Submission template
├── Context.md                     # Comprehensive technical context document
├── AGENTS.md                      # AI Agent guide
└── README.md                      # This project overview
```

---

## ⚡ Quick Start Guide (Vast.ai / SageMaker GPU)

### 1. Setup & Preprocessing (~30 seconds):
```bash
# Clone & install dependencies
git clone https://github.com/AmanDeva/Barbados-Historic-Handwriting-Challenge.git
cd Barbados-Historic-Handwriting-Challenge
pip install -q pandas scipy tqdm opencv-python-headless transformers sentencepiece protobuf tiktoken editdistance peft bitsandbytes accelerate

# Download images & run 4-Pillar Paleography Preprocessing
wget -O images.zip https://storage.googleapis.com/road-handwriting/images.zip && \
unzip -q images.zip -d images/ && \
mv images/*/*.jpg images/ 2>/dev/null || true && \
python scratch/batch_process_all_images.py
```

### 2. Train 5-Fold TrOCR-Large Ensemble:
```bash
python src/train_trocr_folds.py train
```

### 3. Generate 5-Fold MBR Consensus Submission:
```bash
python src/train_trocr_folds.py predict
```
*(Outputs `submission_trocr_5folds_mbr.csv`)*

### 4. Train Qwen2-VL with LoRA:
```bash
python Starters/VLM/trainer.py --config Starters/VLM/config.yaml
python Starters/VLM/inference.py --config Starters/VLM/config.yaml
```

### 5. Multi-Architecture MBR Consensus:
```bash
python src/mbr_consensus.py submission_trocr_5folds_mbr.csv Starters/VLM/submission_vlm_7b.csv submission_final.csv
```

---

## 📜 License
This project is licensed under the MIT License.
