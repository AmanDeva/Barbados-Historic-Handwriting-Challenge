# AGENTS.md — AI Agent & Developer Architecture Guide

Welcome to the **Barbados Historic Handwriting Challenge** repository. This guide provides AI agents with immediate technical orientation, critical domain rules, and architectural standards to operate effectively within this codebase.

---

## 1. Quick Technical Orientation

- **Goal**: Transcribe 17th–19th century Barbados handwriting line images.
- **Metric**: $\text{Score} = 1.0 - (0.5 \times \text{WER}_{\text{weighted}} + 0.5 \times \text{CER}_{\text{weighted}})$.
- **All-Time Best Public Score**: **`0.869784342`** (Submission 12 `mxBHZ2JL`: 5-Fold TrOCR-Large + MBR Consensus).
- **Target Benchmark**: **`0.927773`** (Rank #1 on Zindi).
- **Compute Target**: NVIDIA RTX 3090 (24 GB VRAM) on Vast.ai / Linux containers.

---

## 2. Critical Invariants & Rules for Agents

1. **Generation Length Trap (`max_length=256`)**:
   - `microsoft/trocr-large-handwritten` defaults to `max_length=64` in its config.
   - You **MUST** override `processor.tokenizer.model_max_length = 256` and set `GenerationConfig(max_length=256, ...)` in all generation scripts to avoid truncating long sentences (up to 170 characters).

2. **Greedy Decoding vs. Beam Search for TrOCR**:
   - Greedy decoding (`num_beams=1`, `do_sample=False`) outperforms Beam Search on this historical dataset (`0.857697` vs `0.855388`).
   - Beam search allows the modern RoBERTa language decoder prior to override literal visual quill strokes and modernize archaic spellings.

3. **Text-Level Consensus over Frame-Level Averaging**:
   - **Never** average frame-level CTC softmax probabilities across independent models. It causes `[BLANK]` cancellation and score collapse (Submission 2: `0.222338`).
   - **Always** use text-level **Minimum Bayes Risk (MBR) Consensus** (`src/mbr_consensus.py`), which minimizes pairwise Levenshtein distance across string predictions:
     $$\hat{y} = \arg\min_{y_i \in \mathcal{H}} \sum_{y_j \in \mathcal{H}} w_j \cdot \mathcal{L}_{\text{Levenshtein}}(y_i, y_j)$$

4. **Never Commit Large Image/Model Artifacts**:
   - `images/`, `data/processed_images/`, `data/synthetic_images/`, and `models/` are ignored in `.gitignore`.
   - Never commit binary checkpoints (`.pt`, `.bin`, `.safetensors`, `.zip`) to Git.

5. **Why Synthetic Computer TTF Fonts Failed**:
   - Pre-training TrOCR on synthetic TTF vector fonts caused catastrophic domain shift (Submission 11: `0.826855`).
   - Models must be trained directly on authentic historical document crops from `data/processed_images/` with CIELAB ink isolation.

---

## 3. Directory Layout & Key Scripts

```text
Barbados-Historic-Handwriting-Challenge/
├── data/
│   ├── processed_images/          # 5,472 enhanced, deskewed images (100% normalized)
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
├── Context.md                     # Comprehensive technical context document
└── AGENTS.md                      # This Agent guide
```

---

## 4. Key Execution Commands

### 1. Preprocess Images (takes ~15s on 16 CPU cores):
```bash
python scratch/batch_process_all_images.py
```

### 2. Run 5-Fold TrOCR Stratified Training:
```bash
python src/train_trocr_folds.py train
```

### 3. Generate 5-Fold MBR Consensus Submission:
```bash
python src/train_trocr_folds.py predict
```
*(Outputs `submission_trocr_5folds_mbr.csv`)*

### 4. Run Multi-Architecture MBR Consensus:
```bash
python src/mbr_consensus.py submission_trocr_5folds_mbr.csv Starters/VLM/submission_vlm.csv submission_final.csv
```
