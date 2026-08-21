# R.O.A.D. Barbados Historic Handwriting OCR Challenge — Project Context & Knowledge Base

This document serves as the master context and institutional knowledge base for any AI agent or developer working on this codebase.

---

## 1. Challenge Mission & Evaluation Metric

- **Competition**: R.O.A.D. Barbados Historic Handwriting OCR Challenge (Zindi)
- **Objective**: Transcribe cropped horizontal line images from 17th–19th century Barbados archival deed and notary register records into digital text.
- **Evaluation Metric**:
  $$\text{Public Score} = 1.0 - \left(0.5 \times \text{WER}_{\text{weighted}} + 0.5 \times \text{CER}_{\text{weighted}}\right)$$
  - Higher score is better (Maximum score is $1.0$).
  - Errors and character/word lengths are aggregated globally across all 1,374 test line samples.

---

## 2. Leaderboard Evolution & Verified Empirical History

| Sub # | Submission ID | Method / Architecture Description | Public Score | WER Weighted | CER Weighted | Key Takeaway & Findings |
| :---: | :---: | :--- | :---: | :---: | :---: | :--- |
| **1** | `8znU5QPE` | Baseline CRNN (ResNet-34 + 2-layer BiLSTM + CTC), Fold 0 | `0.807875` | `3.41947` | `5.46108` | Initial deep learning baseline. |
| **2** | `YrU1xae1` | 5-Fold CRNN Frame-Level Softmax Probability Averaging | `0.222338` | `26.8267` | `34.7056` | **Failure**: Frame-level CTC averaging caused `[BLANK]` tokens from out-of-phase alignments to cancel character emissions. |
| **3** | `X7TJkQmx` | 5-Fold CRNN Text-Level Consensus Voting | `0.819862` | `3.20297` | `5.13480` | Proved that text-level Levenshtein consensus works for OCR ensembles. |
| **4** | `ZiDSW4mt` | Qwen2-VL-2B-Instruct (3 Epochs, LoRA $r=16$) | `0.804273` | `3.18645` | `6.92537` | VLM subword tokenizer expanded historical abbreviations (`wch` $\rightarrow$ `which`), penalizing CER. |
| **5** | `SKZTQRr1` | Qwen2-VL-2B-Instruct (3 Epochs, Re-run) | `0.804273` | `3.18645` | `6.92537` | Verification re-run. |
| **6** | `WtrE1ysn` | Qwen2-VL-2B-Instruct (8 Epochs, LoRA $r=32$, $\alpha=64$) | `0.810792` | `2.98777` | `7.11889` | Lowest WER (`2.987`) among early models; confirmed strong 18th-c. syntactic reasoning. |
| **7** | `ziF3sdch` | TrOCR-Large (15 Epochs, CIELAB Preprocessing, Greedy Decoding) | `0.857697` | `2.37525` | `4.76667` | **Major Breakthrough**: `max_length=256` override prevented sentence truncation; greedy decoding anchored visual strokes. |
| **8** | `NaSkMYba` | TrOCR-Large (15 Epochs, 4-Beam Search) | `0.855388` | `2.42918` | `4.77347` | Beam search slightly favored modern English word priors over archaic visual strokes. |
| **9** | `mhyEiUhM` | TrOCR-Large Fold 0 (15 Epochs on 80% data, Greedy Decoding) | `0.853193` | `2.43648` | `4.98153` | Validated single fold stability on 80% training subset. |
| **10**| `DcKM6kpz` | Sliding-Window Slicing ($384\times 384$ Tiles, Stride 128, SequenceMatcher) | `-2.01327` | `35.0009` | `171.0386`| **Failure**: White padding ($128\text{px} \rightarrow 384\text{px}$) reduced text scale $3\times$; imperfect slice stitching caused concatenation explosion (400+ char strings). |
| **11**| `mbDhrcfg` | TrOCR-Large (50k Synthetic Font Pre-Training + 12 Epochs Fine-Tune) | `0.826855` | `2.77687` | `6.31855` | **Domain Shift**: Computer vector TTF fonts shifted the visual encoder away from authentic human quill strokes. |
| **12**| `mxBHZ2JL` | **5-Fold TrOCR-Large Ensemble (15 Epochs/Fold) + MBR Consensus** | **`0.869784`** | **`2.19591`** | **`4.25910`** | **ALL-TIME BEST 🟢**: Fused 5 independent folds with pairwise Levenshtein risk minimization (+0.0121 leap). |

- **Current Leaderboard Benchmark (Rank #1 `eszn`)**: `Public Score: 0.92777385` (WER: `1.297158`, CER: `1.999564`).

---

## 3. Dataset & Preprocessing Pipeline

### Data Structure:
- Total Images: **5,472** archival line crops.
- Training Set: **4,098** raw / **4,077** cleaned (`Train_Cleaned.csv`) / **4,076** sanitized (`Train_UltraCleaned.csv`).
- Test Set: **1,374** images (`Test.csv`).
- 5-Fold Stratified Splits: `Train_Folds.csv` (balanced across 5 length-quantile bins).

### 4-Pillar Paleography Preprocessing Engine (`src/preprocessing.py`):
1. **CIELAB Iron Gall Ink Isolation**: $L^* - 0.65 b^*$ subtracts yellowed parchment background and eliminates reverse-side bleed-through (verso bleed).
2. **Projection Profile Variance Deskewing**: Automatically straightens tilted baselines within $\pm 7^\circ$.
3. **$3\times 3$ Morphological Stroke Healing (`MORPH_CLOSE`)**: Reconnects faded hairline loops (`b`, `d`, `g`, `p`, `o`).
- All 5,472 images are preprocessed into `data/processed_images/` in ~15 seconds across 16 CPU cores via `scratch/batch_process_all_images.py`.

---

## 4. Key Paleographic & Linguistic Domain Insights

### Authentic 17th/18th-Century Barbados Notary Patterns:
- **Archaic Contractions**: `wch` (which), `sd` (said), `ye` (the), `yt` (that), `&c` (etc), `p. Ann:` (per annum), `dd` (delivered), `Xpian` (Christian - Greek Chi notation).
- **Historical Superscripts**: Formatted with `^` caret notation: `y^t`, `Cap^t.`, `Esq:^r`, `w^ch`, `p^r`.
- **Double 'ff' Capitalization**: 17th-century scribe method for capital 'F' (`ffrancis`, `ffortescue`).
- **Archival Commodities & Legal Formulas**: `pounds of Tobaccoe` (early commodity currency), `Sterling money of Barbados`, `househould stuffe`, `heires and assignes`, `Lett suite trouble eviction molest`.

---

## 5. Codebase Directory Map

```text
Barbados-Historic-Handwriting-Challenge/
├── data/
│   ├── processed_images/          # 5,472 CIELAB-enhanced, deskewed images
│   ├── fonts/                     # 10 Historical Cursive TTF fonts
│   ├── archaic_corpus.txt         # 22,648 authentic Barbados legal phrases
│   └── purged_noisy_samples.csv   # Mislabeled sample audit log
├── src/
│   ├── train_trocr_folds.py       # 5-Fold TrOCR Stratified Trainer & MBR Consensus (Sub 12: 0.869784)
│   ├── train_trocr.py             # Single-model TrOCR trainer with max_length=256
│   ├── mbr_consensus.py           # Multi-Architecture Minimum Bayes Risk pairwise Levenshtein decoder
│   ├── train_byt5_refiner.py      # Token-free byte-level post-OCR error corrector
│   ├── clean_hard_mislabeled_samples.py # Active learning noise trimmer (Train_UltraCleaned.csv)
│   ├── preprocessing.py           # 4-Pillar Paleography preprocessor
│   ├── dataset.py                 # PyTorch Dataset, Tokenizer & Dynamic Collate
│   ├── model.py                   # CRNN Baseline model (ResNet34 + BiLSTM + CTC)
│   └── metrics.py                 # Official Zindi Metric calculation engine
├── Starters/
│   └── VLM/
│       ├── trainer.py             # Qwen2-VL LoRA training engine
│       ├── inference.py           # Qwen2-VL inference script
│       └── config.yaml            # VLM training & inference config
├── Train_Cleaned.csv              # 4,077 clean ground-truth training lines
├── Train_UltraCleaned.csv         # 4,076 noise-trimmed training lines
├── Train_Folds.csv                # Primary 5-fold stratified cross-validation CSV
├── Test.csv                       # 1,374 test image IDs
├── SampleSubmission.csv           # Submission template
├── Context.md                     # Master project context & knowledge base
└── README.md                      # Quick-start guide
```

---

## 6. Standard Execution Workflows

### 1. Download & Preprocess All Images (~25 seconds):
```bash
wget -O images.zip https://storage.googleapis.com/road-handwriting/images.zip && \
unzip -q images.zip -d images/ && \
mv images/*/*.jpg images/ 2>/dev/null || true && \
python scratch/batch_process_all_images.py
```

### 2. Train 5-Fold TrOCR-Large Ensemble (~4.2 hours on RTX 3090):
```bash
python src/train_trocr_folds.py train
```

### 3. Generate 5-Fold MBR Consensus Submission (~5 minutes):
```bash
python src/train_trocr_folds.py predict
```
*(Generates `submission_trocr_5folds_mbr.csv` - Current Best: `0.869784`)*

### 4. Train Qwen2-VL with LoRA:
```bash
python Starters/VLM/trainer.py --config Starters/VLM/config.yaml
python Starters/VLM/inference.py --config Starters/VLM/config.yaml
```

### 5. Multi-Architecture MBR Consensus Decoding:
```bash
python src/mbr_consensus.py submission_trocr_5folds_mbr.csv Starters/VLM/submission_vlm_7b.csv submission_final.csv
```
