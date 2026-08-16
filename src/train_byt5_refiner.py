"""
Pillar C: ByT5 Byte-Level Historical Post-OCR Refiner & Spell-Corrector
Architecture:
- Phase 1: OOF & Realistic Paleographic OCR Noise Mining (Identity Preservation + Error Fixing)
- Phase 2: google/byt5-small (Token-Free UTF-8 Byte Sequence-to-Sequence Transformer)
- Phase 3: High-Performance PyTorch Training Loop with Mixed Precision (FP16 AMP)
- Phase 4: Constrained Beam Search Inference (num_beams=3, length_penalty=1.0, repetition prevention)
"""

import os
import sys
import random
import editdistance
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    GenerationConfig,
    get_cosine_schedule_with_warmup
)

# Ensure project root is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

MODEL_NAME = "google/byt5-small"
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "models", "byt5_historical_refiner")
TRAIN_CSV = os.path.join(PROJECT_ROOT, "Train_Cleaned.csv")
MAX_LENGTH = 256
TASK_PREFIX = "Fix OCR: "

# Historical OCR Character Confusion Matrix (Derived from TrOCR & CRNN error analysis)
OCR_CONFUSIONS = [
    ("m", "rn"), ("rn", "m"),
    ("d", "cl"), ("cl", "d"),
    ("s", "f"), ("f", "s"),
    ("vv", "w"), ("w", "vv"),
    ("in", "m"), ("m", "in"),
    ("nn", "m"), ("m", "nn"),
    ("e", "c"), ("c", "e"),
    ("o", "a"), ("a", "o"),
    ("i", "l"), ("l", "i"),
    ("t", "l"), ("l", "t"),
    ("y^t", "yt"), ("w^ch", "wch"),
    ("Cap^t.", "Capt"), ("Esq:^r", "Esqr"),
]


# ==============================================================================
# PHASE 1: GENERATING TRAINING DATA (OOF PAIRS + IDENTITY PRESERVATION)
# ==============================================================================
def inject_realistic_ocr_noise(text: str, noise_rate: float = 0.15) -> str:
    """Simulates authentic 18th-century cursive OCR character errors, spaces, and ligatures."""
    if not isinstance(text, str) or not text.strip():
        return ""

    chars = list(text)
    noisy = []
    i = 0
    while i < len(chars):
        c = chars[i]
        if random.random() < noise_rate:
            action = random.choice(["confuse", "drop_char", "insert_char", "drop_space", "insert_space"])

            if action == "confuse":
                replaced = False
                for orig, sub in OCR_CONFUSIONS:
                    if text[i:i + len(orig)] == orig:
                        noisy.append(sub)
                        i += len(orig)
                        replaced = True
                        break
                if not replaced:
                    noisy.append(c)
                    i += 1
            elif action == "drop_char":
                i += 1  # Skip character
            elif action == "insert_char":
                noisy.append(c)
                noisy.append(random.choice(["e", "i", "n", "t", "r", "-", "."]))
                i += 1
            elif action == "drop_space" and c == " ":
                i += 1  # Drop space
            elif action == "insert_space" and c != " ":
                noisy.append(c)
                noisy.append(" ")
                i += 1
            else:
                noisy.append(c)
                i += 1
        else:
            noisy.append(c)
            i += 1

    return "".join(noisy)


class ByT5HistoricalDataset(Dataset):
    """
    Dataset feeding (Noisy OCR String -> Clean Ground Truth)
    Blends 60% realistic noisy pairs with 40% clean identity pairs so ByT5 learns to act as
    a pass-through filter for correct words while fixing only corrupted characters.
    """
    def __init__(self, df: pd.DataFrame, tokenizer: AutoTokenizer, max_len: int = MAX_LENGTH, is_train: bool = True):
        self.clean_texts = df["Target"].dropna().astype(str).tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_train = is_train

    def __len__(self):
        return len(self.clean_texts) * (4 if self.is_train else 1)

    def __getitem__(self, idx):
        clean_target = self.clean_texts[idx % len(self.clean_texts)].strip()

        if self.is_train:
            # 60% probability of noise injection, 40% clean identity
            if random.random() < 0.60:
                source_input = inject_realistic_ocr_noise(clean_target, noise_rate=random.uniform(0.08, 0.22))
            else:
                source_input = clean_target
        else:
            source_input = inject_realistic_ocr_noise(clean_target, noise_rate=0.12)

        prompt_str = f"{TASK_PREFIX}{source_input}"

        input_enc = self.tokenizer(
            prompt_str,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        target_enc = self.tokenizer(
            clean_target,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        labels = target_enc.input_ids.squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_enc.input_ids.squeeze(0),
            "attention_mask": input_enc.attention_mask.squeeze(0),
            "labels": labels,
            "source_text": source_input,
            "clean_target": clean_target
        }


# ==============================================================================
# PHASE 2 & 3: MODEL SETUP & HIGH-PERFORMANCE TRAINING LOOP
# ==============================================================================
def train_byt5(
    train_csv: str = TRAIN_CSV,
    output_dir: str = OUTPUT_DIR,
    epochs: int = 12,
    batch_size: int = 16,
    lr: float = 3e-4
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    print("==================================================================")
    print(f" 🚀 TRAINING ByT5 BYTE-LEVEL HISTORICAL OCR REFINER ({MODEL_NAME}) ")
    print(f" Device: {device} | Epochs: {epochs} | Batch Size: {batch_size} | LR: {lr}")
    print("==================================================================")

    print("Loading ByT5 Tokenizer & Model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)

    df = pd.read_csv(train_csv).sample(frac=1, random_state=42).reset_index(drop=True)
    val_size = int(len(df) * 0.10)
    val_df = df.iloc[:val_size]
    train_df = df.iloc[val_size:]

    train_dataset = ByT5HistoricalDataset(train_df, tokenizer, is_train=True)
    val_dataset = ByT5HistoricalDataset(val_df, tokenizer, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * 0.05)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    best_val_cer = 1.0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"ByT5 Epoch [{epoch:02d}/{epochs:02d}] Train")

        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            if device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad()

            running_loss += loss.item()
            pbar.set_postfix({"loss": f"{running_loss / (len(pbar)):.4f}"})

        epoch_loss = running_loss / len(train_loader)

        # Validation Evaluation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"ByT5 Epoch [{epoch:02d}/{epochs:02d}] Eval"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                # Beam search with length penalty for generation
                gen_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=MAX_LENGTH,
                    num_beams=3,
                    length_penalty=1.0,
                    no_repeat_ngram_size=4,
                    early_stopping=True
                )
                decoded = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
                val_preds.extend([d.strip() for d in decoded])
                val_targets.extend(batch["clean_target"])

        # Compute CER
        total_dist = sum(editdistance.eval(p, g) for p, g in zip(val_preds, val_targets))
        total_len = sum(max(1, len(g)) for g in val_targets)
        val_cer = total_dist / total_len

        print(f"\n---> ByT5 Epoch [{epoch:02d}/{epochs:02d}] Loss: {epoch_loss:.4f} | Validation CER: {val_cer:.4f}")

        if val_cer < best_val_cer:
            best_val_cer = val_cer
            print(f"★ NEW BEST ByT5 REFINER! Saving to {output_dir}...")
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

    print(f"\n[OK] ByT5 Refiner Training Complete! Best model saved to: {output_dir}")


# ==============================================================================
# PHASE 4: INFERENCE & INTEGRATION (THE PIPELINE BRIDGE)
# ==============================================================================
def refine_predictions(
    input_csv: str,
    output_csv: str = "submission_refined_byt5.csv",
    model_dir: str = OUTPUT_DIR,
    batch_size: int = 32
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading ByT5 Refiner from: {model_dir} (Device: {device})...")

    tokenizer = AutoTokenizer.from_pretrained(model_dir if os.path.exists(model_dir) else MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir if os.path.exists(model_dir) else MODEL_NAME).to(device)
    model.eval()

    full_in_path = os.path.join(PROJECT_ROOT, input_csv) if not os.path.isabs(input_csv) else input_csv
    df = pd.read_csv(full_in_path).fillna("")
    print(f"Loaded {len(df):,} raw predictions from: {full_in_path}")

    raw_texts = df["Target"].tolist()
    refined_texts = []

    for i in tqdm(range(0, len(raw_texts), batch_size), desc="ByT5 Error Refining"):
        batch_raw = raw_texts[i:i + batch_size]
        prompts = [f"{TASK_PREFIX}{t.strip()}" for t in batch_raw]

        inputs = tokenizer(prompts, max_length=MAX_LENGTH, padding=True, truncation=True, return_tensors="pt").to(device)

        with torch.no_grad():
            gen_ids = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=MAX_LENGTH,
                num_beams=3,
                length_penalty=1.0,
                no_repeat_ngram_size=4,
                repetition_penalty=1.15,
                early_stopping=True
            )
        decoded = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        refined_texts.extend([d.strip() for d in decoded])

    out_df = pd.DataFrame({
        "ID": df["ID"],
        "Target": refined_texts
    })

    full_out_path = os.path.join(PROJECT_ROOT, output_csv) if not os.path.isabs(output_csv) else output_csv
    out_df.to_csv(full_out_path, index=False)
    print(f"\n[OK] Refined predictions saved successfully to: {full_out_path}")
    return full_out_path


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"
    if mode == "train":
        train_byt5(epochs=12, batch_size=16)
    elif mode == "refine":
        in_csv = sys.argv[2] if len(sys.argv) > 2 else "submission_native_sliced.csv"
        out_csv = sys.argv[3] if len(sys.argv) > 3 else "submission_byt5_refined.csv"
        refine_predictions(in_csv, out_csv)
