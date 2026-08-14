"""
ByT5 Byte-Level Historical Post-OCR Refiner & Spell-Correction Engine
Operates on raw UTF-8 bytes to correct character-level OCR mistakes, ligature corruptions,
and legal boilerplate phrasing without subword tokenizer fragmentation.
"""

import os
import sys
import random
import editdistance
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    get_cosine_schedule_with_warmup
)

# Ensure project root is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

BYT5_MODEL = "google/byt5-small"
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "models", "byt5_ocr_refiner")

# Historical OCR visual noise confusion pairs
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
]


def inject_synthetic_ocr_noise(text: str, noise_prob: float = 0.12) -> str:
    """Injects realistic visual OCR confusion noise into clean ground-truth text."""
    chars = list(text)
    noisy = []
    i = 0
    while i < len(chars):
        c = chars[i]
        if random.random() < noise_prob:
            action = random.choice(["replace_confuse", "drop", "insert_space", "drop_space"])

            if action == "replace_confuse":
                replaced = False
                for orig, conf in OCR_CONFUSIONS:
                    if text[i:i + len(orig)] == orig:
                        noisy.append(conf)
                        i += len(orig)
                        replaced = True
                        break
                if not replaced:
                    noisy.append(c)
                    i += 1
            elif action == "drop":
                i += 1  # Skip character
            elif action == "insert_space":
                noisy.append(c)
                noisy.append(" ")
                i += 1
            elif action == "drop_space" and c == " ":
                i += 1  # Drop space
            else:
                noisy.append(c)
                i += 1
        else:
            noisy.append(c)
            i += 1

    return "".join(noisy)


class ByT5OCRDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: AutoTokenizer, max_len: int = 256, is_train: bool = True):
        self.texts = df["Target"].dropna().astype(str).tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_train = is_train

    def __len__(self):
        return len(self.texts) * (3 if self.is_train else 1)

    def __getitem__(self, idx):
        clean_text = self.texts[idx % len(self.texts)]

        if self.is_train:
            noisy_text = inject_synthetic_ocr_noise(clean_text)
        else:
            noisy_text = clean_text

        input_enc = self.tokenizer(
            f"correct: {noisy_text}",
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        target_enc = self.tokenizer(
            clean_text,
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
            "clean_text": clean_text,
            "noisy_text": noisy_text
        }


def train_byt5_refiner(epochs: int = 10, batch_size: int = 16, lr: float = 5e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("==================================================================")
    print(f" TRAINING ByT5 BYTE-LEVEL HISTORICAL OCR REFINER ({BYT5_MODEL}) ")
    print(f" Device: {device} | Epochs: {epochs} | Batch Size: {batch_size}")
    print("==================================================================")

    tokenizer = AutoTokenizer.from_pretrained(BYT5_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(BYT5_MODEL).to(device)

    train_csv = os.path.join(PROJECT_ROOT, "Train_Cleaned.csv")
    df = pd.read_csv(train_csv).sample(frac=1, random_state=42).reset_index(drop=True)

    val_size = int(len(df) * 0.10)
    val_df = df.iloc[:val_size]
    train_df = df.iloc[val_size:]

    train_dataset = ByT5OCRDataset(train_df, tokenizer, is_train=True)
    val_dataset = ByT5OCRDataset(val_df, tokenizer, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.05), num_training_steps=total_steps)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"ByT5 Epoch [{epoch:02d}/{epochs:02d}] Train")

        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            pbar.set_postfix({"loss": f"{running_loss / (len(running_loss) if isinstance(running_loss, list) else 1):.4f}"})

        print(f"---> ByT5 Epoch [{epoch:02d}/{epochs:02d}] Loss: {running_loss / len(train_loader):.4f}")

    print(f"\n[OK] ByT5 Training Complete! Saving model to: {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)


def refine_submission_csv(input_csv: str, output_csv: str = "submission_byt5_refined.csv", batch_size: int = 32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nRefining OCR predictions from {input_csv} with ByT5...")

    tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR if os.path.exists(OUTPUT_DIR) else BYT5_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(OUTPUT_DIR if os.path.exists(OUTPUT_DIR) else BYT5_MODEL).to(device)
    model.eval()

    full_in_csv = os.path.join(PROJECT_ROOT, input_csv) if not os.path.isabs(input_csv) else input_csv
    df = pd.read_csv(full_in_csv).fillna("")

    all_refined = []
    texts = df["Target"].tolist()

    for i in tqdm(range(0, len(texts), batch_size), desc="ByT5 Refining"):
        batch_texts = [f"correct: {t}" for t in texts[i:i + batch_size]]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)

        with torch.no_grad():
            gen_ids = model.generate(**inputs, max_length=256)
        preds = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        all_refined.extend(preds)

    out_df = pd.DataFrame({
        "ID": df["ID"],
        "Target": all_refined
    })

    full_out_csv = os.path.join(PROJECT_ROOT, output_csv)
    out_df.to_csv(full_out_csv, index=False)
    print(f"[OK] Refined submission exported to: {full_out_csv}")


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"
    if mode == "train":
        train_byt5_refiner(epochs=10, batch_size=16)
    elif mode == "refine":
        in_file = sys.argv[2] if len(sys.argv) > 2 else "submission_trocr.csv"
        out_file = sys.argv[3] if len(sys.argv) > 3 else "submission_byt5_refined.csv"
        refine_submission_csv(in_file, out_file)
