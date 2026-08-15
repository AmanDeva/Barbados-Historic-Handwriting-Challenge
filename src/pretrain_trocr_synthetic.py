"""
Synthetic Domain Pre-Training & Fine-Tuning Pipeline for TrOCR-Large
Workflow:
Phase 1: Pre-train TrOCR-Large on 50,000 synthetic Barbados cursive images (models/trocr_large_synthetic_pretrained)
Phase 2: Fine-tune on 4,077 competition images from Train_Cleaned.csv (models/trocr_large_grandmaster_best)
Phase 3: Test inference with Native High-Resolution Slicing & SequenceMatcher Stitching
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from transformers import (
    VisionEncoderDecoderModel,
    TrOCRProcessor,
    AutoImageProcessor,
    AutoTokenizer,
    GenerationConfig,
    get_cosine_schedule_with_warmup
)

# Ensure project root is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.train_trocr import BarbadosTrOCRDataset, compute_metrics, MODEL_NAME, MAX_LENGTH

PRETRAINED_DIR = os.path.join(PROJECT_ROOT, "models", "trocr_large_synthetic_pretrained")
FINE_TUNED_DIR = os.path.join(PROJECT_ROOT, "models", "trocr_large_grandmaster_best")


def pretrain_on_synthetic_data(
    synth_csv: str = "data/synthetic_train.csv",
    synth_img_dir: str = "data/synthetic_images",
    output_dir: str = PRETRAINED_DIR,
    epochs: int = 3,
    batch_size: int = 4,
    grad_accum_steps: int = 8,
    lr: float = 5e-5
):
    print("==================================================================")
    print(" PHASE 1: PRE-TRAINING TrOCR-LARGE ON 50k SYNTHETIC BARBADOS DATA ")
    print(f" Effective Batch Size: {batch_size * grad_accum_steps} | Epochs: {epochs}")
    print("==================================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)
    processor.tokenizer.model_max_length = MAX_LENGTH

    model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
    start_token_id = processor.tokenizer.cls_token_id if processor.tokenizer.cls_token_id is not None else processor.tokenizer.bos_token_id
    gen_config = GenerationConfig(
        max_length=MAX_LENGTH,
        decoder_start_token_id=start_token_id,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        vocab_size=model.config.decoder.vocab_size,
        num_beams=1,
        do_sample=False
    )
    model.generation_config = gen_config
    model.config.decoder_start_token_id = start_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.eos_token_id
    model.to(device)

    full_synth_csv = os.path.join(PROJECT_ROOT, synth_csv)
    df = pd.read_csv(full_synth_csv)
    full_img_dir = os.path.join(PROJECT_ROOT, synth_img_dir)

    print(f"Loaded {len(df):,} synthetic samples for domain pre-training.")
    dataset = BarbadosTrOCRDataset(df, full_img_dir, processor, is_train=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = (len(loader) // grad_accum_steps) * epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.05), num_training_steps=total_steps)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()
        pbar = tqdm(loader, desc=f"Pre-Train Epoch [{epoch:02d}/{epochs:02d}]")
        step_in_epoch = 0

        for batch in pbar:
            step_in_epoch += 1
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            if device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    outputs = model(pixel_values=pixel_values, labels=labels)
                    loss = outputs.loss / grad_accum_steps
                scaler.scale(loss).backward()
            else:
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss / grad_accum_steps
                loss.backward()

            running_loss += loss.item() * grad_accum_steps

            if step_in_epoch % grad_accum_steps == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            pbar.set_postfix({"loss": f"{running_loss / step_in_epoch:.4f}"})

        print(f"\n---> Pre-Train Epoch [{epoch:02d}/{epochs:02d}] Loss: {running_loss / len(loader):.4f}")

    print(f"\n★ Pre-training Complete! Saving checkpoint to: {output_dir}")
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)


def finetune_on_competition_data(
    pretrained_model_dir: str = PRETRAINED_DIR,
    train_csv: str = "Train_Cleaned.csv",
    img_dir: str = "data/processed_images",
    output_dir: str = FINE_TUNED_DIR,
    epochs: int = 12,
    batch_size: int = 4,
    grad_accum_steps: int = 8,
    lr: float = 2e-5
):
    print("==================================================================")
    print(" PHASE 2: FINE-TUNING PRE-TRAINED TrOCR ON COMPETITION DATA ")
    print(f" Loaded from: {pretrained_model_dir} | Output: {output_dir}")
    print("==================================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    processor = TrOCRProcessor.from_pretrained(pretrained_model_dir)
    processor.tokenizer.model_max_length = MAX_LENGTH
    model = VisionEncoderDecoderModel.from_pretrained(pretrained_model_dir).to(device)

    full_csv_path = os.path.join(PROJECT_ROOT, train_csv)
    df = pd.read_csv(full_csv_path).sample(frac=1, random_state=42).reset_index(drop=True)

    val_size = int(len(df) * 0.10)
    val_df = df.iloc[:val_size]
    train_df = df.iloc[val_size:]

    full_img_dir = os.path.join(PROJECT_ROOT, img_dir)
    train_dataset = BarbadosTrOCRDataset(train_df, full_img_dir, processor, is_train=True)
    val_dataset = BarbadosTrOCRDataset(val_df, full_img_dir, processor, is_train=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = (len(train_loader) // grad_accum_steps) * epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.05), num_training_steps=total_steps)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    best_val_cer = 1.0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Fine-Tune Epoch [{epoch:02d}/{epochs:02d}]")
        step_in_epoch = 0

        for batch in pbar:
            step_in_epoch += 1
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            if device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    outputs = model(pixel_values=pixel_values, labels=labels)
                    loss = outputs.loss / grad_accum_steps
                scaler.scale(loss).backward()
            else:
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss / grad_accum_steps
                loss.backward()

            running_loss += loss.item() * grad_accum_steps

            if step_in_epoch % grad_accum_steps == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            pbar.set_postfix({"loss": f"{running_loss / step_in_epoch:.4f}"})

        # Validation Phase
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch [{epoch:02d}/{epochs:02d}] Eval"):
                pixel_values = batch["pixel_values"].to(device)
                generated_ids = model.generate(pixel_values)
                preds = processor.batch_decode(generated_ids, skip_special_tokens=True)
                val_preds.extend(preds)
                val_targets.extend(batch["target_text"])

        val_cer, val_wer = compute_metrics(val_preds, val_targets)
        val_score = 1.0 - (0.5 * val_wer + 0.5 * val_cer)
        print(f"\n---> Fine-Tune Epoch [{epoch:02d}/{epochs:02d}] Loss: {running_loss / len(train_loader):.4f} | Val CER: {val_cer:.4f} | Val WER: {val_wer:.4f} | Score: {val_score:.4f}")

        if val_cer < best_val_cer:
            best_val_cer = val_cer
            print(f"★ NEW GRANDMASTER BEST MODEL! Saving to {output_dir}...")
            model.save_pretrained(output_dir)
            processor.save_pretrained(output_dir)


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else "pretrain"
    if mode == "pretrain":
        pretrain_on_synthetic_data(epochs=3)
    elif mode == "finetune":
        finetune_on_competition_data(epochs=12)
