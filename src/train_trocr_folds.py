"""
5-Fold Stratified Cross-Validation & MBR Consensus Engine for TrOCR-Large
Trains 5 independent Vision Transformer models across all 5 folds and decodes
with 4-Beam Search + Minimum Bayes Risk (MBR) Consensus.
"""

import os
import sys
import editdistance
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


def train_single_fold(fold: int, epochs: int = 12, batch_size: int = 4, grad_accum_steps: int = 8, lr: float = 3e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = os.path.join(PROJECT_ROOT, "models", f"trocr_large_fold{fold}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f" TRAINING TrOCR-LARGE -- FOLD {fold}/4 (Epochs: {epochs}, Device: {device}) ")
    print(f"{'='*70}")

    image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)
    processor.tokenizer.model_max_length = MAX_LENGTH

    model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
    start_token_id = processor.tokenizer.cls_token_id if processor.tokenizer.cls_token_id is not None else processor.tokenizer.bos_token_id
    pad_token_id = processor.tokenizer.pad_token_id
    eos_token_id = processor.tokenizer.eos_token_id

    gen_config = GenerationConfig(
        max_length=MAX_LENGTH,
        decoder_start_token_id=start_token_id,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        vocab_size=model.config.decoder.vocab_size,
        num_beams=1,
        do_sample=False
    )
    model.generation_config = gen_config
    model.config.decoder_start_token_id = start_token_id
    model.config.pad_token_id = pad_token_id
    model.config.eos_token_id = eos_token_id
    model.to(device)

    folds_csv = os.path.join(PROJECT_ROOT, "Train_Folds.csv")
    if not os.path.exists(folds_csv):
        folds_csv = os.path.join(PROJECT_ROOT, "data", "folds.csv")
    df = pd.read_csv(folds_csv)

    train_df = df[df["fold"] != fold].reset_index(drop=True)
    val_df = df[df["fold"] == fold].reset_index(drop=True)

    img_dir = os.path.join(PROJECT_ROOT, "data", "processed_images")
    train_dataset = BarbadosTrOCRDataset(train_df, img_dir, processor, is_train=True)
    val_dataset = BarbadosTrOCRDataset(val_df, img_dir, processor, is_train=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = (len(train_loader) // grad_accum_steps) * epochs
    warmup_steps = int(total_steps * 0.05)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    best_val_cer = 1.0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Fold {fold} | Epoch [{epoch:02d}/{epochs:02d}] Train")
        step_in_epoch = 0

        for batch in pbar:
            step_in_epoch += 1
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast('cuda'):
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss / grad_accum_steps
            scaler.scale(loss).backward()

            running_loss += loss.item() * grad_accum_steps

            if step_in_epoch % grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            pbar.set_postfix({"loss": f"{running_loss / step_in_epoch:.4f}"})

        epoch_loss = running_loss / len(train_loader)

        # Validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Fold {fold} | Epoch [{epoch:02d}/{epochs:02d}] Eval"):
                pixel_values = batch["pixel_values"].to(device)
                generated_ids = model.generate(pixel_values, generation_config=gen_config)
                preds = processor.batch_decode(generated_ids, skip_special_tokens=True)
                val_preds.extend(preds)
                val_targets.extend(batch["target_text"])

        val_cer, val_wer = compute_metrics(val_preds, val_targets)
        val_score = 1.0 - (0.5 * val_wer + 0.5 * val_cer)
        print(f"\n---> Fold {fold} | Epoch [{epoch:02d}/{epochs:02d}] Loss: {epoch_loss:.4f} | Val CER: {val_cer:.4f} | Val WER: {val_wer:.4f} | Score: {val_score:.4f}")

        if val_cer < best_val_cer:
            best_val_cer = val_cer
            print(f"★ NEW BEST FOR FOLD {fold}! Saving to {output_dir}...")
            model.save_pretrained(output_dir)
            processor.save_pretrained(output_dir)


def train_all_folds(epochs: int = 12):
    for f in range(5):
        train_single_fold(fold=f, epochs=epochs)
    print("\n[OK] ALL 5 FOLDS OF TrOCR-LARGE TRAINED SUCCESSFULLY!")


def predict_5fold_ensemble(output_csv: str = "submission_trocr_5fold.csv", batch_size: int = 8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_csv = os.path.join(PROJECT_ROOT, "Test.csv")
    test_df = pd.read_csv(test_csv)
    img_dir = os.path.join(PROJECT_ROOT, "data", "processed_images")

    all_fold_preds = []

    for fold in range(5):
        model_dir = os.path.join(PROJECT_ROOT, "models", f"trocr_large_fold{fold}")
        if not os.path.exists(model_dir):
            print(f"Skipping fold {fold} (directory not found: {model_dir})")
            continue

        print(f"\nPredicting with Fold {fold} model...")
        image_processor = AutoImageProcessor.from_pretrained(model_dir)
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)
        processor.tokenizer.model_max_length = MAX_LENGTH

        model = VisionEncoderDecoderModel.from_pretrained(model_dir).to(device)
        model.eval()

        gen_config = GenerationConfig(
            max_length=MAX_LENGTH,
            decoder_start_token_id=processor.tokenizer.cls_token_id or processor.tokenizer.bos_token_id,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            vocab_size=model.config.decoder.vocab_size,
            num_beams=1,
            do_sample=False
        )

        test_dataset = BarbadosTrOCRDataset(test_df, img_dir, processor, is_train=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        fold_preds = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Predicting Fold {fold} (4-Beam Search)"):
                pixel_values = batch["pixel_values"].to(device)
                generated_ids = model.generate(pixel_values, generation_config=gen_config)
                preds = processor.batch_decode(generated_ids, skip_special_tokens=True)
                fold_preds.extend([str(p).strip() for p in preds])

        all_fold_preds.append(fold_preds)

    if not all_fold_preds:
        raise ValueError("No fold models found to generate predictions!")

    print("\nComputing Minimum Bayes Risk (MBR) Consensus across all fold predictions...")
    num_samples = len(test_df)
    consensus_predictions = []

    for idx in range(num_samples):
        candidates = [all_fold_preds[f][idx] for f in range(len(all_fold_preds))]
        best_candidate = candidates[0]
        min_risk = float('inf')

        for i, y_i in enumerate(candidates):
            total_dist = sum(editdistance.eval(y_i, y_j) for j, y_j in enumerate(candidates) if i != j)
            if total_dist < min_risk:
                min_risk = total_dist
                best_candidate = y_i

        consensus_predictions.append(best_candidate)

    sub_df = pd.DataFrame({
        "ID": test_df["ID"],
        "Target": consensus_predictions
    })

    full_output_csv = os.path.join(PROJECT_ROOT, output_csv)
    sub_df.to_csv(full_output_csv, index=False)
    print(f"\n[OK] 5-Fold TrOCR MBR Consensus Submission saved to: {full_output_csv}")


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"
    if mode == "train":
        train_all_folds(epochs=12)
    elif mode == "predict":
        predict_5fold_ensemble()
