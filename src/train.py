"""
Complete PyTorch Training & Validation Pipeline for R.O.A.D. Barbados Historic Handwriting Challenge.

Features:
- Fold-based training (5-Fold Stratified CV)
- AdamW Optimizer + CosineAnnealingLR Scheduler
- Mixed Precision Training (torch.cuda.amp) for VRAM efficiency & speed
- PyTorch CTCLoss integration
- Per-epoch validation evaluation with Zindi Weighted Metric (0.5 CER + 0.5 WER)
- Model checkpoint saving (models/crnn_best_fold{X}.pt)
"""

import os
import sys
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Any, List, Tuple, Optional

# Ensure project root is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dataset import Tokenizer, BarbadosOCRDataset, OCRCollateFn
from src.model import CRNN_OCR
from src.metrics import compute_zindi_metric


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: Optional[torch.cuda.amp.GradScaler],
    device: torch.device
) -> float:
    """Runs single training epoch and returns average CTC loss."""
    model.train()
    running_loss = 0.0
    total_samples = 0

    for batch_idx, batch in enumerate(dataloader):
        images = batch['images'].to(device)                 # (B, 3, 128, W_max)
        targets = batch['targets'].to(device)               # Flat 1D targets
        input_lengths = batch['input_lengths'].to(device)   # (B,)
        target_lengths = batch['target_lengths'].to(device) # (B,)

        optimizer.zero_grad()

        if scaler is not None and device.type == 'cuda':
            with torch.cuda.amp.autocast():
                logits = model(images)  # (B, W_seq, Vocab_Size)
                log_probs = F.log_softmax(logits, dim=2).permute(1, 0, 2)
                loss = criterion(log_probs, targets, input_lengths, target_lengths)

            if not (torch.isnan(loss) or torch.isinf(loss)):
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
        else:
            logits = model(images)
            log_probs = F.log_softmax(logits, dim=2).permute(1, 0, 2)
            loss = criterion(log_probs, targets, input_lengths, target_lengths)

            if not (torch.isnan(loss) or torch.isinf(loss)):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        if not (torch.isnan(loss) or torch.isinf(loss)):
            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

    avg_loss = running_loss / max(1, total_samples)
    return avg_loss


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    tokenizer: Tokenizer,
    device: torch.device
) -> Dict[str, Any]:
    """Runs evaluation on validation dataloader and returns Zindi metrics."""
    model.eval()

    all_preds = []
    all_refs = []

    for batch in dataloader:
        images = batch['images'].to(device)
        target_texts = batch['target_texts']

        if device.type == 'cuda':
            with torch.cuda.amp.autocast():
                logits = model(images)
        else:
            logits = model(images)

        preds = model.decode_greedy(logits, tokenizer)

        all_preds.extend(preds)
        all_refs.extend(target_texts)

    # Compute Zindi Metric (0.5 CER + 0.5 WER)
    res = compute_zindi_metric(all_preds, all_refs, cer_weight=0.5, wer_weight=0.5)
    return res


def train_fold(
    fold: int = 0,
    epochs: int = 25,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    device_str: Optional[str] = None,
    use_amp: bool = True
) -> Dict[str, Any]:
    """
    Trains CRNN model on specified fold and saves best checkpoint.
    """
    if device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(device_str)

    print(f"\n==================================================================")
    print(f" STARTING TRAINING ON FOLD {fold} ({epochs} Epochs, Batch Size: {batch_size})")
    print(f" Execution Device: {device} | Mixed Precision (AMP): {use_amp and device.type == 'cuda'}")
    print(f"==================================================================")

    train_folds_path = os.path.join(PROJECT_ROOT, 'Train_Folds.csv')
    if not os.path.exists(train_folds_path):
        train_folds_path = os.path.join(PROJECT_ROOT, 'data', 'train_folds.csv')

    df_all = pd.read_csv(train_folds_path)

    df_train = df_all[df_all['fold'] != fold].reset_index(drop=True)
    df_val = df_all[df_all['fold'] == fold].reset_index(drop=True)

    print(f"Fold {fold} Training Samples  : {len(df_train):,}")
    print(f"Fold {fold} Validation Samples: {len(df_val):,}")

    img_dir = os.path.join(PROJECT_ROOT, 'data', 'processed_images')

    tokenizer = Tokenizer()
    vocab_size = len(tokenizer)

    train_dataset = BarbadosOCRDataset(df_train, img_dir=img_dir, tokenizer=tokenizer, is_train=True)
    val_dataset = BarbadosOCRDataset(df_val, img_dir=img_dir, tokenizer=tokenizer, is_train=False)

    collate_fn = OCRCollateFn(downsample_factor=4, pad_value=0.0)

    num_workers = 4 if device.type == 'cuda' else 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda')
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda')
    )

    # Initialize CRNN Model
    model = CRNN_OCR(vocab_size=vocab_size, hidden_size=256, pretrained_backbone=True).to(device)

    # Criterion, Optimizer, Scheduler, Scaler
    criterion = nn.CTCLoss(blank=tokenizer.blank_idx, zero_infinity=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    scaler = torch.cuda.amp.GradScaler() if (use_amp and device.type == 'cuda') else None

    models_dir = os.path.join(PROJECT_ROOT, 'models')
    os.makedirs(models_dir, exist_ok=True)
    best_checkpoint_path = os.path.join(models_dir, f"crnn_best_fold{fold}.pt")

    best_val_score = float('inf')
    best_epoch = -1

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, scaler, device)
        scheduler.step()

        # Evaluate on Validation Fold
        val_res = evaluate_epoch(model, val_loader, tokenizer, device)
        val_score = val_res['final_score']
        val_cer = val_res['weighted_cer']
        val_wer = val_res['weighted_wer']

        epoch_time = time.time() - epoch_start

        # Checkpoint saving on best score
        is_best = val_score < best_val_score
        if is_best:
            best_val_score = val_score
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_score': val_score,
                'val_cer': val_cer,
                'val_wer': val_wer,
                'vocab_size': vocab_size,
                'fold': fold
            }, best_checkpoint_path)

        mark = "★ BEST" if is_best else ""
        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({epoch_time:.1f}s) | Train CTC Loss: {train_loss:.4f} | Val Score: {val_score:.4f} (CER: {val_cer:.4f}, WER: {val_wer:.4f}) {mark}")

    total_training_time = time.time() - start_time
    print(f"\n==================================================================")
    print(f" FOLD {fold} TRAINING COMPLETED IN {total_training_time/60:.2f} MINUTES ")
    print(f" Best Validation Score: {best_val_score:.4f} (Epoch {best_epoch})")
    print(f" Best Checkpoint Saved: {best_checkpoint_path}")
    print(f"==================================================================")

    return {
        'fold': fold,
        'best_epoch': best_epoch,
        'best_val_score': best_val_score,
        'checkpoint_path': best_checkpoint_path
    }

if __name__ == '__main__':
    train_fold(fold=0, epochs=25, batch_size=16)
