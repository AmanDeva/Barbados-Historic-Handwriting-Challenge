"""
5-Fold Cross-Validation Training & Ensemble Inference Generator
for R.O.A.D. Barbados Historic Handwriting Challenge.

Features:
- Sequential training across all 5 Folds (Fold 0..4)
- Automated checkpoint saving for each fold (models/crnn_best_fold{K}.pt)
- Multi-Fold Softmax Probability Averaging (Ensemble Inference)
- Generation of final 5-Fold Ensemble submission.csv
"""

import os
import sys
import time
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import List, Dict, Any

# Ensure project root is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dataset import Tokenizer, BarbadosOCRDataset, OCRCollateFn
from src.model import CRNN_OCR
from src.train import train_fold


def generate_ensemble_submission(
    checkpoint_paths: List[str],
    output_csv_path: str = "submission.csv",
    batch_size: int = 16,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> str:
    """
    Runs multi-model ensemble inference on Test dataset by averaging predicted softmax probabilities.
    """
    print("\n==================================================================")
    print(f" GENERATING 5-FOLD ENSEMBLE SUBMISSION ({len(checkpoint_paths)} CHECKPOINTS) ")
    print("==================================================================")

    device = torch.device(device_str)
    print(f"Execution Device: {device}")

    test_csv = os.path.join(PROJECT_ROOT, 'Test.csv')
    test_df = pd.read_csv(test_csv)
    print(f"Loaded Test.csv with {len(test_df):,} rows.")

    img_dir = os.path.join(PROJECT_ROOT, 'data', 'processed_images')
    tokenizer = Tokenizer()
    vocab_size = len(tokenizer)

    # Load all trained models
    models = []
    for ckpt_path in checkpoint_paths:
        if not os.path.exists(ckpt_path):
            print(f"⚠️ Checkpoint not found: {ckpt_path}, skipping...")
            continue
        model = CRNN_OCR(vocab_size=vocab_size, hidden_size=256, pretrained_backbone=False).to(device)
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        models.append(model)
        print(f"[OK] Loaded checkpoint: {os.path.basename(ckpt_path)} (Epoch {checkpoint.get('epoch', 'N/A')}, Val Score: {checkpoint.get('val_score', 'N/A'):.4f})")

    if not models:
        raise ValueError("No valid model checkpoints were loaded!")

    print(f"\n[OK] Ensembling {len(models)} active model checkpoints...")

    test_dataset = BarbadosOCRDataset(test_df, img_dir=img_dir, tokenizer=tokenizer, is_train=False)
    collate_fn = OCRCollateFn(downsample_factor=4, pad_value=0.0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    all_ids = []
    all_preds = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            images = batch['images'].to(device)
            img_ids = batch['img_ids']

            # Accumulate softmax probability predictions across all models
            prob_sum = None
            for model in models:
                if device.type == 'cuda':
                    with torch.amp.autocast('cuda'):
                        logits = model(images)
                else:
                    logits = model(images)

                probs = F.softmax(logits, dim=2)  # (Batch, W_seq, Vocab_Size)

                if prob_sum is None:
                    prob_sum = probs
                else:
                    prob_sum += probs

            # Average probabilities
            avg_probs = prob_sum / float(len(models))
            arg_maxes = torch.argmax(avg_probs, dim=2)  # (Batch, W_seq)

            # CTC Greedy decoding on ensembled probabilities
            for i in range(arg_maxes.size(0)):
                tokens = arg_maxes[i].tolist()
                decoded_text = tokenizer.decode(tokens, remove_ctc_duplicates=True)
                all_preds.append(decoded_text)

            all_ids.extend(img_ids)

            if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == len(test_loader):
                print(f"  Ensemble Inference: {len(all_ids):,} / {len(test_df):,} test predictions processed...")

    # Build Submission Dataframe
    sub_df = pd.DataFrame({'ID': all_ids, 'Target': all_preds})

    sample_sub_path = os.path.join(PROJECT_ROOT, 'SampleSubmission.csv')
    if os.path.exists(sample_sub_path):
        sample_sub = pd.read_csv(sample_sub_path)
        sub_df = sample_sub[['ID']].merge(sub_df, on='ID', how='left').fillna("")

    full_output_path = os.path.join(PROJECT_ROOT, output_csv_path)
    sub_df.to_csv(full_output_path, index=False)

    print(f"\n==================================================================")
    print(f" 5-FOLD ENSEMBLE SUBMISSION.CSV SUCCESSFULLY GENERATED ")
    print(f"==================================================================")
    print(f"✓ Location            : {full_output_path}")
    print(f"✓ Total Predictions   : {len(sub_df):,}")
    print(f"✓ Non-empty Predictions: {(sub_df['Target'].str.len() > 0).sum():,}")

    print("\nSample Ensembled Predictions (First 5 Rows):")
    print(sub_df.head().to_string(index=False))

    return full_output_path


def run_full_5fold_training(epochs: int = 25, batch_size: int = 16, learning_rate: float = 1e-3):
    """
    Trains models on all 5 folds (Fold 0..4) and generates 5-fold ensemble submission.csv.
    """
    print("==================================================================")
    print(" LAUNCHING FULL 5-FOLD CROSS-VALIDATION TRAINING PIPELINE ")
    print("==================================================================")

    checkpoint_paths = []
    fold_results = []

    start_total = time.time()

    for fold in range(5):
        print(f"\n>>> Starting Fold {fold} / 4 Training...")
        res = train_fold(
            fold=fold,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )
        checkpoint_paths.append(res['checkpoint_path'])
        fold_results.append(res)

    print("\n==================================================================")
    print(" ALL 5 FOLDS TRAINED SUCCESSFULLY! ")
    print("==================================================================")
    for res in fold_results:
        print(f"  • Fold {res['fold']}: Best Val Score = {res['best_val_score']:.4f} (Epoch {res['best_epoch']})")

    # Generate 5-Fold Ensemble Submission
    sub_path = generate_ensemble_submission(checkpoint_paths, output_csv_path="submission.csv")
    print(f"\n✓ Pipeline Complete in {(time.time() - start_total)/60:.2f} Minutes!")


if __name__ == '__main__':
    run_full_5fold_training(epochs=25, batch_size=16)
