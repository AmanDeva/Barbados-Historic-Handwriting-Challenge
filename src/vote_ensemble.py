"""
Text-Level Consensus Ensemble (Majority Voting) for CTC OCR Models.

Unlike frame-level probability averaging (which suffers from CTC blank token alignment cancellation),
Text-Level Majority Voting decodes each fold model independently into full text strings,
then computes consensus transcriptions across folds.
"""

import os
import sys
import pandas as pd
from collections import Counter
import torch
from torch.utils.data import DataLoader
from typing import List, Dict, Any

# Ensure project root is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dataset import Tokenizer, BarbadosOCRDataset, OCRCollateFn
from src.model import CRNN_OCR


def run_single_model_inference(checkpoint_path: str, test_loader: DataLoader, tokenizer: Tokenizer, device: torch.device) -> List[str]:
    """Runs inference for a single model checkpoint and returns predicted text list."""
    vocab_size = len(tokenizer)
    model = CRNN_OCR(vocab_size=vocab_size, hidden_size=256, pretrained_backbone=False).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    preds = []
    with torch.no_grad():
        for batch in test_loader:
            images = batch['images'].to(device)
            if device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    logits = model(images)
            else:
                logits = model(images)
            batch_preds = model.decode_greedy(logits, tokenizer)
            preds.extend(batch_preds)
    return preds


def vote_ensemble_submissions(
    checkpoint_paths: List[str],
    output_csv_path: str = "submission.csv",
    batch_size: int = 16,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> str:
    """
    Decodes each fold model independently and performs majority voting per line sample.
    """
    print("\n==================================================================")
    print(f" GENERATING TEXT-LEVEL CONSENSUS ENSEMBLE ({len(checkpoint_paths)} MODELS) ")
    print("==================================================================")

    device = torch.device(device_str)
    test_csv = os.path.join(PROJECT_ROOT, 'Test.csv')
    test_df = pd.read_csv(test_csv)

    img_dir = os.path.join(PROJECT_ROOT, 'data', 'processed_images')
    tokenizer = Tokenizer()

    test_dataset = BarbadosOCRDataset(test_df, img_dir=img_dir, tokenizer=tokenizer, is_train=False)
    collate_fn = OCRCollateFn(downsample_factor=4, pad_value=0.0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Collect predictions from each model independently
    all_model_preds = []
    valid_ckpts = []

    for ckpt_path in checkpoint_paths:
        if os.path.exists(ckpt_path):
            print(f"Running inference for model: {os.path.basename(ckpt_path)}...")
            preds = run_single_model_inference(ckpt_path, test_loader, tokenizer, device)
            all_model_preds.append(preds)
            valid_ckpts.append(ckpt_path)

    if not all_model_preds:
        raise ValueError("No valid model checkpoints found for consensus ensemble!")

    num_models = len(all_model_preds)
    num_samples = len(test_df)
    print(f"\n[OK] Successfully decoded {num_models} models across {num_samples:,} test samples.")

    # Perform Majority Voting per sample
    consensus_preds = []
    for i in range(num_samples):
        sample_votes = [all_model_preds[m][i] for m in range(num_models)]
        # Most common prediction
        most_common = Counter(sample_votes).most_common(1)[0][0]
        consensus_preds.append(most_common)

    sub_df = pd.DataFrame({
        'ID': test_df['ID'],
        'Target': consensus_preds
    })

    sample_sub_path = os.path.join(PROJECT_ROOT, 'SampleSubmission.csv')
    if os.path.exists(sample_sub_path):
        sample_sub = pd.read_csv(sample_sub_path)
        sub_df = sample_sub[['ID']].merge(sub_df, on='ID', how='left').fillna("")

    full_output_path = os.path.join(PROJECT_ROOT, output_csv_path)
    sub_df.to_csv(full_output_path, index=False)

    print(f"\n==================================================================")
    print(f" TEXT-LEVEL CONSENSUS SUBMISSION.CSV GENERATED ")
    print(f"==================================================================")
    print(f"✓ Location: {full_output_path}")
    print(f"✓ Total Predictions: {len(sub_df):,}")

    return full_output_path

if __name__ == '__main__':
    ckpts = [os.path.join(PROJECT_ROOT, 'models', f"crnn_best_fold{k}.pt") for k in range(5)]
    vote_ensemble_submissions(ckpts)
