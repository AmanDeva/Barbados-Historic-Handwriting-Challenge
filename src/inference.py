"""
Inference & Submission Generator for R.O.A.D. Barbados Historic Handwriting Challenge.

Loads trained CRNN model checkpoint, performs inference on Test set,
and formats submission.csv matching SampleSubmission.csv specs.
"""

import os
import sys
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Ensure project root is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dataset import Tokenizer, BarbadosOCRDataset, OCRCollateFn
from src.model import CRNN_OCR

def generate_submission(
    checkpoint_path: str,
    output_csv_path: str = "submission.csv",
    batch_size: int = 16,
    device_str: str = "cpu"
) -> str:
    """
    Generates submission CSV predictions for all 1,374 test set images.
    """
    print("==================================================================")
    print(" GENERATING INFERENCE PREDICTIONS FOR SUBMISSION.CSV ")
    print("==================================================================")

    device = torch.device(device_str)
    print(f"Execution Device: {device}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    test_csv = os.path.join(PROJECT_ROOT, 'Test.csv')
    test_df = pd.read_csv(test_csv)
    print(f"Loaded Test.csv with {len(test_df):,} rows.")

    img_dir = os.path.join(PROJECT_ROOT, 'data', 'processed_images')

    tokenizer = Tokenizer()
    vocab_size = len(tokenizer)

    # Load Model Checkpoint
    model = CRNN_OCR(vocab_size=vocab_size, hidden_size=256, pretrained_backbone=False).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"[OK] Successfully loaded trained checkpoint: {checkpoint_path}")
    print(f"     • Best Validation Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"     • Best Validation Score: {checkpoint.get('val_score', 'N/A'):.4f}")

    test_dataset = BarbadosOCRDataset(test_df, img_dir=img_dir, tokenizer=tokenizer, is_train=False)
    collate_fn = OCRCollateFn(downsample_factor=4, pad_value=0.0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    all_ids = []
    all_preds = []

    print("\nRunning inference on test dataset...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            images = batch['images'].to(device)
            img_ids = batch['img_ids']

            logits = model(images)
            preds = model.decode_greedy(logits, tokenizer)

            all_ids.extend(img_ids)
            all_preds.extend(preds)

            if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == len(test_loader):
                print(f"  Processed {len(all_ids):,} / {len(test_df):,} test predictions...")

    # Build Submission Dataframe
    sub_df = pd.DataFrame({
        'ID': all_ids,
        'Target': all_preds
    })

    # Ensure format matches SampleSubmission.csv
    sample_sub_path = os.path.join(PROJECT_ROOT, 'SampleSubmission.csv')
    if os.path.exists(sample_sub_path):
        sample_sub = pd.read_csv(sample_sub_path)
        # Order rows exactly as in SampleSubmission.csv
        sub_df = sample_sub[['ID']].merge(sub_df, on='ID', how='left').fillna("")

    full_output_path = os.path.join(PROJECT_ROOT, output_csv_path)
    sub_df.to_csv(full_output_path, index=False)

    print(f"\n[OK] Submission CSV successfully generated at: {full_output_path}")
    print(f"     • Total Rows     : {len(sub_df):,}")
    print(f"     • Columns        : {list(sub_df.columns)}")
    print(f"     • Non-empty Preds: {(sub_df['Target'].str.len() > 0).sum():,}")

    print("\nSample Submission Predictions (First 5 Rows):")
    print(sub_df.head().to_string(index=False))

    return full_output_path

if __name__ == '__main__':
    checkpoint_file = os.path.join(PROJECT_ROOT, 'models', 'crnn_best_fold0.pt')
    if os.path.exists(checkpoint_file):
        generate_submission(checkpoint_file)
    else:
        print(f"Checkpoint file {checkpoint_file} not found. Please run src/train.py first.")
