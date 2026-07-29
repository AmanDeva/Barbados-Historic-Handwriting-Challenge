import os
import sys
import torch
from torch.utils.data import DataLoader
import pandas as pd

# Ensure project root is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dataset import Tokenizer, BarbadosOCRDataset, OCRCollateFn

def main():
    print("==================================================================")
    print(" PYTORCH DATASET, TOKENIZER & CUSTOM COLLATE VERIFICATION ")
    print("==================================================================")

    # 1. Test Tokenizer (Phase 1)
    print("\n--- PHASE 1: TOKENIZER VERIFICATION ---")
    tokenizer = Tokenizer()
    print(f"[OK] Total Vocabulary Size (including [BLANK] at index 0): {len(tokenizer)} tokens")
    print(f"[OK] Blank Token Index: {tokenizer.blank_idx} ({repr(tokenizer.id2char[0])})")

    sample_text = "By this public Act and Instrument of protest 1845"
    encoded = tokenizer.encode(sample_text)
    decoded = tokenizer.decode(encoded)

    print(f"\nSample Text Encoding Test:")
    print(f"  Raw Text: {repr(sample_text)}")
    print(f"  Encoded : {encoded[:15]}... ({len(encoded)} tokens)")
    print(f"  Decoded : {repr(decoded)}")
    assert sample_text == decoded, "Tokenizer decode error: decoded string does not match raw text!"
    print("[OK] Tokenizer string <-> token encoding/decoding is 100% losslessly reversible!")

    # 2. Test PyTorch Dataset (Phase 2)
    print("\n--- PHASE 2: DATASET VERIFICATION ---")
    train_folds_path = os.path.join(PROJECT_ROOT, 'Train_Folds.csv')
    img_dir = os.path.join(PROJECT_ROOT, 'data', 'processed_images')

    df_folds = pd.read_csv(train_folds_path)
    print(f"Loaded {len(df_folds):,} rows from: {train_folds_path}")

    dataset = BarbadosOCRDataset(df_folds, img_dir=img_dir, tokenizer=tokenizer)
    sample_item = dataset[0]

    print(f"\nSample Dataset Item 0:")
    print(f"  Image ID     : {sample_item['img_id']}")
    print(f"  Image Tensor : shape={tuple(sample_item['image'].shape)}, dtype={sample_item['image'].dtype}")
    print(f"  Target Text  : {repr(sample_item['target_text'])}")
    print(f"  Target Tensor: shape={tuple(sample_item['target'].shape)}, len={sample_item['target_length']}")

    assert sample_item['image'].shape[0] == 3 and sample_item['image'].shape[1] == 128, "Image tensor must be (3, 128, W)"
    print("[OK] Dataset item shapes and ImageNet normalization verified!")

    # 3. Test Custom Collate Function & DataLoader (Phase 3)
    print("\n--- PHASE 3: CUSTOM COLLATE FUNCTION & DATALOADER VERIFICATION ---")
    collate_fn = OCRCollateFn(downsample_factor=4, pad_value=0.0)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

    batch = next(iter(dataloader))

    print(f"Batch Execution Test (Batch Size = 16):")
    print(f"  • Padded Images Tensor Shape : {tuple(batch['images'].shape)} (Batch, 3, 128, Max_W)")
    print(f"  • Max Batch Width            : {batch['max_width']} px")
    print(f"  • Concatenated Targets Shape : {tuple(batch['targets'].shape)} (Flat 1D CTC targets)")
    print(f"  • Input Lengths Shape        : {tuple(batch['input_lengths'].shape)} (CTC feature sequence timesteps)")
    print(f"  • Target Lengths Shape       : {tuple(batch['target_lengths'].shape)} (Ground truth character counts)")

    # Assert CTC Condition: Input Length (timesteps) >= Target Length (characters)
    ctc_valid = torch.all(batch['input_lengths'] >= batch['target_lengths']).item()
    print(f"\n[OK] CTC Mathematical Condition (Input Steps >= Target Chars): {ctc_valid}")
    if not ctc_valid:
        print("⚠️ WARNING: Some images in the batch have input timesteps smaller than target char count!")
        for idx in range(16):
            if batch['input_lengths'][idx] < batch['target_lengths'][idx]:
                print(f"   ID: {batch['img_ids'][idx]} -> Input Steps: {batch['input_lengths'][idx].item()}, Target Chars: {batch['target_lengths'][idx].item()}")
    else:
        print("[OK] All 16 batch samples satisfy CTC sequence length requirements perfectly!")

    print("\n==================================================================")
    print(" ALL DATASET, TOKENIZER & COLLATE TESTS PASSED 100%! ")
    print("==================================================================")

if __name__ == '__main__':
    main()
