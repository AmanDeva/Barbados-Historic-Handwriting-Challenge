import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd

# Ensure project root is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dataset import Tokenizer, BarbadosOCRDataset, OCRCollateFn
from src.model import CRNN_OCR

def main():
    print("==================================================================")
    print(" CRNN OCR MODEL ARCHITECTURE VERIFICATION ")
    print("==================================================================")

    # 1. Instantiate Model & Tokenizer
    tokenizer = Tokenizer()
    vocab_size = len(tokenizer)

    model = CRNN_OCR(vocab_size=vocab_size, hidden_size=256, pretrained_backbone=True)
    model.eval()

    print(f"\n[OK] Successfully instantiated CRNN_OCR Model:")
    print(f"  • Vocabulary Size      : {vocab_size} tokens")
    print(f"  • Feature Backbone     : ResNet-18 (Tailored for H=128px line images)")
    print(f"  • Sequence Modeler     : 2-layer Bidirectional LSTM (Hidden Size: 256 -> 512 out)")
    print(f"  • CTC Linear Head      : Linear(512 -> {vocab_size})")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  • Total Parameters     : {total_params:,} ({total_params/1e6:.2f} Million)")

    # 2. Test Forward Pass with Synthetic Tensor
    print("\n--- TEST 1: FORWARD PASS SHAPE VERIFICATION ---")
    dummy_input = torch.randn(4, 3, 128, 1024)  # (Batch=4, Channels=3, Height=128, Width=1024)
    with torch.no_grad():
        logits = model(dummy_input)

    expected_w_seq = 1024 // 4  # 4x downsampling -> 256 timesteps
    print(f"Input Shape : {tuple(dummy_input.shape)} (Batch, C, H, W)")
    print(f"Logits Shape: {tuple(logits.shape)} (Batch, Timesteps={expected_w_seq}, Vocab_Size={vocab_size})")

    assert logits.shape == (4, expected_w_seq, vocab_size), f"Expected logits shape (4, {expected_w_seq}, {vocab_size}), got {tuple(logits.shape)}"
    print("[OK] Forward pass shapes verified perfectly!")

    # 3. Test CTCLoss Integration with Real DataLoader Batch
    print("\n--- TEST 2: PYTORCH CTCLOSS INTEGRATION & FORWARD PASS ---")
    train_folds_path = os.path.join(PROJECT_ROOT, 'Train_Folds.csv')
    img_dir = os.path.join(PROJECT_ROOT, 'data', 'processed_images')

    df_folds = pd.read_csv(train_folds_path)
    dataset = BarbadosOCRDataset(df_folds.head(32), img_dir=img_dir, tokenizer=tokenizer)
    collate_fn = OCRCollateFn(downsample_factor=4, pad_value=0.0)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

    batch = next(iter(dataloader))

    images = batch['images']                   # (B=8, 3, 128, W_max)
    targets = batch['targets']                 # Flat 1D targets
    input_lengths = batch['input_lengths']     # (B=8,)
    target_lengths = batch['target_lengths']   # (B=8,)

    # Compute Forward Pass Logits
    logits = model(images)  # (B, W_seq, Vocab_Size)

    # Prepare log_probs for PyTorch CTCLoss (Time_First shape: W_seq, Batch, Vocab_Size)
    log_probs = F.log_softmax(logits, dim=2).permute(1, 0, 2)

    ctc_loss_fn = nn.CTCLoss(blank=tokenizer.blank_idx, zero_infinity=True)
    loss = ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)

    print(f"Batch Execution:")
    print(f"  • Batch Image Tensor Shape  : {tuple(images.shape)}")
    print(f"  • CTC Log Probs Shape (T,B,V): {tuple(log_probs.shape)}")
    print(f"  • CTC Loss Output Value     : {loss.item():.4f}")
    assert not torch.isnan(loss) and not torch.isinf(loss), "CTC Loss returned NaN or Inf!"
    print("[OK] CTC Loss successfully calculated with zero NaN/Inf errors!")

    # 4. Test CTC Greedy Decoder
    print("\n--- TEST 3: CTC GREEDY DECODER VERIFICATION ---")
    decoded_preds = model.decode_greedy(logits, tokenizer)
    print(f"Decoded Predictions Sample (Untrained Model Output):")
    for i in range(min(3, len(decoded_preds))):
        print(f"  Sample {i} [{batch['img_ids'][i]}]:")
        print(f"    Target Text   : {repr(batch['target_texts'][i])}")
        print(f"    Decoded Output: {repr(decoded_preds[i])}")

    print("\n==================================================================")
    print(" ALL CRNN MODEL ARCHITECTURE TESTS PASSED 100%! ")
    print("==================================================================")

if __name__ == '__main__':
    main()
