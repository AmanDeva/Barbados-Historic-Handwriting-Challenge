"""
Pillar A: Native High-Resolution Sliding-Window Slicing & SequenceMatcher Stitching Engine
Architecture:
- Phase 1: Computer Vision Slicer (Window: 384px, Stride: 128px, Overlap: 256px, Dynamic White-Padding)
- Phase 2: Independent TrOCR-Large Batch Inference (Native 1:1 Pixel Resolution)
- Phase 3: NLP Sequence Alignment & Iterative Stitching via difflib.SequenceMatcher
"""

import os
import sys
import difflib
import math
import pandas as pd
import numpy as np
import torch
from PIL import Image, ImageOps
from tqdm import tqdm
from transformers import (
    VisionEncoderDecoderModel,
    TrOCRProcessor,
    AutoImageProcessor,
    AutoTokenizer,
    GenerationConfig
)

# Ensure project root is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

POSSIBLE_MODEL_DIRS = [
    os.path.join(PROJECT_ROOT, "models", "trocr_large_grandmaster_best"),
    os.path.join(PROJECT_ROOT, "models", "trocr_large_best"),
    os.path.join(PROJECT_ROOT, "models", "trocr_large_synthetic_pretrained"),
]

MODEL_DIR = next((d for d in POSSIBLE_MODEL_DIRS if os.path.exists(d)), os.path.join(PROJECT_ROOT, "models", "trocr_large_grandmaster_best"))
TEST_CSV = os.path.join(PROJECT_ROOT, "Test.csv")
IMG_DIR = os.path.join(PROJECT_ROOT, "data", "processed_images")
OUTPUT_CSV = os.path.join(PROJECT_ROOT, "submission_native_sliced.csv")

WINDOW_WIDTH = 384
WINDOW_HEIGHT = 384
STRIDE = 128
OVERLAP = WINDOW_WIDTH - STRIDE  # 256px


# ==============================================================================
# PHASE 1: THE SLIDING WINDOW IMAGE SLICER (COMPUTER VISION)
# ==============================================================================
def slice_image_sliding_window(
    img: Image.Image,
    window_w: int = WINDOW_WIDTH,
    window_h: int = WINDOW_HEIGHT,
    stride: int = STRIDE
) -> list:
    """
    Slices an image into overlapping 384x384 squares at native pixel resolution.
    1. Pads height to 384px with white pixels (255)
    2. Pads right width so it evenly fits the sliding window
    3. Crops slices at [0:384], [128:512], [256:640], ...
    """
    orig_w, orig_h = img.size

    # Step 1: Pad height to 384px (white background: 255)
    if orig_h < window_h:
        pad_top = (window_h - orig_h) // 2
        pad_bottom = window_h - orig_h - pad_top
        padded_img = ImageOps.expand(img, border=(0, pad_top, 0, pad_bottom), fill=(255, 255, 255))
    elif orig_h > window_h:
        # Scale proportionally to height = 384
        scale = window_h / float(orig_h)
        new_w = max(64, int(round(orig_w * scale)))
        padded_img = img.resize((new_w, window_h), Image.Resampling.BICUBIC)
    else:
        padded_img = img

    curr_w, curr_h = padded_img.size

    # Step 2: Handle short images (<= 384px) in 1 slice
    if curr_w <= window_w:
        pad_right = window_w - curr_w
        single_slice = ImageOps.expand(padded_img, border=(0, 0, pad_right, 0), fill=(255, 255, 255))
        return [single_slice]

    # Step 3: Pad right side to ensure complete coverage by sliding window
    num_steps = math.ceil((curr_w - window_w) / float(stride))
    target_padded_w = window_w + num_steps * stride
    if target_padded_w > curr_w:
        padded_img = ImageOps.expand(padded_img, border=(0, 0, target_padded_w - curr_w, 0), fill=(255, 255, 255))

    curr_w, curr_h = padded_img.size

    # Step 4: Slicing loop with 256px overlap
    slices = []
    for left in range(0, curr_w - window_w + 1, stride):
        right = left + window_w
        crop = padded_img.crop((left, 0, right, window_h))
        slices.append(crop)

    return slices


# ==============================================================================
# PHASE 3: TEXT SEQUENCE ALIGNMENT (NLP STITCHING VIA SequenceMatcher)
# ==============================================================================
def merge_two_text_fragments(left_str: str, right_str: str, min_match_len: int = 2) -> str:
    """
    Finds the longest contiguous overlap block between the suffix of left_str
    and the prefix of right_str, and appends only new characters from right_str.
    """
    left_str = str(left_str).strip()
    right_str = str(right_str).strip()

    if not left_str:
        return right_str
    if not right_str:
        return left_str
    if left_str == right_str:
        return left_str

    # 1. Character-level SequenceMatcher alignment
    matcher = difflib.SequenceMatcher(None, left_str, right_str)
    best_match = None
    max_score = -1

    # Search all matching blocks to find one that bridges left suffix to right prefix
    for match in matcher.get_matching_blocks():
        if match.size >= min_match_len:
            left_tail_dist = len(left_str) - (match.a + match.size)
            right_head_dist = match.b

            # Score match based on proximity to edges
            if left_tail_dist <= 8 and right_head_dist <= 8:
                score = match.size - (left_tail_dist + right_head_dist)
                if score > max_score:
                    max_score = score
                    best_match = match

    if best_match is not None:
        # Fuse left before match with right starting from match.b
        stitched = left_str[:best_match.a] + right_str[best_match.b:]
        return stitched.strip()

    # 2. Word-level fallback alignment
    left_words = left_str.split()
    right_words = right_str.split()

    for k in range(min(len(left_words), len(right_words), 6), 0, -1):
        if left_words[-k:] == right_words[:k]:
            return " ".join(left_words + right_words[k:])

    # 3. Default fallback: join with space if no clean overlap
    return f"{left_str} {right_str}".strip()


def stitch_all_slice_predictions(predictions: list) -> str:
    """Iteratively merges an ordered list of slice text predictions into one complete sentence."""
    valid_preds = [str(p).strip() for p in predictions if str(p).strip()]
    if not valid_preds:
        return ""

    final_text = valid_preds[0]
    for next_frag in valid_preds[1:]:
        final_text = merge_two_text_fragments(final_text, next_frag)

    return final_text


# ==============================================================================
# PHASE 2: INDEPENDENT TrOCR BATCH INFERENCE & PIPELINE RUNNER
# ==============================================================================
def run_native_slicing_pipeline():
    print("==================================================================")
    print(" PILLAR A: NATIVE SLICING & SequenceMatcher STITCHING INFERENCE ")
    print(f" Window: {WINDOW_WIDTH}px | Stride: {STRIDE}px | Overlap: {OVERLAP}px ")
    print("==================================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading TrOCR model from: {MODEL_DIR} (Device: {device})...")

    try:
        processor = TrOCRProcessor.from_pretrained(MODEL_DIR)
    except Exception:
        img_proc = AutoImageProcessor.from_pretrained(MODEL_DIR)
        tok = AutoTokenizer.from_pretrained("roberta-base")
        processor = TrOCRProcessor(image_processor=img_proc, tokenizer=tok)

    processor.tokenizer.model_max_length = 256
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_DIR).to(device)
    model.eval()

    start_token_id = processor.tokenizer.cls_token_id or processor.tokenizer.bos_token_id
    gen_config = GenerationConfig(
        max_length=256,
        decoder_start_token_id=start_token_id,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        vocab_size=model.config.decoder.vocab_size,
        num_beams=1,
        do_sample=False
    )
    model.generation_config = gen_config

    test_df = pd.read_csv(TEST_CSV)
    print(f"Loaded {len(test_df):,} test samples from: {TEST_CSV}\n")

    final_predictions = []

    for idx in tqdm(range(len(test_df)), desc="Transcribing (Native Slicing)"):
        img_id = str(test_df.iloc[idx]["ID"]).strip()
        img_path = os.path.join(IMG_DIR, f"{img_id}.jpg")

        if not os.path.exists(img_path):
            final_predictions.append("")
            continue

        img = Image.open(img_path).convert("RGB")

        # Phase 1: Slice into 384x384 overlapping windows
        slices = slice_image_sliding_window(img, window_w=WINDOW_WIDTH, window_h=WINDOW_HEIGHT, stride=STRIDE)

        # Phase 2: Batch inference across all slices of this image
        slice_tensors = [processor(s, return_tensors="pt").pixel_values.squeeze(0) for s in slices]
        batch_tensor = torch.stack(slice_tensors, dim=0).to(device)

        with torch.no_grad():
            gen_ids = model.generate(batch_tensor)
        slice_preds = processor.batch_decode(gen_ids, skip_special_tokens=True)

        # Phase 3: NLP Sequence Alignment & Stitching
        reconstructed_text = stitch_all_slice_predictions(slice_preds)
        final_predictions.append(reconstructed_text)

    sub_df = pd.DataFrame({
        "ID": test_df["ID"],
        "Target": final_predictions
    })

    sub_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[OK] Native Slicing submission saved successfully to: {OUTPUT_CSV}")
    return OUTPUT_CSV


if __name__ == '__main__':
    run_native_slicing_pipeline()
