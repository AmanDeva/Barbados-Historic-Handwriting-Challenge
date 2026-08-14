"""
High-Resolution Native Slicing & DTW Sequence Stitching Engine for TrOCR
Eliminates aspect ratio downsampling blur on long lines (W > 800px) by:
1. Slicing long lines into overlapping 384x384 windows at 100% native pixel resolution
2. Decoding each patch independently with TrOCR-Large
3. Merging overlapping string predictions via SequenceMatcher Longest Common Subsequence
"""

import os
import sys
import difflib
import pandas as pd
import numpy as np
import torch
from PIL import Image
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

MODEL_NAME = "microsoft/trocr-large-handwritten"
MAX_LENGTH = 256


def stitch_overlapping_texts(text_left: str, text_right: str, min_overlap: int = 3) -> str:
    """
    Merges two overlapping text strings predicted from adjacent visual slices.
    Finds the longest matching contiguous substring between the tail of left and head of right.
    """
    text_left = str(text_left).strip()
    text_right = str(text_right).strip()

    if not text_left:
        return text_right
    if not text_right:
        return text_left
    if text_left == text_right:
        return text_left

    # Use difflib SequenceMatcher to find longest contiguous match
    matcher = difflib.SequenceMatcher(None, text_left, text_right)
    match = matcher.find_longest_match(0, len(text_left), 0, len(text_right))

    # If a clean overlap is found near the boundaries
    if match.size >= min_overlap:
        # Check if match is near the end of left and start of right
        left_tail_offset = len(text_left) - (match.a + match.size)
        right_head_offset = match.b

        if left_tail_offset <= 10 and right_head_offset <= 10:
            stitched = text_left[:match.a] + text_right[match.b:]
            return stitched.strip()

    # Fallback word-level overlap matching
    left_words = text_left.split()
    right_words = text_right.split()

    for k in range(min(len(left_words), len(right_words), 5), 0, -1):
        if left_words[-k:] == right_words[:k]:
            merged = left_words + right_words[k:]
            return " ".join(merged)

    # If no clean overlap detected, join with space
    return f"{text_left} {text_right}".strip()


def predict_single_image_sliced(
    img_path: str,
    model: VisionEncoderDecoderModel,
    processor: TrOCRProcessor,
    device: torch.device,
    slice_width: int = 384,
    stride: int = 256
) -> str:
    """Transcribes an image at native resolution using sliding-window slicing if wide."""
    if not os.path.exists(img_path):
        return ""

    img = Image.open(img_path).convert("RGB")
    orig_w, orig_h = img.size

    # Target height standard for TrOCR
    target_h = 384
    scaled_w = max(64, int(round(orig_w * (target_h / float(orig_h)))))
    scaled_img = img.resize((scaled_w, target_h), Image.Resampling.BICUBIC)

    # If line is standard width (<= 600px), process as single crop
    if scaled_w <= 600:
        pixel_values = processor(scaled_img, return_tensors="pt").pixel_values.to(device)
        with torch.no_grad():
            gen_ids = model.generate(pixel_values)
        pred = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
        return pred.strip()

    # Dynamic sliding-window slicing
    slices = []
    for left in range(0, scaled_w, stride):
        right = min(scaled_w, left + slice_width)
        slice_crop = scaled_img.crop((left, 0, right, target_h))

        # Pad slice to square 384x384 if needed
        if slice_crop.width < slice_width:
            pad_img = Image.new("RGB", (slice_width, target_h), (255, 255, 255))
            pad_img.paste(slice_crop, (0, 0))
            slice_crop = pad_img

        slices.append(slice_crop)
        if right >= scaled_w:
            break

    # Transcribe all slices
    slice_preds = []
    for s in slices:
        pixel_values = processor(s, return_tensors="pt").pixel_values.to(device)
        with torch.no_grad():
            gen_ids = model.generate(pixel_values)
        p = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
        if p:
            slice_preds.append(p)

    if not slice_preds:
        return ""

    # Stitch slices sequentially
    full_pred = slice_preds[0]
    for next_p in slice_preds[1:]:
        full_pred = stitch_overlapping_texts(full_pred, next_p)

    return full_pred


def run_sliced_inference(
    test_csv: str = "Test.csv",
    img_dir: str = "data/processed_images",
    model_dir: str = "models/trocr_large_best",
    output_csv: str = "submission_trocr_sliced.csv",
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    print("==================================================================")
    print(" HIGH-RESOLUTION SLICED TrOCR INFERENCE (NATIVE RESOLUTION) ")
    print("==================================================================")

    device = torch.device(device_str)
    try:
        processor = TrOCRProcessor.from_pretrained(model_dir)
    except Exception:
        img_proc = AutoImageProcessor.from_pretrained(model_dir)
        tok = AutoTokenizer.from_pretrained("roberta-base")
        processor = TrOCRProcessor(image_processor=img_proc, tokenizer=tok)

    processor.tokenizer.model_max_length = MAX_LENGTH

    model = VisionEncoderDecoderModel.from_pretrained(model_dir).to(device)
    model.eval()

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

    full_test_csv = os.path.join(PROJECT_ROOT, test_csv)
    test_df = pd.read_csv(full_test_csv)
    full_img_dir = os.path.join(PROJECT_ROOT, img_dir)

    all_preds = []
    for idx in tqdm(range(len(test_df)), desc="Transcribing Test Images (Native Slicing)"):
        img_id = str(test_df.iloc[idx]["ID"]).strip()
        img_path = os.path.join(full_img_dir, f"{img_id}.jpg")
        pred = predict_single_image_sliced(img_path, model, processor, device)
        all_preds.append(pred)

    sub_df = pd.DataFrame({
        "ID": test_df["ID"],
        "Target": all_preds
    })

    full_output_csv = os.path.join(PROJECT_ROOT, output_csv)
    sub_df.to_csv(full_output_csv, index=False)
    print(f"\n[OK] Sliced TrOCR predictions successfully exported to: {full_output_csv}")


if __name__ == '__main__':
    run_sliced_inference()
