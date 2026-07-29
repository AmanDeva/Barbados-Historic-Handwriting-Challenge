"""
Unified Preprocessing and Text Cleaning Pipeline for R.O.A.D. Barbados Historic Handwriting Challenge.

Phases:
- Phase 1: Ground-Truth Text Normalization (Label Cleaning)
- Phase 2: Image Contrast Enhancement (CLAHE)
- Phase 3: Dimensional Standardization (Proportional Height-Fixed Scaling)
"""

import os
import re
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from typing import Tuple, Optional, Dict, Any, List

# Tier 1 Noise Characters identified during EDA
TIER1_NOISE_CHARS = ['\\', '|', '#', '?', '*']
# Regular expression pattern matching any of the noise characters
NOISE_REGEX_PATTERN = re.compile(r'[' + re.escape(''.join(TIER1_NOISE_CHARS)) + r']')


class TextNormalizer:
    """
    Phase 1: Ground-Truth Text Normalizer.
    Cleans target text strings by removing ultra-rare noise symbols
    and compressing extra whitespaces.
    """
    def __init__(self, noise_chars: List[str] = TIER1_NOISE_CHARS):
        self.noise_chars = noise_chars
        self.pattern = re.compile(r'[' + re.escape(''.join(noise_chars)) + r']')

    def clean_text(self, text: str) -> Tuple[str, bool]:
        """
        Cleans input target text string.
        
        Returns:
            cleaned_text (str): The normalized text string.
            is_valid (bool): False if string becomes empty after cleaning, True otherwise.
        """
        if not isinstance(text, str) or pd.isnull(text):
            return "", False

        # 1. Remove Tier 1 Noise Characters
        cleaned = self.pattern.sub('', text)

        # 2. Collapse consecutive spaces into a single space
        cleaned = re.sub(r'\s+', ' ', cleaned)

        # 3. Strip leading and trailing whitespace
        cleaned = cleaned.strip()

        # 4. Validation step
        is_valid = len(cleaned) > 0

        return cleaned, is_valid


class ImageEnhancer:
    """
    Phases 2 & 3: Image Contrast Enhancement and Dimensional Standardization.
    Applies CLAHE to grayscale line images, performs proportional height-fixed scaling,
    and converts output back to 3-channel RGB.
    """
    def __init__(self, target_height: int = 128, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)):
        self.target_height = target_height
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def enhance_and_resize(self, image: np.ndarray) -> np.ndarray:
        """
        Processes a raw input image array (RGB or Grayscale).
        
        Steps:
            1. Grayscale Conversion
            2. CLAHE Contrast Enhancement
            3. Proportional Height-Fixed Scaling
            4. Channel Restoration (3-channel RGB)
            
        Returns:
            processed_image (np.ndarray): Enhanced RGB image with standardized height.
        """
        # Step 1: Grayscale Conversion
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif len(image.shape) == 3 and image.shape[2] == 4:
            gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
        else:
            gray = image

        # Step 2: Apply CLAHE Contrast Enhancement
        enhanced = self.clahe.apply(gray)

        # Step 3: Proportional Height-Fixed Scaling
        orig_h, orig_w = enhanced.shape[:2]
        if orig_h <= 0 or orig_w <= 0:
            raise ValueError(f"Invalid image dimensions: {orig_h}x{orig_w}")

        aspect_ratio = orig_w / float(orig_h)
        new_w = int(round(self.target_height * aspect_ratio))
        new_w = max(1, new_w)  # Ensure width is at least 1px

        resized = cv2.resize(enhanced, (new_w, self.target_height), interpolation=cv2.INTER_CUBIC)

        # Step 4: Channel Restoration to 3-channel RGB
        rgb_out = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)

        return rgb_out

    def process_file(self, file_path: str) -> np.ndarray:
        """Loads image file from disk and applies enhancement pipeline."""
        img_bgr = cv2.imread(file_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"Could not load image file: {file_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return self.enhance_and_resize(img_rgb)


def process_dataset_labels(df: pd.DataFrame, text_column: str = 'Target') -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Runs Phase 1 Text Normalization across an entire dataframe.
    
    Returns:
        cleaned_df (pd.DataFrame): Dataframe with cleaned Target strings.
        report (dict): Summary statistics of label cleaning.
    """
    normalizer = TextNormalizer()
    
    cleaned_targets = []
    valid_flags = []
    modified_count = 0
    noise_found_counts = {c: 0 for c in TIER1_NOISE_CHARS}

    for idx, row in df.iterrows():
        raw_text = str(row[text_column]) if pd.notnull(row[text_column]) else ""
        cleaned, is_valid = normalizer.clean_text(raw_text)

        # Track noise characters removed
        for c in TIER1_NOISE_CHARS:
            if c in raw_text:
                noise_found_counts[c] += raw_text.count(c)

        if cleaned != raw_text:
            modified_count += 1

        cleaned_targets.append(cleaned)
        valid_flags.append(is_valid)

    df_result = df.copy()
    df_result['Target_Cleaned'] = cleaned_targets
    df_result['Is_Valid'] = valid_flags

    valid_df = df_result[df_result['Is_Valid']].drop(columns=['Is_Valid']).reset_index(drop=True)
    dropped_ids = df_result[~df_result['Is_Valid']]['ID'].tolist()

    report = {
        'total_rows': len(df),
        'modified_rows': modified_count,
        'dropped_rows': len(dropped_ids),
        'dropped_ids': dropped_ids,
        'noise_character_counts': noise_found_counts
    }

    return valid_df, report
