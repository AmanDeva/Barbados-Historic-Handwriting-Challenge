"""
Unified Preprocessing and Text Cleaning Pipeline for R.O.A.D. Barbados Historic Handwriting Challenge.
Uses pure PIL & NumPy fallback for 100% thread-safe Docker execution with zero segfaults.
"""

import os
import re
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from typing import Tuple, Optional, Dict, Any, List

# Tier 1 Noise Characters identified during EDA
TIER1_NOISE_CHARS = ['\\', '|', '#', '?', '*']
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
        if not isinstance(text, str) or pd.isnull(text):
            return "", False

        cleaned = self.pattern.sub('', text)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = cleaned.strip()
        is_valid = len(cleaned) > 0

        return cleaned, is_valid


class ImageEnhancer:
    """
    Phases 2 & 3: Image Contrast Enhancement and Dimensional Standardization.
    Uses pure PIL & ImageOps for 100% thread-safe execution across all Linux/Docker environments.
    """
    def __init__(self, target_height: int = 128, clip_limit: float = 2.0):
        self.target_height = target_height
        self.clip_limit = clip_limit

    def process_file_to_pil(self, file_path: str) -> Image.Image:
        """Loads image file and applies proportional 128px scaling and contrast enhancement."""
        with Image.open(file_path) as img:
            gray = img.convert('L')
            enhanced = ImageOps.autocontrast(gray, cutoff=0.5)

            orig_w, orig_h = enhanced.size
            if orig_h <= 0 or orig_w <= 0:
                raise ValueError(f"Invalid image dimensions: {orig_w}x{orig_h}")

            aspect_ratio = orig_w / float(orig_h)
            new_w = max(1, int(round(self.target_height * aspect_ratio)))

            resized = enhanced.resize((new_w, self.target_height), resample=Image.Resampling.BICUBIC)
            rgb_out = resized.convert('RGB')
            return rgb_out

    def process_file(self, file_path: str) -> np.ndarray:
        """Loads image file and returns numpy RGB array."""
        pil_img = self.process_file_to_pil(file_path)
        return np.array(pil_img)
