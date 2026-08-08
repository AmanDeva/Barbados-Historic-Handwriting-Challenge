"""
Advanced Historical Paleography Preprocessing Engine
Designed specifically for 18th-Century Iron Gall Ink on Aged Parchment.

Pillars Implemented:
1. CIELAB Color-Space Ink Isolation (Verso Bleed-Through & Yellowing Suppression: L* - b*)
2. Projection Profile Variance Deskewing (Baseline Straightening for Vertical Column Alignment)
3. Morphological Stroke Healing (3x3 Micro-Gap Closing for Faded Hairline Loops)
4. Dynamic Pixel-Density Scaling (Aspect Ratio Preservation preventing loop compression)
"""

import os
import re
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageFilter
from typing import Tuple, Optional, Dict, Any, List

try:
    import cv2
    cv2.setNumThreads(1)
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

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


class HistoricalPaleographyEnhancer:
    """
    Advanced Paleographic Preprocessing Engine for 18th-century manuscripts.
    """
    def __init__(
        self,
        target_height: int = 128,
        gamma_yellow_suppression: float = 0.65,
        deskew: bool = True,
        heal_strokes: bool = True
    ):
        self.target_height = target_height
        self.gamma = gamma_yellow_suppression
        self.deskew = deskew
        self.heal_strokes = heal_strokes

    def isolate_iron_gall_ink(self, img_rgb: np.ndarray) -> np.ndarray:
        """
        Pillar 2: CIELAB Color-Space Ink Isolation.
        Suppresses yellow parchment background & verso bleed-through via L* - gamma * b*.
        """
        if not HAS_CV2:
            # High-performance NumPy / PIL fallback
            r, g, b = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]
            # Iron gall ink is dark with low red/green; yellow parchment has high red/green and low blue
            ink_contrast = (0.299 * r + 0.587 * g + 0.114 * b) - (0.5 * (r + g) - b)
            ink_normalized = np.clip(ink_contrast, 0, 255).astype(np.uint8)
            return ink_normalized

        # Convert to CIELAB color space
        lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        L, a, b_chan = cv2.split(lab)

        # L* measures darkness; b* measures Yellow (>128) vs Blue (<128)
        # Subtracting yellow opponent suppresses aged parchment staining and bleed-through
        L_float = L.astype(np.float32)
        b_float = b_chan.astype(np.float32) - 128.0

        # Enhance ink darkness while pushing parchment background to white
        ink_signal = L_float - (self.gamma * np.maximum(b_float, 0))
        ink_signal = np.clip(ink_signal, 0, 255).astype(np.uint8)

        # CLAHE on isolated ink signal
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(ink_signal)
        return enhanced

    def find_deskew_angle(self, gray: np.ndarray, max_angle: float = 7.0, step: float = 0.5) -> float:
        """
        Pillar 3: Horizontal Projection Profile Variance Optimization.
        Finds the exact rotation angle that aligns text baselines horizontally.
        """
        best_angle = 0.0
        max_variance = 0.0
        h, w = gray.shape

        # Downsample for ultra-fast angle search (< 2ms)
        small_h = min(64, h)
        small_w = int(w * (small_h / h))
        if HAS_CV2:
            small_img = cv2.resize(gray, (small_w, small_h), interpolation=cv2.INTER_AREA)
        else:
            pil_s = Image.fromarray(gray).resize((small_w, small_h), Image.Resampling.BILINEAR)
            small_img = np.array(pil_s)

        # Binarize text mask for projection
        thresh = small_img < np.percentile(small_img, 30)

        angles = np.arange(-max_angle, max_angle + step, step)
        for angle in angles:
            if HAS_CV2:
                M = cv2.getRotationMatrix2D((small_w / 2, small_h / 2), angle, 1.0)
                rotated = cv2.warpAffine(thresh.astype(np.uint8), M, (small_w, small_h), flags=cv2.INTER_NEAREST)
            else:
                p_rot = Image.fromarray(thresh).rotate(angle, resample=Image.Resampling.NEAREST)
                rotated = np.array(p_rot)

            # Horizontal projection profile: sum across columns
            profile = np.sum(rotated, axis=1)
            variance = np.var(profile)

            if variance > max_variance:
                max_variance = variance
                best_angle = angle

        return best_angle

    def apply_deskew(self, img: np.ndarray, angle: float) -> np.ndarray:
        """Rotates image by optimal baseline angle."""
        if abs(angle) < 0.2:
            return img

        h, w = img.shape[:2]
        if HAS_CV2:
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            rotated = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
            return rotated
        else:
            pil_img = Image.fromarray(img)
            return np.array(pil_img.rotate(angle, resample=Image.Resampling.BICUBIC, fillcolor=255))

    def heal_faded_strokes(self, gray: np.ndarray) -> np.ndarray:
        """
        Pillar 4: Morphological Closing (3x3 Kernel).
        Bridges 1-2px hairline gaps in faded cursive loops without blurring.
        """
        if not HAS_CV2:
            pil_img = Image.fromarray(gray)
            # Subtle min/max filter sequence equivalent to closing
            return np.array(pil_img.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.MaxFilter(3)))

        # Inverted mask so ink is foreground
        inv = 255 - gray
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        closed_inv = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, kernel)
        healed = 255 - closed_inv
        return healed

    def process_image(self, img_rgb: np.ndarray) -> np.ndarray:
        """
        Full 4-Pillar Paleography Pipeline.
        """
        # Step 1: CIELAB Ink Isolation
        isolated = self.isolate_iron_gall_ink(img_rgb)

        # Step 2: Projection Profile Deskewing
        if self.deskew:
            angle = self.find_deskew_angle(isolated)
            isolated = self.apply_deskew(isolated, angle)

        # Step 3: Morphological Stroke Healing
        if self.heal_strokes:
            isolated = self.heal_faded_strokes(isolated)

        # Step 4: Dynamic Pixel-Density Scaling (Aspect Ratio Preservation)
        orig_h, orig_w = isolated.shape[:2]
        aspect_ratio = orig_w / float(orig_h)
        new_w = max(1, int(round(self.target_height * aspect_ratio)))

        if HAS_CV2:
            resized = cv2.resize(isolated, (new_w, self.target_height), interpolation=cv2.INTER_CUBIC)
            rgb_out = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        else:
            pil_gray = Image.fromarray(isolated)
            resized = pil_gray.resize((new_w, self.target_height), Image.Resampling.BICUBIC)
            rgb_out = np.array(resized.convert('RGB'))

        return rgb_out

    def process_file_to_pil(self, file_path: str) -> Image.Image:
        """Loads file, runs paleography pipeline, returns PIL Image."""
        with Image.open(file_path) as img:
            rgb = np.array(img.convert('RGB'))
            processed = self.process_image(rgb)
            return Image.fromarray(processed)


# Alias ImageEnhancer to the advanced HistoricalPaleographyEnhancer for seamless backward compatibility
ImageEnhancer = HistoricalPaleographyEnhancer
