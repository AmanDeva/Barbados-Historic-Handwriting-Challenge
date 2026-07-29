"""
PyTorch Dataset, Tokenizer, Data Augmentations, and Custom Collate Function 
for R.O.A.D. Barbados Historic Handwriting Challenge (GPU & SageMaker Ready).

Features:
- Phase 1: Vocabulary & Token Encoding (Tokenizer with [BLANK] at index 0)
- Phase 2: PyTorch Dataset (BarbadosOCRDataset with ImageNet normalization & Handwriting Data Augmentations)
- Phase 3: Custom Collate Function (Dynamic batch width padding & CTC length tracking)
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from typing import List, Tuple, Dict, Any, Optional, Union

# Standard ImageNet Normalization Constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class Tokenizer:
    """
    Bidirectional Vocabulary Tokenizer for CTC Loss.
    Index 0 is strictly reserved for the CTC [BLANK] token.
    """
    def __init__(self, chars: Optional[List[str]] = None):
        self.blank_token = "[BLANK]"
        self.blank_idx = 0

        if chars is None:
            chars = sorted(list(
                " #&'()*+,-.0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXY\\^_abcdefghijklmnopqrstuvwxyz|~"
            ))

        noise = {'\\', '|', '#', '?', '*'}
        valid_chars = sorted(list(set(chars) - noise))

        self.id2char = {0: self.blank_token}
        self.char2id = {self.blank_token: 0}

        for idx, char in enumerate(valid_chars, start=1):
            self.id2char[idx] = char
            self.char2id[char] = idx

    def __len__(self) -> int:
        return len(self.id2char)

    def encode(self, text: str) -> List[int]:
        """Converts string text into list of integer token IDs."""
        tokens = []
        for char in str(text):
            if char in self.char2id:
                tokens.append(self.char2id[char])
        return tokens

    def decode(self, tokens: List[int], remove_ctc_duplicates: bool = True) -> str:
        """
        Decodes sequence of integer token IDs back into string text.
        Optionally collapses consecutive CTC duplicate tokens and removes [BLANK] (0).
        """
        decoded_chars = []
        prev_token = None

        for token in tokens:
            if isinstance(token, torch.Tensor):
                token = token.item()

            if remove_ctc_duplicates:
                if token != self.blank_idx and token != prev_token:
                    if token in self.id2char:
                        decoded_chars.append(self.id2char[token])
                prev_token = token
            else:
                if token != self.blank_idx and token in self.id2char:
                    decoded_chars.append(self.id2char[token])

        return "".join(decoded_chars)


def get_transforms(is_train: bool = True) -> transforms.Compose:
    """
    Returns image transformation pipeline.
    Includes handwriting-specific data augmentations for training.
    """
    if is_train:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(degrees=(-3, 3), fill=255),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])


class BarbadosOCRDataset(Dataset):
    """
    PyTorch Dataset for Barbados Historic Line Images.
    Loads preprocessed images (128px height), applies data augmentations,
    normalizes pixels, and returns tokenized target tensors.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        img_dir: str,
        tokenizer: Tokenizer,
        is_train: bool = True,
        transform: Optional[transforms.Compose] = None
    ):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.is_train = is_train

        if transform is None:
            self.transform = get_transforms(is_train=is_train)
        else:
            self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        img_id = row['ID']
        img_path = os.path.join(self.img_dir, f"{img_id}.jpg")

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found at path: {img_path}")

        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError(f"Failed to read image file: {img_path}")
            
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w, c = img_rgb.shape

        # Apply image transform
        img_tensor = self.transform(img_rgb)

        target_text = ""
        target_tokens = []

        if 'Target' in row and pd.notnull(row['Target']):
            target_text = str(row['Target'])
            target_tokens = self.tokenizer.encode(target_text)

        target_tensor = torch.tensor(target_tokens, dtype=torch.long)

        return {
            'image': img_tensor,           # Tensor (3, 128, W)
            'target': target_tensor,       # Tensor (L,)
            'target_text': target_text,
            'target_length': len(target_tokens),
            'img_id': img_id,
            'width': w,
            'height': h
        }


class OCRCollateFn:
    """
    Custom Collate Function.
    Pads variable-width images to batch max width, concatenates targets for CTC loss,
    and calculates CNN downsampled input sequence lengths.
    """
    def __init__(self, downsample_factor: int = 4, pad_value: float = 0.0):
        self.downsample_factor = downsample_factor
        self.pad_value = pad_value

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch_size = len(batch)
        
        widths = [item['width'] for item in batch]
        max_width = max(widths)
        
        if max_width % self.downsample_factor != 0:
            max_width = ((max_width // self.downsample_factor) + 1) * self.downsample_factor

        padded_images = torch.full((batch_size, 3, 128, max_width), self.pad_value, dtype=torch.float32)

        targets_list = []
        target_lengths = []
        input_lengths = []
        img_ids = []
        target_texts = []

        for i, item in enumerate(batch):
            img_tensor = item['image']  # (3, 128, W)
            w = item['width']
            
            padded_images[i, :, :, :w] = img_tensor

            input_len = int(np.ceil(w / float(self.downsample_factor)))
            input_lengths.append(input_len)

            targets_list.append(item['target'])
            target_lengths.append(item['target_length'])
            img_ids.append(item['img_id'])
            target_texts.append(item['target_text'])

        if targets_list and len(torch.cat(targets_list)) > 0:
            targets_flat = torch.cat(targets_list)
        else:
            targets_flat = torch.tensor([], dtype=torch.long)

        input_lengths_tensor = torch.tensor(input_lengths, dtype=torch.long)
        target_lengths_tensor = torch.tensor(target_lengths, dtype=torch.long)

        return {
            'images': padded_images,                   # (B, 3, 128, W_max)
            'targets': targets_flat,                   # 1D Tensor of concatenated tokens
            'input_lengths': input_lengths_tensor,     # (B,) CTC input sequence lengths
            'target_lengths': target_lengths_tensor,   # (B,) Ground-truth text lengths
            'img_ids': img_ids,
            'target_texts': target_texts,
            'max_width': max_width
        }
