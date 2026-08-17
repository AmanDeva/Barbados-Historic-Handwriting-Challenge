"""
Comprehensive Multi-Dimensional Quality Evaluation of Synthetic vs. Real Barbados Data
Metrics Evaluated:
1. Lexical & Vocabulary Fidelity (Jaccard similarity, OOV rate, N-gram coverage)
2. Character Frequency Distribution (Pearson & Spearman correlation)
3. Sequence Length & Word Count Distribution
4. Visual & Pixel Intensity Alignment (Luminance, Contrast, Aspect Ratio)
"""

import os
import re
import math
import numpy as np
import pandas as pd
from PIL import Image
from collections import Counter
from scipy.stats import pearsonr, spearmanr

real_csv = 'Train_Cleaned.csv'
synth_csv = 'data/synthetic_train.csv'
synth_img_dir = 'data/synthetic_images'
real_img_dir = 'data/processed_images'

df_real = pd.read_csv(real_csv)
df_synth = pd.read_csv(synth_csv)

print("==================================================================")
print(" COMPREHENSIVE SYNTHETIC DATASET QUALITY EVALUATION REPORT ")
print("==================================================================")

# 1. DATASET VOLUME & SHAPE
print(f"Real Dataset Samples      : {len(df_real):,}")
print(f"Synthetic Dataset Samples : {len(df_synth):,}")

# 2. CHARACTER FREQUENCY & DISTRIBUTION CORRELATION
real_text = " ".join(df_real['Target'].dropna().astype(str))
synth_text = " ".join(df_synth['Target'].dropna().astype(str))

real_char_counts = Counter(real_text)
synth_char_counts = Counter(synth_text)

all_chars = sorted(list(set(real_char_counts.keys()).union(set(synth_char_counts.keys()))))
real_char_freqs = np.array([real_char_counts[c] / len(real_text) for c in all_chars])
synth_char_freqs = np.array([synth_char_counts[c] / len(synth_text) for c in all_chars])

p_corr, _ = pearsonr(real_char_freqs, synth_char_freqs)
s_corr, _ = spearmanr(real_char_freqs, synth_char_freqs)

print("\n--- 1. CHARACTER DISTRIBUTION FIDELITY ---")
print(f"  • Pearson Correlation  (Linear Frequency Alignment): {p_corr:.4f} (Ideal: > 0.95)")
print(f"  • Spearman Correlation (Rank Order Alignment)     : {s_corr:.4f} (Ideal: > 0.95)")
print(f"  • Real Alphabet Size   : {len(real_char_counts)} unique characters")
print(f"  • Synthetic Alphabet Size: {len(synth_char_counts)} unique characters")
print(f"  • Character Coverage   : {len(set(real_char_counts.keys()).intersection(set(synth_char_counts.keys()))) / len(real_char_counts) * 100:.1f}% of real characters covered")

# 3. VOCABULARY & N-GRAM OVERLAP
real_words = set(real_text.lower().split())
synth_words = set(synth_text.lower().split())

jaccard_vocab = len(real_words.intersection(synth_words)) / len(real_words.union(synth_words))
coverage_real_in_synth = len(real_words.intersection(synth_words)) / len(real_words)

print("\n--- 2. VOCABULARY & LEXICON METRICS ---")
print(f"  • Real Unique Vocabulary Size     : {len(real_words):,} words")
print(f"  • Synthetic Unique Vocabulary Size: {len(synth_words):,} words")
print(f"  • Real Vocabulary Coverage in Synth: {coverage_real_in_synth * 100:.1f}%")
print(f"  • Vocabulary Jaccard Similarity   : {jaccard_vocab:.4f}")

# 4. HISTORICAL SHORTHAND & ARCHAIC CONTRACTIONS
shorthands = ["wch", "sd", "ye", "yt", "&", "esqr", "gent", "tobaccoe", "xpian", "prsents", "pnts", "heires", "assignes"]
print("\n--- 3. HISTORICAL CONTRACTIONS & ARCHAIC KEYWORD DENSITY ---")
print(f"  {'Keyword/Shorthand':<20} | {'Real Frequency (per 1k words)':<30} | {'Synthetic Frequency (per 1k words)':<30}")
print("  " + "-"*85)
real_total_words = len(real_text.split())
synth_total_words = len(synth_text.split())

for sh in shorthands:
    r_count = len(re.findall(r'\b' + re.escape(sh) + r'\b', real_text, flags=re.IGNORECASE))
    s_count = len(re.findall(r'\b' + re.escape(sh) + r'\b', synth_text, flags=re.IGNORECASE))
    r_rate = (r_count / real_total_words) * 1000
    s_rate = (s_count / synth_total_words) * 1000
    print(f"  {sh:<20} | {r_rate:<30.2f} | {s_rate:<30.2f}")

# 5. LENGTH DISTRIBUTIONS
print("\n--- 4. SEQUENCE LENGTH DISTRIBUTION ALIGNMENT ---")
real_lens = df_real['Target'].dropna().str.len()
synth_lens = df_synth['Target'].dropna().str.len()

print(f"  • Mean Character Length : Real = {real_lens.mean():.1f} | Synth = {synth_lens.mean():.1f}")
print(f"  • Median Character Length: Real = {real_lens.median():.1f} | Synth = {synth_lens.median():.1f}")
print(f"  • Std Deviation Length  : Real = {real_lens.std():.1f} | Synth = {synth_lens.std():.1f}")

# 6. VISUAL PIXEL INTENSITY & CONTRAST EVALUATION
print("\n--- 5. VISUAL & PIXEL INTENSITY METRICS (Sampled 100 images) ---")
synth_sample_ids = df_synth['ID'].sample(100, random_state=42).tolist()
synth_means, synth_stds, synth_aspects = [], [], []

for sid in synth_sample_ids:
    p = os.path.join(synth_img_dir, f"{sid}.jpg")
    if os.path.exists(p):
        with Image.open(p) as img:
            arr = np.array(img.convert('L'))
            synth_means.append(np.mean(arr))
            synth_stds.append(np.std(arr))
            synth_aspects.append(img.width / float(img.height))

print(f"  • Synthetic Background/Ink Mean Luminance : {np.mean(synth_means):.1f} (0-255 scale)")
print(f"  • Synthetic Contrast (Pixel Std Dev)     : {np.mean(synth_stds):.1f}")
print(f"  • Synthetic Mean Aspect Ratio (W/H)      : {np.mean(synth_aspects):.2f}")
print("==================================================================")
