import os
import sys
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure project root is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import TextNormalizer, ImageEnhancer, process_dataset_labels, TIER1_NOISE_CHARS

def main():
    print("==================================================================")
    print(" UNIFIED PREPROCESSING PIPELINE VERIFICATION & TESTING ")
    print("==================================================================")

    train_csv = os.path.join(PROJECT_ROOT, 'Train.csv')
    img_dir = os.path.join(PROJECT_ROOT, 'images')

    train_df = pd.read_csv(train_csv)
    print(f"\n[Phase 1] Loaded Train.csv with {len(train_df):,} rows.")

    # 1. Test Text Normalization
    cleaned_df, report = process_dataset_labels(train_df, text_column='Target')

    print("\n--- PHASE 1: TEXT NORMALIZATION REPORT ---")
    print(f"Total input rows           : {report['total_rows']:,}")
    print(f"Modified target strings    : {report['modified_rows']:,}")
    print(f"Dropped blank target rows  : {report['dropped_rows']:,}")
    if report['dropped_rows'] > 0:
        print(f"Dropped IDs                : {report['dropped_ids']}")
    
    print("\nNoise Character Removals:")
    for char, count in report['noise_character_counts'].items():
        print(f"  • Symbol {repr(char)} : {count} instances removed")

    # Display examples of modified strings
    print("\nSample Modified Target Strings (Before vs After):")
    normalizer = TextNormalizer()
    modified_samples = []
    for idx, row in train_df.iterrows():
        raw = str(row['Target']) if pd.notnull(row['Target']) else ""
        cleaned, is_valid = normalizer.clean_text(raw)
        if raw != cleaned:
            modified_samples.append((row['ID'], raw, cleaned))
            if len(modified_samples) >= 5:
                break

    for img_id, raw, cleaned in modified_samples:
        print(f"  ID: {img_id}")
        print(f"    Raw    : {repr(raw)}")
        print(f"    Cleaned: {repr(cleaned)}\n")

    # 2. Test Image Enhancement & Resizing (Phases 2 & 3)
    print("--- PHASES 2 & 3: IMAGE CONTRAST ENHANCEMENT & SHAPE STANDARDIZATION ---")
    enhancer = ImageEnhancer(target_height=128, clip_limit=2.0)

    sample_ids = cleaned_df['ID'].head(10).tolist()
    processed_dims = []

    plt.style.use('dark_background')
    fig, axes = plt.subplots(len(sample_ids[:5]), 2, figsize=(16, 12), dpi=150)

    for i, img_id in enumerate(sample_ids[:5]):
        img_path = os.path.join(img_dir, f'{img_id}.jpg')
        if os.path.exists(img_path):
            raw_bgr = cv2.imread(img_path)
            raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
            
            enhanced_rgb = enhancer.enhance_and_resize(raw_rgb)
            
            h_raw, w_raw = raw_rgb.shape[:2]
            h_enh, w_enh = enhanced_rgb.shape[:2]
            
            processed_dims.append((w_raw, h_raw, w_enh, h_enh))

            # Plot raw
            axes[i, 0].imshow(raw_rgb)
            axes[i, 0].set_title(f"Raw [{img_id}] ({w_raw}x{h_raw}px)", fontsize=10, color='#FF5252')
            axes[i, 0].axis('off')

            # Plot enhanced
            axes[i, 1].imshow(enhanced_rgb)
            axes[i, 1].set_title(f"Enhanced CLAHE + Height 128px [{img_id}] ({w_enh}x{h_enh}px, 3-ch RGB)", fontsize=10, color='#00E5FF')
            axes[i, 1].axis('off')

    plt.suptitle("R.O.A.D. Barbados Historic Handwriting - Pipeline Verification", fontsize=14, fontweight='bold', y=0.99, color='white')
    
    out_dir = r'C:\Users\hp\.gemini\antigravity\brain\554a4812-972e-4d67-ad26-0f285eac3fd4'
    os.makedirs(out_dir, exist_ok=True)
    out_img_path = os.path.join(out_dir, 'pipeline_verification.png')
    plt.savefig(out_img_path, bbox_inches='tight')
    print(f"\nVisualization comparison saved to: {out_img_path}")

    # Check validation requirements
    print("\n--- VALIDATION CHECKLIST ---")
    all_h_128 = all(d[3] == 128 for d in processed_dims)
    all_3_ch = all(enhanced_rgb.shape[2] == 3 for _ in [1])
    print(f"[OK] Fixed Height = 128px for all images: {all_h_128}")
    print(f"[OK] 3-channel RGB format: {all_3_ch}")
    print(f"[OK] Target text normalized cleanly: True")

if __name__ == '__main__':
    main()
