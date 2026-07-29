import os
import sys
import cv2
import numpy as np
import pandas as pd
from PIL import Image

# Ensure project root is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import process_dataset_labels, ImageEnhancer

def main():
    print("==================================================================")
    print(" EXPORTING PROCESSED DATASET (METADATA & PREPROCESSED IMAGES) ")
    print("==================================================================")

    data_output_dir = os.path.join(PROJECT_ROOT, 'data')
    processed_img_dir = os.path.join(data_output_dir, 'processed_images')
    os.makedirs(processed_img_dir, exist_ok=True)

    train_csv = os.path.join(PROJECT_ROOT, 'Train.csv')
    test_csv = os.path.join(PROJECT_ROOT, 'Test.csv')
    img_dir = os.path.join(PROJECT_ROOT, 'images')

    # 1. Process Train Labels & Dimensions
    print(f"\n[1/3] Processing Train dataset...")
    train_df = pd.read_csv(train_csv)
    cleaned_train_df, report = process_dataset_labels(train_df, text_column='Target')

    train_meta = []
    enhancer = ImageEnhancer(target_height=128, clip_limit=2.0)

    for idx, row in cleaned_train_df.iterrows():
        img_id = row['ID']
        target = row['Target_Cleaned']
        img_path = os.path.join(img_dir, f'{img_id}.jpg')
        if os.path.exists(img_path):
            with Image.open(img_path) as img:
                w, h = img.size
                ar = w / float(h)
                scaled_w = int(round(128 * ar))
                train_meta.append({
                    'ID': img_id,
                    'Target': target,
                    'orig_width': w,
                    'orig_height': h,
                    'aspect_ratio': round(ar, 4),
                    'scaled_width': scaled_w,
                    'scaled_height': 128
                })

    df_train_out = pd.DataFrame(train_meta)
    train_out_csv = os.path.join(data_output_dir, 'processed_train.csv')
    df_train_out.to_csv(train_out_csv, index=False)
    print(f"[OK] Saved processed train labels and metadata to: {train_out_csv} ({len(df_train_out):,} rows)")

    # 2. Process Test Dimensions
    print(f"\n[2/3] Processing Test dataset...")
    test_df = pd.read_csv(test_csv)
    test_meta = []

    for idx, row in test_df.iterrows():
        img_id = row['ID']
        img_path = os.path.join(img_dir, f'{img_id}.jpg')
        if os.path.exists(img_path):
            with Image.open(img_path) as img:
                w, h = img.size
                ar = w / float(h)
                scaled_w = int(round(128 * ar))
                test_meta.append({
                    'ID': img_id,
                    'orig_width': w,
                    'orig_height': h,
                    'aspect_ratio': round(ar, 4),
                    'scaled_width': scaled_w,
                    'scaled_height': 128
                })

    df_test_out = pd.DataFrame(test_meta)
    test_out_csv = os.path.join(data_output_dir, 'processed_test.csv')
    df_test_out.to_csv(test_out_csv, index=False)
    print(f"[OK] Saved processed test metadata to: {test_out_csv} ({len(df_test_out):,} rows)")

    # 3. Export Sample Preprocessed Images
    print(f"\n[3/3] Pre-rendering first 100 sample images to data/processed_images/...")
    for idx, row in df_train_out.head(100).iterrows():
        img_id = row['ID']
        src_path = os.path.join(img_dir, f'{img_id}.jpg')
        dst_path = os.path.join(processed_img_dir, f'{img_id}.jpg')
        if os.path.exists(src_path) and not os.path.exists(dst_path):
            enh_rgb = enhancer.process_file(src_path)
            enh_bgr = cv2.cvtColor(enh_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(dst_path, enh_bgr)

    print(f"[OK] Saved sample enhanced images to: {processed_img_dir}")
    print("\nPROCESSED DATA EXPORT COMPLETED SUCCESSFULLY!")

if __name__ == '__main__':
    main()
