import os
import sys
import cv2
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image

# Disable OpenCV internal multithreading to prevent thread pool segfaults
cv2.setNumThreads(1)

# Ensure project root is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import ImageEnhancer

def process_single_image(img_id, src_dir, dst_dir, enhancer):
    src_path = os.path.join(src_dir, f"{img_id}.jpg")
    dst_path = os.path.join(dst_dir, f"{img_id}.jpg")

    if not os.path.exists(src_path):
        return img_id, False, "Source file not found"

    try:
        enhanced_rgb = enhancer.process_file(src_path)
        enhanced_bgr = cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(dst_path, enhanced_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return img_id, True, "Success"
    except Exception as e:
        return img_id, False, str(e)

def main():
    print("==================================================================")
    print(" BATCH PRE-PROCESSING ALL 5,472 IMAGES (TRAIN + TEST) ")
    print("==================================================================")

    src_img_dir = os.path.join(PROJECT_ROOT, 'images')
    dst_img_dir = os.path.join(PROJECT_ROOT, 'data', 'processed_images')
    os.makedirs(dst_img_dir, exist_ok=True)

    train_csv = os.path.join(PROJECT_ROOT, 'Train.csv')
    test_csv = os.path.join(PROJECT_ROOT, 'Test.csv')

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    all_ids = sorted(list(set(train_df['ID'].tolist() + test_df['ID'].tolist())))
    print(f"Total unique images to process: {len(all_ids):,} (Train: {len(train_df):,}, Test: {len(test_df):,})")

    enhancer = ImageEnhancer(target_height=128, clip_limit=2.0)

    success_count = 0
    fail_count = 0
    errors = []

    # Safe worker thread pool
    num_workers = 4
    print(f"Starting parallel processing with {num_workers} worker threads...\n")

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(process_single_image, img_id, src_img_dir, dst_img_dir, enhancer): img_id
            for img_id in all_ids
        }

        completed = 0
        for future in as_completed(futures):
            completed += 1
            img_id, success, msg = future.result()
            if success:
                success_count += 1
            else:
                fail_count += 1
                errors.append((img_id, msg))

            if completed % 500 == 0 or completed == len(all_ids):
                print(f"Progress: {completed:,} / {len(all_ids):,} ({completed/len(all_ids)*100:.1f}%) processed...")

    print("\n--- BATCH PROCESSING SUMMARY ---")
    print(f"[OK] Total Successfully Processed & Saved : {success_count:,}")
    print(f"[OK] Total Destination Images in Directory: {len(os.listdir(dst_img_dir)):,}")
    print(f"Failed Count                           : {fail_count}")

    if errors:
        print("\nErrors encountered:")
        for eid, emsg in errors[:10]:
            print(f"  ID: {eid} -> {emsg}")

if __name__ == '__main__':
    main()
