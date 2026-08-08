"""
TrOCR Inference Script for generating test predictions on Barbados Historic Handwriting Challenge.
"""

import os
import sys
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

def run_trocr_inference(
    model_path: str = "./trocr-barbados-final/best_model",
    test_csv_path: str = "Test.csv",
    output_csv_path: str = "submission_trocr.csv",
    batch_size: int = 16,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    print("==================================================================")
    print(" RUNNING TrOCR-LARGE INFERENCE ON TEST DATASET ")
    print("==================================================================")

    device = torch.device(device_str)
    full_test_csv = os.path.join(PROJECT_ROOT, test_csv_path) if not os.path.isabs(test_csv_path) else test_csv_path
    full_output_csv = os.path.join(PROJECT_ROOT, output_csv_path) if not os.path.isabs(output_csv_path) else output_csv_path
    img_dir = os.path.join(PROJECT_ROOT, "data", "processed_images")

    test_df = pd.read_csv(full_test_csv)
    print(f"Loaded {len(test_df):,} test samples from {full_test_csv}")

    processor = TrOCRProcessor.from_pretrained(model_path)
    model = VisionEncoderDecoderModel.from_pretrained(model_path).to(device)
    model.eval()

    predictions = []

    for i in tqdm(range(0, len(test_df), batch_size), desc="Generating TrOCR Transcriptions"):
        batch_df = test_df.iloc[i:i+batch_size]
        images = []
        for _, row in batch_df.iterrows():
            img_path = os.path.join(img_dir, f"{str(row['ID']).strip()}.jpg")
            img = Image.open(img_path).convert("RGB")
            images.append(img)

        pixel_values = processor(images, return_tensors="pt").pixel_values.to(device)
        with torch.no_grad():
            generated_ids = model.generate(pixel_values, max_new_tokens=128, num_beams=4)
        
        preds = processor.batch_decode(generated_ids, skip_special_tokens=True)
        predictions.extend([p.strip() for p in preds])

    out_df = pd.DataFrame({
        "ID": test_df["ID"],
        "Target": predictions
    })
    out_df.to_csv(full_output_csv, index=False)
    print(f"\n[OK] TrOCR submission successfully generated at: {full_output_csv}")

if __name__ == '__main__':
    run_trocr_inference()
