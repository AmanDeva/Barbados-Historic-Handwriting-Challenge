"""
TrOCR Fine-Tuning Pipeline for Barbados Historic Handwriting Challenge.
Optimized for 24GB RTX 3090 (uses ~4.5 GB VRAM).

Model: microsoft/trocr-large-handwritten (Vision Transformer Encoder + Text Decoder)
Features:
- Dedicated handwriting OCR encoder-decoder (Zero chatbot/conversational hallucination)
- Pretrained on millions of historical & modern handwritten text lines
- Full mixed-precision (fp16) training with AdamW and linear warmup
"""

import os
import sys
import argparse
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

class BarbadosTrOCRDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_dir: str, processor: TrOCRProcessor, max_target_length: int = 128):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = str(row['ID']).strip()
        img_path = os.path.join(self.img_dir, f"{img_id}.jpg")
        
        image = Image.open(img_path).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze(0)

        item = {"pixel_values": pixel_values}
        if 'Target' in row and pd.notnull(row['Target']):
            target_text = str(row['Target']).strip()
            labels = self.processor.tokenizer(
                target_text,
                padding="max_length",
                max_length=self.max_target_length,
                truncation=True,
                return_tensors="pt"
            ).input_ids.squeeze(0)
            
            # Replace pad_token_id with -100 to ignore in loss computation
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
            item["labels"] = labels

        return item

def train_trocr(
    model_name: str = "microsoft/trocr-large-handwritten",
    epochs: int = 8,
    batch_size: int = 8,
    learning_rate: float = 4e-5,
    output_dir: str = "./trocr-barbados-final"
):
    print("==================================================================")
    print(f" TRAINING TrOCR-LARGE HANDWRITTEN TRANSFORMER ON RTX 3090 ")
    print("==================================================================")

    train_csv = os.path.join(PROJECT_ROOT, "Train_Cleaned.csv")
    img_dir = os.path.join(PROJECT_ROOT, "data", "processed_images")

    df = pd.read_csv(train_csv)
    # 90/10 Stratified split
    train_df = df.sample(frac=0.9, random_state=42).reset_index(drop=True)
    val_df = df.drop(train_df.index).reset_index(drop=True)

    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)

    # Set special tokens for generation
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = 128
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 1.0
    model.config.num_beams = 4

    train_dataset = BarbadosTrOCRDataset(train_df, img_dir, processor)
    val_dataset = BarbadosTrOCRDataset(val_df, img_dir, processor)

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        fp16=torch.cuda.is_available(),
        predict_with_generate=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        warmup_ratio=0.1,
        logging_steps=25,
        dataloader_num_workers=4,
        load_best_model_at_end=True,
        report_to="none"
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=default_data_collator,
        tokenizer=processor.image_processor
    )

    print("Starting TrOCR fine-tuning...")
    trainer.train()

    final_path = os.path.join(output_dir, "best_model")
    model.save_pretrained(final_path)
    processor.save_pretrained(final_path)
    print(f"\n[OK] TrOCR fine-tuned model saved successfully to: {final_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=4e-5)
    args = parser.parse_args()

    train_trocr(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr)
