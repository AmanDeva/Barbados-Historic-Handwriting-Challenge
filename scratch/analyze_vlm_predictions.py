"""
VLM Historical OCR Research Diagnostic & Sanitizer
Performs deep linguistic error analysis on Qwen2-VL predictions:
1. Strips hallucinated prompt echoes & markdown formatting
2. Normalizes non-verbatim whitespace and trailing punctuation
3. Maps common modern LLM abbreviation expansions back to 18th-century verbatim forms
"""

import os
import sys
import re
import pandas as pd

def research_sanitize_vlm_transcription(text: str) -> str:
    if not isinstance(text, str) or pd.isna(text):
        return ""
    
    # 1. Strip ChatML, Markdown, and System Prompt Artifacts
    text = re.sub(r'^(Transcription|Transcript|Text|Output|Result|Image):\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'<\|im_end\|>|<\|endoftext\|>|<\|im_start\|>|<\|.*?\|>', '', text)
    text = re.sub(r'^["\'`](.*)["\'`]$', r'\1', text) # strip wrapping quotes
    
    # 2. Remove Tier 1 EDA noise symbols
    for noise in ['\\', '|', '#', '?', '*']:
        text = text.replace(noise, '')
        
    # 3. Fix Common Modern LLM Artifacts on 18th Century Text:
    # LLMs frequently hallucinate modern quotes/ellipses for ink smudges
    text = text.replace('…', '...').replace('“', '"').replace('”', '"').replace('’', "'")
    
    # 4. Standardize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def analyze_vlm_csv(vlm_csv_path: str, output_csv_path: str = "submission_vlm_cleaned.csv"):
    df = pd.read_csv(vlm_csv_path)
    print(f"Loaded {len(df):,} predictions from {vlm_csv_path}")
    
    cleaned = []
    modified_count = 0
    empty_count = 0
    
    for _, row in df.iterrows():
        raw = str(row['Target']) if pd.notnull(row['Target']) else ""
        sanitized = research_sanitize_vlm_transcription(raw)
        
        if sanitized != raw:
            modified_count += 1
        if len(sanitized) == 0:
            empty_count += 1
            
        cleaned.append(sanitized)
        
    out_df = pd.DataFrame({
        'ID': df['ID'],
        'Target': cleaned
    })
    out_df.to_csv(output_csv_path, index=False)
    
    print("\n--- RESEARCH SANITIZATION SUMMARY ---")
    print(f"• Total Rows               : {len(df):,}")
    print(f"• Artifacts Cleaned/Fixed  : {modified_count:,} ({modified_count/len(df)*100:.1f}%)")
    print(f"• Empty Transcriptions     : {empty_count:,}")
    print(f"• Output Saved At          : {output_csv_path}")

if __name__ == '__main__':
    vlm_path = sys.argv[1] if len(sys.argv) > 1 else "submission_vlm.csv"
    analyze_vlm_csv(vlm_path)
