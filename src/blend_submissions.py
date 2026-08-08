"""
Hybrid Ensembling & Formatting Sanitizer (CRNN + Qwen2-VL)
Blends the ultra-low CER of CRNN with the deep linguistic WER accuracy of Qwen2-VL.
"""

import os
import sys
import re
import pandas as pd
import editdistance

def sanitize_text(text: str) -> str:
    """Removes special prompt artifacts, markdown tags, and normalizes spacing."""
    if not isinstance(text, str) or pd.isna(text):
        return ""
    
    # Strip common VLM prompt leakages
    text = re.sub(r'^(Transcription|Text|Transcribe|Output|Result):\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'<\|im_end\|>|<\|endoftext\|>|<\|im_start\|>', '', text)
    
    # Strip Tier 1 noise characters
    for char in ['\\', '|', '#', '?', '*']:
        text = text.replace(char, '')
        
    # Standardize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def blend_crnn_and_vlm(crnn_csv_path: str, vlm_csv_path: str, output_csv_path: str = "submission_hybrid.csv") -> str:
    print("==================================================================")
    print(" HYBRID BLEND: CRNN (LOW CER) + QWEN2-VL (LOW WER) ")
    print("==================================================================")
    
    crnn_df = pd.read_csv(crnn_csv_path).fillna("")
    vlm_df = pd.read_csv(vlm_csv_path).fillna("")
    
    merged = pd.merge(crnn_df, vlm_df, on="ID", suffixes=("_crnn", "_vlm"))
    
    hybrid_preds = []
    num_crnn_chosen = 0
    num_vlm_chosen = 0
    num_exact_matches = 0
    
    for _, row in merged.iterrows():
        t_crnn = sanitize_text(row["Target_crnn"])
        t_vlm = sanitize_text(row["Target_vlm"])
        
        if t_crnn == t_vlm:
            hybrid_preds.append(t_crnn)
            num_exact_matches += 1
            continue
            
        dist = editdistance.eval(t_crnn, t_vlm)
        len_diff = abs(len(t_crnn) - len(t_vlm))
        
        # If Qwen2-VL produced text of similar length with low edit distance, use Qwen2-VL's language refinement
        if len_diff <= 3 and dist <= 5 and len(t_vlm) > 0:
            hybrid_preds.append(t_vlm)
            num_vlm_chosen += 1
        # If lengths diverge significantly, CRNN's visual bounding is more conservative and reliable
        elif len(t_crnn) > 0:
            hybrid_preds.append(t_crnn)
            num_crnn_chosen += 1
        else:
            hybrid_preds.append(t_vlm)
            num_vlm_chosen += 1
            
    out_df = pd.DataFrame({
        "ID": merged["ID"],
        "Target": hybrid_preds
    })
    
    out_df.to_csv(output_csv_path, index=False)
    print(f"\n[OK] Hybrid submission generated successfully at: {output_csv_path}")
    print(f"  • Exact Matches     : {num_exact_matches:,} ({num_exact_matches/len(merged)*100:.1f}%)")
    print(f"  • Qwen2-VL Refined  : {num_vlm_chosen:,}")
    print(f"  • CRNN Constrained  : {num_crnn_chosen:,}")
    return output_csv_path

if __name__ == '__main__':
    crnn_path = sys.argv[1] if len(sys.argv) > 1 else "submission.csv"
    vlm_path = sys.argv[2] if len(sys.argv) > 2 else "submission_vlm.csv"
    out_path = sys.argv[3] if len(sys.argv) > 3 else "submission_hybrid.csv"
    blend_crnn_and_vlm(crnn_path, vlm_path, out_path)
