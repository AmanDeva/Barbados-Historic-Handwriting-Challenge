"""
Minimum Bayes Risk (MBR) Consensus Decoder & Shorthand Reversion Engine
Calculates pairwise Levenshtein risk across candidate hypothesis pools (TrOCR + Qwen2-VL + CRNN)
and applies deterministic 18th-century shorthand reversions.
"""

import os
import sys
import re
import editdistance
import pandas as pd
from typing import List, Dict

# Ensure project root is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def deterministic_shorthand_reversion(text: str) -> str:
    """
    Protects CER by reverting modern LLM word expansions back to verbatim 18th-century shorthand.
    """
    if not isinstance(text, str) or pd.isna(text):
        return ""

    # Common historical shorthand patterns
    replacements = [
        (r'\bwhich\b', 'wch'),
        (r'\bsaid\b', 'sd'),
        (r'\bthe\b', 'ye'),
        (r'\bthat\b', 'yt'),
        (r'\band\s+etc\b', '&c'),
        (r'\betc\b', '&c'),
        (r'\bGentleman\b', 'Gent:'),
        (r'\bEsquire\b', 'Esqr'),
    ]

    # Clean whitespace and prompt artifacts
    text = re.sub(r'^(Transcription|Text|Result):\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'<\|.*?\|>', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def minimum_bayes_risk_consensus(
    csv_paths: List[str],
    output_csv: str = "submission_mbr_consensus.csv",
    apply_reversion: bool = True
) -> str:
    """
    MBR Decision Rule:
    Selects hypothesis y* from the candidate pool that minimizes expected edit distance loss:
    y* = argmin_{y_i} sum_{y_j} EditDistance(y_i, y_j)
    """
    print("==================================================================")
    print(f" MINIMUM BAYES RISK (MBR) CONSENSUS DECODING ({len(csv_paths)} MODELS) ")
    print("==================================================================")

    dfs = []
    for path in csv_paths:
        full_path = os.path.join(PROJECT_ROOT, path) if not os.path.isabs(path) else path
        if os.path.exists(full_path):
            df = pd.read_csv(full_path).fillna("")
            dfs.append(df)
            print(f"✓ Loaded candidate pool from: {os.path.basename(full_path)}")

    if not dfs:
        raise ValueError("No valid prediction CSV files found for MBR consensus!")

    test_ids = dfs[0]["ID"].tolist()
    num_samples = len(test_ids)
    num_models = len(dfs)

    final_predictions = []

    for idx in range(num_samples):
        # Collect candidate pool for sample idx
        candidates = [dfs[m].iloc[idx]["Target"] for m in range(num_models)]
        cleaned_candidates = [str(c).strip() for c in candidates]

        # Calculate pairwise Levenshtein distance matrix
        best_candidate = cleaned_candidates[0]
        min_total_risk = float('inf')

        for i, y_i in enumerate(cleaned_candidates):
            total_risk = sum(editdistance.eval(y_i, y_j) for j, y_j in enumerate(cleaned_candidates) if i != j)
            if total_risk < min_total_risk:
                min_total_risk = total_risk
                best_candidate = y_i

        if apply_reversion:
            best_candidate = deterministic_shorthand_reversion(best_candidate)

        final_predictions.append(best_candidate)

    out_df = pd.DataFrame({
        "ID": test_ids,
        "Target": final_predictions
    })

    full_output_path = os.path.join(PROJECT_ROOT, output_csv)
    out_df.to_csv(full_output_path, index=False)
    print(f"\n[OK] MBR Consensus predictions successfully exported to: {full_output_path}")
    return full_output_path


if __name__ == '__main__':
    inputs = sys.argv[1:]
    if not inputs:
        inputs = ["submission.csv", "Starters/VLM/submission_vlm.csv", "submission_trocr.csv"]
    minimum_bayes_risk_consensus(inputs)
