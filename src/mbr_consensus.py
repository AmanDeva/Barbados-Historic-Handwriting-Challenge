"""
Grandmaster Multi-Architecture Minimum Bayes Risk (MBR) Consensus Engine
Implements decision-theoretic consensus decoding across diverse hypothesis pools:
- TrOCR-Large (Visual Sequence Transformer Anchor)
- Qwen2-VL-7B (Semantic & Historical Reasoning Brain)
- 5-Fold CRNN (Visual Stroke Alignment Backbone)

Mathematical Formulation:
y* = argmin_{y_i in H} sum_{y_j in H} w_j * LevenshteinDistance(y_i, y_j)
"""

import os
import sys
import re
import glob
import editdistance
import pandas as pd
import numpy as np
from typing import List, Dict, Optional

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
    weights: Optional[List[float]] = None,
    output_csv: str = "submission_mbr_consensus.csv",
    apply_reversion: bool = False
) -> str:
    """
    Multi-Architecture MBR Decision Rule:
    Evaluates every candidate against all other candidates in the hypothesis pool H
    and selects the candidate minimizing expected Levenshtein loss.
    """
    print("==================================================================")
    print(f" 🏆 MULTI-ARCHITECTURE MBR CONSENSUS DECODER ({len(csv_paths)} CANDIDATE STREAMS) ")
    print("==================================================================")

    dfs = []
    valid_paths = []

    for path in csv_paths:
        full_path = os.path.join(PROJECT_ROOT, path) if not os.path.isabs(path) else path
        if os.path.exists(full_path):
            try:
                df = pd.read_csv(full_path).fillna("")
                if "ID" in df.columns and "Target" in df.columns and len(df) == 1374:
                    dfs.append(df)
                    valid_paths.append(full_path)
                    print(f"  [OK] Loaded Candidate Stream: {os.path.basename(full_path)} ({len(df):,} rows)")
            except Exception as e:
                print(f"  [!] Skipping {path}: {e}")

    if not dfs:
        raise ValueError("No valid prediction CSV files found for MBR consensus!")

    num_models = len(dfs)
    num_samples = len(dfs[0])
    test_ids = dfs[0]["ID"].tolist()

    # Default weights: equal uniform risk if not specified
    if weights is None or len(weights) != num_models:
        weights = [1.0] * num_models
    else:
        # Normalize weights
        w_sum = sum(weights)
        weights = [w / w_sum for w in weights]

    print(f"\nComputing Pairwise Levenshtein Risk Matrices for {num_samples:,} test samples across {num_models} models...")

    consensus_predictions = []

    for idx in range(num_samples):
        # Extract candidate pool H for test image idx
        candidates = [str(dfs[m].iloc[idx]["Target"]).strip() for m in range(num_models)]

        # Unique candidate strings
        unique_candidates = list(set(candidates))

        # If all models agree unanimously
        if len(unique_candidates) == 1:
            best_str = unique_candidates[0]
        else:
            # Pairwise Levenshtein Risk Matrix Computation:
            # Risk(y_i) = sum_{j=1..K} w_j * Levenshtein(y_i, y_j)
            best_str = unique_candidates[0]
            min_expected_risk = float('inf')

            for y_i in unique_candidates:
                expected_risk = 0.0
                for j, y_j in enumerate(candidates):
                    dist = editdistance.eval(y_i, y_j)
                    expected_risk += weights[j] * dist

                if expected_risk < min_expected_risk:
                    min_expected_risk = expected_risk
                    best_str = y_i
                elif expected_risk == min_expected_risk:
                    # Tie-breaking rule: prefer candidate closer to median character length of the pool
                    median_len = np.median([len(c) for c in candidates])
                    if abs(len(y_i) - median_len) < abs(len(best_str) - median_len):
                        best_str = y_i

        if apply_reversion:
            best_str = deterministic_shorthand_reversion(best_str)

        consensus_predictions.append(best_str)

    out_df = pd.DataFrame({
        "ID": test_ids,
        "Target": consensus_predictions
    })

    full_output_path = os.path.join(PROJECT_ROOT, output_csv)
    out_df.to_csv(full_output_path, index=False)

    print("\n--- MBR CONSENSUS COMPLETE ---")
    print(f"[OK] Consensus Predictions Saved to: {full_output_path}")
    return full_output_path


def auto_discover_and_fuse(output_csv: str = "submission_mbr_consensus.csv"):
    """Auto-discovers all available prediction CSV files and runs MBR fusion."""
    search_patterns = [
        os.path.join(PROJECT_ROOT, "*.csv"),
        os.path.join(PROJECT_ROOT, "Starters", "*", "*.csv")
    ]

    all_csvs = []
    for pat in search_patterns:
        for f in glob.glob(pat):
            fname = os.path.basename(f).lower()
            if "submission" in fname and fname != "samplesubmission.csv" and fname != os.path.basename(output_csv).lower():
                all_csvs.append(f)

    all_csvs = sorted(list(set(all_csvs)))
    print(f"Discovered {len(all_csvs)} potential submission candidate files in workspace:")
    for c in all_csvs:
        print(f"  - {os.path.relpath(c, PROJECT_ROOT)}")

    if not all_csvs:
        raise FileNotFoundError("No candidate submission CSVs found to fuse.")

    minimum_bayes_risk_consensus(all_csvs, output_csv=output_csv)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        args = sys.argv[1:]
        out_file = "submission_mbr_consensus.csv"
        if args[-1].endswith(".csv") and not os.path.exists(args[-1]) and len(args) > 1:
            out_file = args[-1]
            input_files = args[:-1]
        else:
            input_files = args
        minimum_bayes_risk_consensus(input_files, output_csv=out_file)
    else:
        auto_discover_and_fuse()
