"""
Official Zindi Evaluation Metric Engine for R.O.A.D. Barbados Historic Handwriting Challenge.

Metrics:
- Weighted Character Error Rate (Weighted CER): Sum of char edit distances / Sum of reference char lengths
- Weighted Word Error Rate (Weighted WER): Sum of word edit distances / Sum of reference word counts
- Final Competition Metric: 0.7 * Weighted CER + 0.3 * Weighted WER
"""

import os
import sys
import pandas as pd
from typing import List, Tuple, Dict, Any, Union

def levenshtein_distance(seq1: Union[str, List[str]], seq2: Union[str, List[str]]) -> int:
    """
    Computes Levenshtein edit distance between two sequences (strings or lists of words).
    """
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = [[0] * size_y for _ in range(size_x)]

    for x in range(size_x):
        matrix[x][0] = x
    for y in range(size_y):
        matrix[0][y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x - 1] == seq2[y - 1]:
                matrix[x][y] = matrix[x - 1][y - 1]
            else:
                matrix[x][y] = min(
                    matrix[x - 1][y] + 1,      # Deletion
                    matrix[x][y - 1] + 1,      # Insertion
                    matrix[x - 1][y - 1] + 1   # Substitution
                )
    return matrix[size_x - 1][size_y - 1]


def compute_weighted_cer(preds: List[str], refs: List[str]) -> Tuple[float, int, int]:
    """
    Computes Weighted Character Error Rate (CER).
    
    Returns:
        cer (float): total_char_errors / total_ref_chars
        total_errors (int)
        total_ref_chars (int)
    """
    total_errors = 0
    total_ref_chars = 0

    for pred, ref in zip(preds, refs):
        pred_str = str(pred) if pd.notnull(pred) else ""
        ref_str = str(ref) if pd.notnull(ref) else ""

        dist = levenshtein_distance(pred_str, ref_str)
        ref_len = len(ref_str)

        total_errors += dist
        total_ref_chars += ref_len

    if total_ref_chars == 0:
        return 0.0, 0, 0

    cer = total_errors / float(total_ref_chars)
    return cer, total_errors, total_ref_chars


def compute_weighted_wer(preds: List[str], refs: List[str]) -> Tuple[float, int, int]:
    """
    Computes Weighted Word Error Rate (WER).
    
    Returns:
        wer (float): total_word_errors / total_ref_words
        total_errors (int)
        total_ref_words (int)
    """
    total_errors = 0
    total_ref_words = 0

    for pred, ref in zip(preds, refs):
        pred_str = str(pred) if pd.notnull(pred) else ""
        ref_str = str(ref) if pd.notnull(ref) else ""

        pred_words = pred_str.strip().split()
        ref_words = ref_str.strip().split()

        dist = levenshtein_distance(pred_words, ref_words)
        ref_len = len(ref_words)

        total_errors += dist
        total_ref_words += ref_len

    if total_ref_words == 0:
        return 0.0, 0, 0

    wer = total_errors / float(total_ref_words)
    return wer, total_errors, total_ref_words


def compute_zindi_metric(
    preds: List[str], 
    refs: List[str], 
    cer_weight: float = 0.5, 
    wer_weight: float = 0.5
) -> Dict[str, Any]:
    """
    Computes the complete Zindi Weighted Evaluation Metric.
    
    Final Metric = cer_weight * Weighted_CER + wer_weight * Weighted_WER
    Official Challenge Default: 0.5 * Weighted_CER + 0.5 * Weighted_WER
    """
    cer, char_errs, char_total = compute_weighted_cer(preds, refs)
    wer, word_errs, word_total = compute_weighted_wer(preds, refs)

    final_score = cer_weight * cer + wer_weight * wer

    return {
        'final_score': round(final_score, 6),
        'weighted_cer': round(cer, 6),
        'weighted_wer': round(wer, 6),
        'char_errors': char_errs,
        'total_ref_chars': char_total,
        'word_errors': word_errs,
        'total_ref_words': word_total,
        'cer_weight': cer_weight,
        'wer_weight': wer_weight
    }


def evaluate_predictions_df(pred_df: pd.DataFrame, ref_df: pd.DataFrame, id_col: str = 'ID', target_col: str = 'Target') -> Dict[str, Any]:
    """
    Evaluates prediction dataframe against reference dataframe matching by ID.
    """
    merged = pd.merge(ref_df, pred_df, on=id_col, suffixes=('_ref', '_pred'))
    
    if len(merged) == 0:
        raise ValueError("No matching sample IDs between predictions and reference dataframes!")

    preds = merged[f"{target_col}_pred"].fillna("").tolist()
    refs = merged[f"{target_col}_ref"].fillna("").tolist()

    res = compute_zindi_metric(preds, refs)
    res['matched_samples'] = len(merged)
    res['total_reference_samples'] = len(ref_df)

    return res
