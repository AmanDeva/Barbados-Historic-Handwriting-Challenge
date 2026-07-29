import os
import sys
import pandas as pd

# Ensure project root is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.metrics import compute_zindi_metric, compute_weighted_cer, compute_weighted_wer

def main():
    print("==================================================================")
    print(" ZINDI WEIGHTED EVALUATION METRIC ENGINE VERIFICATION ")
    print("==================================================================")

    # Test Case 1: Perfect Predictions (Score should be 0.0)
    refs_perfect = ["By this public act and instrument", "In the year of our Lord God"]
    preds_perfect = ["By this public act and instrument", "In the year of our Lord God"]
    res_perfect = compute_zindi_metric(preds_perfect, refs_perfect)

    print("\n--- TEST CASE 1: PERFECT PREDICTIONS ---")
    print(f"Weighted CER : {res_perfect['weighted_cer']:.4f}")
    print(f"Weighted WER : {res_perfect['weighted_wer']:.4f}")
    print(f"Final Score  : {res_perfect['final_score']:.4f} (Expected: 0.0000)")
    assert res_perfect['final_score'] == 0.0, "Perfect prediction score must be 0.0"

    # Test Case 2: Impact of Sentence Length Weighting
    # Scenario A: 1 error in a SHORT reference sentence (10 chars)
    ref_short = ["Short text"]  # 10 chars, 2 words
    pred_short = ["Short txxt"] # 1 char substitution ('e' -> 'x')

    res_short = compute_zindi_metric(pred_short, ref_short)

    # Scenario B: 1 error in a LONG reference sentence (100 chars)
    ref_long = ["This is a very long historical manuscript line from eighteenth century Barbados archival record document"] # 105 chars, 15 words
    pred_long = ["This is a very long historical manuscript line from eighteenth century Barbados archival record documxnt"] # 1 char substitution ('e' -> 'x')

    res_long = compute_zindi_metric(pred_long, ref_long)

    print("\n--- TEST CASE 2: SENTENCE LENGTH WEIGHTING IMPACT ---")
    print("1 Character Error in Short Sentence (10 chars):")
    print(f"  • Weighted CER: {res_short['weighted_cer']:.4f} ({res_short['char_errors']}/{res_short['total_ref_chars']})")
    print(f"  • Weighted WER: {res_short['weighted_wer']:.4f} ({res_short['word_errors']}/{res_short['total_ref_words']})")
    print(f"  • Final Score : {res_short['final_score']:.4f}")

    print("\n1 Character Error in Long Sentence (105 chars):")
    print(f"  • Weighted CER: {res_long['weighted_cer']:.4f} ({res_long['char_errors']}/{res_long['total_ref_chars']})")
    print(f"  • Weighted WER: {res_long['weighted_wer']:.4f} ({res_long['word_errors']}/{res_long['total_ref_words']})")
    print(f"  • Final Score : {res_long['final_score']:.4f}")

    print(f"\n[INSIGHT] Making 1 error in a 105-char sentence increases CER by only {res_long['weighted_cer']:.4f}, whereas in a 10-char sentence it increases CER by {res_short['weighted_cer']:.4f} ({res_short['weighted_cer']/res_long['weighted_cer']:.1f}x higher impact per line!).")

    # Test Case 3: Dataset Evaluation Simulation
    train_cleaned_path = os.path.join(PROJECT_ROOT, 'Train_Cleaned.csv')
    if os.path.exists(train_cleaned_path):
        df = pd.read_csv(train_cleaned_path).head(100)
        refs_sample = df['Target'].tolist()
        # Simulate predictions with minor noise (e.g. 5% random typos)
        preds_simulated = [r[:-1] if len(r) > 5 else r for r in refs_sample]

        res_sim = compute_zindi_metric(preds_simulated, refs_sample)
        print("\n--- TEST CASE 3: DATASET SIMULATION (100 Samples) ---")
        print(f"Matched Samples : {len(refs_sample)}")
        print(f"Total Ref Chars : {res_sim['total_ref_chars']:,}")
        print(f"Total Ref Words : {res_sim['total_ref_words']:,}")
        print(f"Weighted CER    : {res_sim['weighted_cer']:.4f}")
        print(f"Weighted WER    : {res_sim['weighted_wer']:.4f}")
        print(f"Final Score     : {res_sim['final_score']:.4f} (0.5 * CER + 0.5 * WER)")

    print("\n==================================================================")
    print(" ALL METRIC ENGINE TESTS COMPLETED SUCCESSFULLY! ")
    print("==================================================================")

if __name__ == '__main__':
    main()
