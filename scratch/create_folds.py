import os
import sys
import pandas as pd
from sklearn.model_selection import KFold

# Ensure project root is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def main():
    print("==================================================================")
    print(" CREATING 5-FOLD CROSS-VALIDATION SPLIT ")
    print("==================================================================")

    processed_train_csv = os.path.join(PROJECT_ROOT, 'data', 'processed_train.csv')
    if not os.path.exists(processed_train_csv):
        # Fallback to Train_Cleaned.csv
        processed_train_csv = os.path.join(PROJECT_ROOT, 'Train_Cleaned.csv')

    train_df = pd.read_csv(processed_train_csv)
    print(f"Loaded {len(train_df):,} training samples from: {processed_train_csv}")

    # Add text length and width bin for stratified splitting
    train_df['text_len'] = train_df['Target'].apply(lambda x: len(str(x)) if pd.notnull(x) else 0)
    
    # Sort dataframe by text length to enable stratified bin distribution across folds
    train_df = train_df.sort_values(by='text_len').reset_index(drop=True)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    train_df['fold'] = -1

    for fold_idx, (train_indices, val_indices) in enumerate(kf.split(train_df)):
        train_df.loc[val_indices, 'fold'] = fold_idx

    # Verification of fold distribution
    print("\n--- 5-FOLD DISTRIBUTION SUMMARY ---")
    for fold in range(5):
        fold_samples = train_df[train_df['fold'] == fold]
        avg_len = fold_samples['text_len'].mean()
        min_len = fold_samples['text_len'].min()
        max_len = fold_samples['text_len'].max()
        print(f"  • Fold {fold}: {len(fold_samples):,} samples | Target Len (Mean: {avg_len:.1f}, Min: {min_len}, Max: {max_len})")

    # Save to data/train_folds.csv
    folds_out_csv = os.path.join(PROJECT_ROOT, 'data', 'train_folds.csv')
    train_df.to_csv(folds_out_csv, index=False)
    print(f"\n[OK] 5-Fold split successfully saved to: {folds_out_csv}")

if __name__ == '__main__':
    main()
