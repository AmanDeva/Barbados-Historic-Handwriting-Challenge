"""
Stratified K-Fold Split Generator based on Target Text Length for R.O.A.D. Barbados Challenge.

Phases:
- Phase 1: Feature Engineering (Target Length calculation)
- Phase 2: Establish Stratification Bins (Dynamic 5-Quantile Binning: P20, P40, P60, P80)
- Phase 3: Execute StratifiedKFold (5 splits, random_state=42)
- Phase 4: Validation & Export to Train_Folds.csv & data/train_folds.csv
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold

# Ensure project root is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def generate_stratified_folds(train_csv_path: str, n_splits: int = 5, seed: int = 42) -> pd.DataFrame:
    """
    Loads dataset, engineers text length features, creates 5 quantile stratification bins,
    applies StratifiedKFold, and assigns fold numbers 0..4.
    """
    if not os.path.exists(train_csv_path):
        raise FileNotFoundError(f"Training file not found: {train_csv_path}")

    df = pd.read_csv(train_csv_path)

    # Phase 1: Feature Engineering (Target Length)
    df['text_length'] = df['Target'].apply(lambda x: len(str(x)) if pd.notnull(x) else 0)

    # Phase 2: Establish Stratification Bins (5 Quantile Bins: P20, P40, P60, P80)
    df['length_bin'] = pd.qcut(df['text_length'], q=n_splits, labels=False, duplicates='drop')

    # Phase 3: Execute Stratified Split
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    df['fold'] = -1

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(df, df['length_bin'])):
        df.loc[val_idx, 'fold'] = fold_idx

    return df


def main():
    print("==================================================================")
    print(" STRATIFIED K-FOLD SPLIT GENERATION (TARGET TEXT LENGTH) ")
    print("==================================================================")

    train_cleaned_path = os.path.join(PROJECT_ROOT, 'Train_Cleaned.csv')
    if not os.path.exists(train_cleaned_path):
        train_cleaned_path = os.path.join(PROJECT_ROOT, 'data', 'processed_train.csv')

    print(f"Loading dataset from: {train_cleaned_path}")
    df_folds = generate_stratified_folds(train_cleaned_path, n_splits=5, seed=42)

    print(f"\n[OK] Successfully assigned 5 folds for {len(df_folds):,} training rows.")

    # Phase 4: Validation & Distribution Check
    print("\n--- PHASE 4: FOLD STATISTICAL DISTRIBUTION CHECK ---")
    
    fold_stats = []
    for fold in range(5):
        fold_subset = df_folds[df_folds['fold'] == fold]
        mean_len = fold_subset['text_length'].mean()
        std_len = fold_subset['text_length'].std()
        min_len = fold_subset['text_length'].min()
        max_len = fold_subset['text_length'].max()
        p50_len = fold_subset['text_length'].median()

        fold_stats.append({
            'fold': fold,
            'count': len(fold_subset),
            'mean_length': round(mean_len, 2),
            'std_length': round(std_len, 2),
            'median_length': p50_len,
            'min_length': min_len,
            'max_length': max_len
        })

    df_stats = pd.DataFrame(fold_stats)
    print(df_stats.to_string(index=False))

    # Verify bin breakdown per fold
    print("\n--- PER-FOLD BIN BREAKDOWN ---")
    bin_table = pd.crosstab(df_folds['fold'], df_folds['length_bin'], margins=True)
    print(bin_table)

    # Plot Distribution
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=150)

    # Plot 1: Text Length KDE per Fold
    colors = ['#00E5FF', '#76FF03', '#FFD700', '#FF4081', '#E040FB']
    for fold in range(5):
        fold_data = df_folds[df_folds['fold'] == fold]['text_length']
        sns.kdeplot(fold_data, ax=ax1, label=f'Fold {fold} (Mean: {fold_data.mean():.1f})', color=colors[fold], linewidth=2)
    
    ax1.set_title('Text Length Probability Density across Folds', fontsize=13, fontweight='bold', pad=12, color='#00E5FF')
    ax1.set_xlabel('Target Text Length (Characters)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Density', fontsize=11, fontweight='bold')
    ax1.legend(loc='upper right', framealpha=0.8)

    # Plot 2: Bin Counts Per Fold (Bar Chart)
    bin_df_long = df_folds.groupby(['fold', 'length_bin']).size().reset_index(name='count')
    sns.barplot(data=bin_df_long, x='length_bin', y='count', hue='fold', palette=colors, ax=ax2)
    ax2.set_title('Stratified Quantile Bin Balance per Fold', fontsize=13, fontweight='bold', pad=12, color='#00E5FF')
    ax2.set_xlabel('Quantile Length Bin (0 = Shortest, 4 = Longest)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Sample Count', fontsize=11, fontweight='bold')
    ax2.legend(title='Fold', framealpha=0.8)

    plt.suptitle('R.O.A.D. Barbados Historic Handwriting - Stratified 5-Fold Split Verification', fontsize=15, fontweight='bold', y=0.98, color='white')

    out_dir = r'C:\Users\hp\.gemini\antigravity\brain\554a4812-972e-4d67-ad26-0f285eac3fd4'
    os.makedirs(out_dir, exist_ok=True)
    visual_out_path = os.path.join(out_dir, 'stratified_folds_verification.png')
    plt.savefig(visual_out_path, bbox_inches='tight')
    print(f"\n[OK] Fold distribution visualization saved to: {visual_out_path}")

    # Export Train_Folds.csv to root directory and data/
    train_folds_root = os.path.join(PROJECT_ROOT, 'Train_Folds.csv')
    train_folds_data = os.path.join(PROJECT_ROOT, 'data', 'train_folds.csv')

    export_df = df_folds[['ID', 'Target', 'text_length', 'length_bin', 'fold']]
    export_df.to_csv(train_folds_root, index=False)
    export_df.to_csv(train_folds_data, index=False)

    print(f"[OK] Exported Train_Folds.csv to root: {train_folds_root} ({len(export_df):,} rows)")
    print(f"[OK] Exported train_folds.csv to data: {train_folds_data} ({len(export_df):,} rows)")

if __name__ == '__main__':
    main()
