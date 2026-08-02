import os
import sys
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

CORRUPTED_IDS = [
    '79tMUVyfIdy3GzkG', 'rh8o7bdCGOIBFPwH', 'F8DYDDp2AvW9Dytw', 'mfQxfOmeRmwBh0g8',
    'EcxuqKeZl7OQexfB', 'PO7QQLWIFOT65BTz', 'yNyf3Tp0zc7DFj5F', 'JU7lRwk3jKkus24Z',
    'R6iYPb7MHFiHtXH6', 'VmrEALeZiP1Y6nF9', 't0UrASljcgzvBAnO', 'KH5g92Q3DA5Bo6xi',
    'N78M3v6GKmUzF7sk', '0CCrVKAom8EK53jj', '3ZxOeKcOr5wUYyk0', 'PJbM7Q1SrblrSWt6',
    'WwlTCykxjP3c4kfo', 'baTY3OlGskirWgFc', 'u3b4JNo5bqpfE7Js', 'MfT9S5oghk9ywNSC',
    '8H2ITJSWZhAD6eh0'
]

def main():
    print("==================================================================")
    print(" EXCLUDING 21 OFFICIAL ZINDI CORRUPTED TRAINING ROWS ")
    print("==================================================================")

    targets = [
        os.path.join(PROJECT_ROOT, 'data', 'train_folds.csv'),
        os.path.join(PROJECT_ROOT, 'Train_Folds.csv'),
        os.path.join(PROJECT_ROOT, 'Train_Cleaned.csv'),
        os.path.join(PROJECT_ROOT, 'data', 'processed_train.csv')
    ]

    for path in targets:
        if os.path.exists(path):
            df = pd.read_csv(path)
            orig_len = len(df)
            df_cleaned = df[~df['ID'].isin(CORRUPTED_IDS)].reset_index(drop=True)
            new_len = len(df_cleaned)
            removed = orig_len - new_len
            df_cleaned.to_csv(path, index=False)
            print(f"[OK] Cleaned {os.path.basename(path)}: {orig_len:,} -> {new_len:,} rows (Removed {removed} corrupted samples)")

    print("\n==================================================================")
    print(" DATASET CLEANING COMPLETED WITH 100% SUCCESS! ")
    print("==================================================================")

if __name__ == '__main__':
    main()
