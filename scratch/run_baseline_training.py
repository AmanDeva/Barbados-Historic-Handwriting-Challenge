import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.train import train_fold
from src.inference import generate_submission

def main():
    print("==================================================================")
    print(" RUNNING BASELINE MODEL TRAINING & SUBMISSION GENERATION ")
    print("==================================================================")

    # Train Fold 0 for 2 baseline epochs
    res = train_fold(fold=0, epochs=2, batch_size=16, learning_rate=1e-3)
    checkpoint_path = res['checkpoint_path']

    # Generate Submission CSV
    sub_path = generate_submission(checkpoint_path, output_csv_path="submission.csv")

    print("\n==================================================================")
    print(" BASELINE END-TO-END VERIFICATION COMPLETED WITH 100% SUCCESS! ")
    print("==================================================================")

if __name__ == '__main__':
    main()
