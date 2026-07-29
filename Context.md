# Coding Agent Guidelines

## Project Overview

This project is for the **R.O.A.D. Barbados Historic Handwriting Challenge**. The objective is to build an OCR model that transcribes historical handwritten text from cropped line images.

The handwriting comes from 18th–19th century Barbados archival records and includes challenges such as:

- Historical handwriting styles
- Variable spacing
- Faded ink
- Damaged documents
- Other scanning imperfections

The model should be robust to these real-world conditions.

---

## Project Structure

```text
OCR-chall/
│
├── images/                  # All training and test images
├── Starters/                # Baseline OCR starter code
├── Context.md               # Project documentation
├── Train.csv                # Training labels
├── Test.csv                 # Test image IDs
└── SampleSubmission.csv     # Submission format
```

---

## Dataset Files

### Train.csv

Contains the training data with:

- Image ID
- Ground-truth transcription

Use this file for model training and validation.

### Test.csv

Contains the test image IDs.

Predictions generated for these images must be written to the submission file.

### images/

Contains all image files referenced by both `Train.csv` and `Test.csv`.

### SampleSubmission.csv

Use this as the template for the final submission.

Requirements:

- Preserve all IDs exactly.
- Output predictions in the same format.
- Row order is not important.

### Starters/

Contains the baseline OCR implementation and reference code.

---

## Development Guidelines

- Always use paths relative to the project root.
- Do not modify the original dataset files.
- Keep training, inference, and preprocessing code modular.
- Save trained models and checkpoints in separate output directories.
- Follow the submission format exactly as shown in `SampleSubmission.csv`.
