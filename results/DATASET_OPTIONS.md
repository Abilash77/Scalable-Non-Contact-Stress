# DATASET OPTIONS: KEYSTROKE & STRESS

This document outlines potential alternative datasets to support subject-independent keystroke stress modeling if complete access to SWELL-KW cannot be acquired.

## Option 1: "Stress Detection by Keystroke, App & Mouse Changes" (Kaggle)
- **Official Source:** Kaggle / Independent Researchers
- **License/Access:** Open Access (typically CC0 or similar public license on Kaggle)
- **Modality:** Keystroke dynamics (timings), mouse movements, application usage.
- **Labels:** Includes explicit stress/fatigue labels.
- **Academic Use:** Permitted (commonly cited in Kaggle-based machine learning studies).
- **Temporal Structure:** Contains timestamps/session structures.
- **Subject-Independent Evaluation:** Possible, provided participant IDs are tracked in the dataset to allow for proper leave-one-subject-out cross-validation.

## Option 2: CMU Keystroke Dynamics Benchmark Dataset
- **Official Source:** Carnegie Mellon University (CMU)
- **License/Access:** Publicly available for academic research.
- **Modality:** High-fidelity keystroke dynamics (dwell times, flight times).
- **Labels:** *Not explicitly stress.* This is an authentication dataset.
- **Why it's listed:** Often used as a baseline pre-training dataset to learn generalized keystroke representations before fine-tuning on a smaller, stress-labeled dataset.
- **Subject-Independent Evaluation:** Yes (has hundreds of unique subjects).

## Option 3: KeyRecs Dataset (Zenodo)
- **Official Source:** Zenodo / Academic Researchers
- **License/Access:** Open Access.
- **Modality:** Free-text and fixed-text keystroke dynamics.
- **Labels:** Identity/demographics (Age, Gender). *Not explicitly stress.*
- **Why it's listed:** Can be used to build robust temporal feature extractors for free-text typing in the wild.

## Summary Recommendation
The **Kaggle Stress Detection Dataset** is the only immediately accessible open dataset containing both keystroke data and explicit stress labels. However, its exact label validation (e.g., whether stress was self-reported or induced) must be scrutinized before acceptance. If rigorous academic ground-truth is required, acquiring the complete **SWELL-KW** dataset via DANS/TNO remains the scientifically superior path.
