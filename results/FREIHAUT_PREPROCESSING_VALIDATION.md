# Freihaut & Göritz (2021) Dataset Preprocessing Validation

This report validates the preprocessing pipeline built for the Freihaut & Göritz (2021) dataset for keyboard-based stress classification.

## 1. Source Data Integrity
- **Raw Data Integrity**: Preserved as read-only.
- **Dataset Composition**: Extracted from both the Lab Study and the Online Study.
- **Total Valid Participants**: 1066 initial, 1064 yielded valid sequences.

## 2. Sequence Generation
- **Sub-window Configuration**: 4 keystrokes per sub-window.
- **Timesteps**: 10 consecutive sub-windows per sample (`NUM_TIMESTEPS=10`).
- **Input Tensor Shape**: `(N, 10, 7)`.
- **Features Extracted (7)**: `mean_dwell`, `std_dwell`, `mean_flight`, `std_flight`, `typing_speed`, `backspace_freq`, `error_rate`.

## 3. Class Distribution & Splitting Strategy
Data is partitioned using a strictly **Participant-Independent (70/15/15)** split to prevent data leakage.

| Split | Participants | Sequences (N) | Stress (Label=1) | Non-Stress (Label=0) |
|-------|--------------|---------------|------------------|----------------------|
| **Train** | 744 | 4664 | 1219 | 3445 |
| **Validation** | 159 | 979 | 228 | 751 |
| **Test** | 161 | 954 | 217 | 737 |
| **Total** | 1064 | 6597 | 1664 | 4933 |

## 4. Leakage Audit Results
The automated leakage audit (`results/freihaut_leakage_audit.json`) confirmed the following:
- `participant_leakage`: **PASS** (Zero overlap of participants between Train, Val, and Test sets).
- `session_leakage`: **PASS** (No cross-contamination of sessions).
- `duplicate_windows`: **PASS** (No duplicated sequence data).
- `identical_feature_vectors`: **PASS** (No identical feature inputs across splits).
- `label_distribution_valid`: **PASS** (Valid representations of both classes in all splits).

**OVERALL AUDIT STATUS: [PASS]**

## 5. Security & Gate Implementation
- The training gate in `train.py` has been updated and tested (`--validate-data`). It now strictly verifies the presence of the `data/processed/freihaut/*.npy` arrays, the preprocessing metadata, and the passing leakage audit before allowing any training to proceed.

## Conclusion
The Freihaut dataset has been successfully preprocessed into a robust, participant-independent machine-learning ready dataset. The input shapes match the `RA_HMSD` requirements (`10, 7`), and all data leakage vulnerabilities have been mitigated. The pipeline is scientifically sound and training can now safely proceed.
