# DATASET READINESS AUDIT

This document establishes the data sufficiency status of the Keystroke-Stress model and defines the strict criteria required before any future scientific retraining occurs. 

## 1. Current Dataset Completeness
**INCOMPLETE (Demo Subset)**
The local directory `data/keystroke-stress/` contains only a fragmentary demonstration subset of the official SWELL-KW dataset. 

## 2. Number of Participants
**UNKNOWN (Subset)**
The exact participant mapping is missing from the local files. The full SWELL-KW dataset contains 25 participants, but the local subset does not capture this.

## 3. Number of Raw Records
- **Keystroke Log Files:** 2 (`1430503249.txt`, `1430599923.txt`)
- **PAM Self-Report Records:** 48 

## 4. Number of Valid Stress Labels
- **Stress Samples (Class 1):** 1

## 5. Number of Valid Non-Stress Labels
- **Non-Stress Samples (Class 0):** 1

## 6. Number of Usable Windows
**Total Usable Windows:** 2

## 7. Why Only 2 Windows Currently Exist
The data generation in `preprocess_keystroke_stress.py` enforces a strict temporal alignment: it looks back exactly 30 minutes from a valid PAM self-report and extracts all keystrokes. It requires at least 50 keystrokes to form a valid window. Because there are only 2 raw keystroke log files available locally, only 2 PAM timestamps successfully matched with sufficient keystroke density to generate complete 10-timestep sequences.

## 8. Whether Complete Data is Locally Available
**NO.**
A recursive search across the project workspace and typical user directories (`Downloads`, `Documents`) confirms that no additional SWELL-KW files, archives, or logs exist locally.

## 9. Whether Legitimate Acquisition is Possible
**YES, but MANUAL ACQUISITION REQUIRED.**
SWELL-KW is an open-access academic dataset hosted on DANS and maintained by TNO/Radboud University. However, it is categorized as `OPEN_ACCESS_FOR_REGISTERED_USERS`. Access requires the user to formally request permission, accept a license agreement, and manually download the multi-gigabyte dataset.

## 10. Alternative Datasets if Acquisition is Impossible
If manual acquisition of SWELL-KW is not possible, the closest free, open-access alternative containing both keystroke timings and stress labels is the **"Stress Detection by Keystroke, App & Mouse Changes"** dataset hosted on Kaggle. (See `results/DATASET_OPTIONS.md` for full evaluation).

## 11. What Must Happen Before Retraining
The current checkpoint (`results/checkpoints/keystroke_stress.keras`) must remain untouched as an operational prototype. 

**NO RETRAINING MAY OCCUR UNTIL THE FOLLOWING CRITERIA ARE MET:**
1. **Minimum Participants:** Data from at least 15 distinct human subjects must be acquired.
2. **Minimum Labeled Windows:** At least 500 valid temporal sequence windows.
3. **Minimum Class Distribution:** At least 200 STRESS samples and 200 NON-STRESS samples.
4. **Data Split:** A strict **Participant-Independent** (Leave-One-Subject-Out or N-Fold Subject) train/test split must be implemented.
5. **No Participant Overlap:** Zero subject IDs shared between the train and test sets.
6. **Leakage Check:** Sequences must not span across session boundaries or leak future states into the training history.
7. **Label Mapping:** Ground truth must be deterministically linked to validated physiological/psychological scales (e.g., PAM, NASA-TLX, RSME) provided by the dataset authors.
