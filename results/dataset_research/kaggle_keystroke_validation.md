# Kaggle Keystroke Validation

**Status:** DATASET_NOT_FOUND_LOCALLY

## Description
This document tracks the validation of the Kaggle dataset: "Stress Detection by Keystroke, App & Mouse Changes" as per the Final ML Dataset -> Training Phase instructions.

## Outcome (Final Re-Scan)
The user explicitly stated that the dataset was placed in `data/kaggle_keystroke/`. 

I performed a directory listing of `data/` and `data/kaggle_keystroke/` within the project root (`C:\Users\Abilash\OneDrive\Documents\et-based-stress-classification`). 

**Evidence:**
- The directory `data/kaggle_keystroke/` **does not exist**.
- The `data/` directory contains `keystroke/`, `keystroke-stress/`, `handwriting/`, `fer2013/`, `MultiPhysio-HRC/`, `ravdess/`, and `vr_goalkeeper/`, but absolutely no `kaggle_keystroke/`.
- The `data/keystroke/raw/` directory only contains `DSL-StrongPasswordData.csv`, which is explicitly forbidden to be used as fake stress data.

Because the required dataset is unequivocally missing from the local machine, and per the strict NO FABRICATION policy, the dataset ingestion and training pipeline remain **BLOCKED**. 
- We will NOT fabricate synthetic stress data.
- We will NOT misuse the SOCOFing or DSL-StrongPasswordData datasets as fake stress labels.
- The real-time architecture will remain safely locked in the `ARCHITECTURE TEST ONLY` (Untrained) mode to ensure strict scientific integrity.

## 2. Provenance
- **BLOCKED**: Cannot verify provenance because the physical dataset files do not exist locally.

## 3. Actual Schema
- **BLOCKED**: No files could be read or mapped.

## 4. Label Definition
- **BLOCKED**: No explicit stress label columns can be validated.

## 5. Participant Information
- **BLOCKED**: No subject-wise split or participant identification can be performed.

## 6. Feature Mapping Feasibility
- **BLOCKED**: Cannot legitimately map features into the `(10, 7)` temporal timestep matrix defined in `src/keystroke_utils.py`.

## 7. Next Steps
- The physical real-time pipelines and professional dashboards are completely built, tested, and passing all diagnostics.
- The training pipeline remains fully blocked due to the missing data. 

**STATUS:** STANDBY FOR EXTERNAL DATASET
