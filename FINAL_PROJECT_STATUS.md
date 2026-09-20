# FINAL PROJECT STATUS

## A. Project Purpose
This repository implements the infrastructure for "Scalable Non-Contact Stress Detection Using Hybrid Multimodal Intelligence". It provides a dual-architecture system:
1. A live engineering demonstrator capable of processing real-time multimodal inputs (Keyboard, Speech, Facial, Eye/Pupil, Handwriting) with a T=10 Reliability-Aware Attention Fusion model.
2. A scientific research pipeline mathematically constrained to reproduce the original ForDigitStress paper's Unimodal Pupil Diameter (PD) experiments, without fabricating unrecorded modalities.

## B. Live Runtime Architecture
- **Command:** `python run.py`
- **Modalities:** 5 (Keyboard, Speech, Facial, Eye/Pupil, Handwriting)
- **Status:** **FULLY OPERATIONAL (BUT UNTRAINED)**
- **Role:** Demonstrates real-time sensor ingestion, sequential modeling (T=10), and attention fusion logic. 

## C. Research/Training Architecture
- **Command:** `python train_pd.py`
- **Modalities:** 1 (Eye/Pupil PD)
- **Status:** **IMPLEMENTED, BUT BLOCKED**
- **Role:** Handles the legitimate ForDigitStress dataset parsing, splitting, and training for the baseline PD-only claims.

## D. Dataset Status
The legitimate ForDigitStress dataset (~360 GB) is currently **absent**.
Because the scientific pipeline enforces strict adherence to real data, training is halted to prevent pollution by synthetic generation or unrelated dataset concatenation.

## E. Current Checkpoint Status
- **Runtime:** `results/checkpoints/stress_model.keras` (Expected but currently absent; system safely falls back to UNTRAINED inference mode).
- **Research:** `results/checkpoints/research_pd_model.keras` (Configured to save here, but no training has occurred yet).
- **Isolation:** The `run.py` runtime has been strictly cordoned off from accidentally loading the `research_pd_model.keras` checkpoint.

## F. Test Results
- **Suite:** 12/12 End-to-End Regression Tests **PASS**
- The live dashboard starts, the API responds correctly, and all feature extractors gracefully handle missing/noisy inputs.

## G. Scientific Validity Status
**NOT YET ESTABLISHED.**
The system architecture is validated from a software engineering perspective, but the actual classification metrics (Accuracy, AUROC, F1) remain pending the acquisition of the ForDigitStress dataset. The system truthfully reports this via `MODEL STATUS: UNTRAINED`.

## H. What is Operational Today
- Multi-threaded hardware polling (Webcam, Microphone, Keyboard, Canvas)
- Feature extraction pipelines (FACS, GeMAPS, keystroke dynamics, stroke vectors)
- Neural network forward passes (Reliability-Aware Attention Fusion)
- HTTP Dashboard and API

## I. What Requires ForDigitStress Acquisition
- Actual subject-level training and cross-validation
- Model weight optimization
- Generation of the `stress_model.keras` and `research_pd_model.keras` checkpoints
- Validating the test set metrics against the published paper

## J. Exact Commands

**NORMAL USE:**
`python run.py`
Training is an explicit offline operation. Normal application startup never trains or processes the training dataset.

**TRAINING:**
`python train.py`
OR
`python train_pd.py`
depending on the research experiment.

**General Training Validation:**
`python train.py --validate-data`

**Tests:**
`python run_tests.py`

## K. Explicit Limitations
- The 5-modality fusion model currently lacks a corresponding 5-modality dataset to train on, as ForDigitStress is strictly unimodal in its deep-learning experimental context.
- Attempting to bypass the dataset block by concatenating unrelated datasets (e.g., Keystrokes + RAVDESS + ForDigitStress) is explicitly forbidden as it destroys the temporal integrity of the attention mechanism.
