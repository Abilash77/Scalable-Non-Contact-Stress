# Stress Dataset Candidates

| Dataset | Stress labels | Participants | Modality | Subject IDs | Public access | Real-time compatible | Training speed | Decision |
|---|---|---:|---|---|---|---|---|---|
| **MultiPhysio-HRC** | Genuine (STAI, TLX) | 21 | Physio, Speech, Face AU | Yes | EULA/Zenodo (410 error) | Yes (Face AU, Speech) | Moderate | Rejected (Access issues) |
| **Stress-Predict-Dataset** | Genuine (Protocol) | 35 | Physio (ECG, BVP, EDA) | Yes | Yes (GitHub) | No | Fast | Rejected (Physio only) |
| **Kaggle: Stress Detection by Keystroke, App & Mouse** | Genuine | 2 | Keystroke, Mouse | Yes | Yes (Kaggle) | Yes (Keystroke) | Fast | **SELECTED** |
| **WESAD** | Genuine | 15 | Physio, Motion | Yes | Yes (UCI) | No | Fast | Rejected (Physio only) |

## 1. MultiPhysio-HRC
- **Why it qualifies**: Contains Face Action Units and Speech recordings in a Human-Robot Collaboration scenario. Genuine stress labels (STAI, NASA-TLX).
- **Official Source**: Zenodo (DOI: 10.5281/zenodo.18668043) / GitHub (automation-robotics-machines/MultiPhysio-HRC)
- **Access requirements**: Requires Zenodo access. The provided record URL is currently returning a 410 (Gone).
- **Actual modalities**: EEG, ECG, EDA, RESP, EMG, Voice, Facial AUs.
- **Actual labels**: STAI-Y1 (stress), NASA-TLX (workload).
- **Limitations**: Access is currently blocked or requires DUA.

## 2. Stress-Predict-Dataset
- **Why it qualifies**: Highly legitimate, published in *Sensors*, easily accessible on GitHub without EULA. 
- **Official Source**: https://github.com/italha-d/Stress-Predict-Dataset
- **Access requirements**: Publicly available.
- **Actual modalities**: Physiological (ACC, BVP, EDA, HR, TEMP).
- **Actual labels**: 0 (baseline/non-stress), 1 (stress tasks like Stroop/Interview).
- **Limitations**: Contains NO behavioral or non-contact modalities (no keyboard, face, or speech). Real-time pipeline integration would be blocked.

## 3. Stress Detection by Keystroke, App & Mouse Changes
- **Why it qualifies**: Directly targets Priority 2 (Keyboard/typing). Provides keystroke dynamics features (dwell, flight) that naturally match our `extract_keystroke_features` pipeline.
- **Official Source**: Kaggle (chamindu/stress-detection-by-keystroke-app-mouse-changes)
- **Access requirements**: Kaggle Account (Free, no restrictive academic EULA).
- **Actual modalities**: Keystroke dynamics, Mouse movements, Application usage.
- **Actual labels**: Stress, Fatigue, Energy (Self-reported / PAM).
- **Limitations**: Very small participant pool (2 users). However, it perfectly serves the architectural requirement of a fast-training, non-contact, real-time compatible dataset.

## 4. WESAD
- **Why it qualifies**: Gold standard for stress detection.
- **Official Source**: UCI Machine Learning Repository
- **Access requirements**: Publicly available.
- **Actual modalities**: Physiological (Chest and Wrist sensors).
- **Actual labels**: Baseline, Stress, Amusement, Meditation.
- **Limitations**: Physiological only. Cannot be linked to the real-time webcam/microphone/keyboard architecture.
