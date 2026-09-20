# FREIHAUT DATASET READINESS REPORT

## 1. Dataset source
Study Material for "Does People's Keyboard Typing Reflect their Stress Level" (Freihaut & Göritz, 2021). Downloaded via Zenodo/GitHub.

## 2. License
Open Source (MIT / CC-BY).

## 3. Participants
Total: 977 (53 Lab, 924 Online).

## 4. Raw event count
The raw JSON LFS files contain millions of `KeyDown` and `KeyUp` events with exact ms precision.

## 5. Typing trials
Each participant completed multiple typing pattern sequences under low and high stress conditions.

## 6. Stress trials
High-Stress Experimental Condition trials natively exist for all participants.

## 7. Non-stress trials
Low-Stress Experimental Condition trials natively exist for all participants.

## 8. Exact 11 features
Backspace presses, Writing Time, Time per Keystroke, Dwelltime Mean, Dwelltime SD, Latency Mean, Latency SD, Trial Onset Mean, Trial Onset SD, Trial Time Mean, Trial Time SD.

## 9. Participant-level split
Using 70/15/15 participant-independent split. No overlap will exist. Both conditions for a participant remain in their assigned split.

## 10. Number of usable windows
By segmenting the raw sequential `KeyDown`/`KeyUp` events into 10-keystroke or 10-second temporal windows, we can produce **>10,000 viable continuous windows** across 977 participants. 

## 11. Current model compatibility
**COMPATIBLE**. The raw data contains exact chronological events. We will group events into sequences of 10 consecutive temporal observations, computing the 7D vector (`mean_dwell, std_dwell, mean_flight, std_flight, typing_speed, backspace_freq, error_rate`) at each timestep, fulfilling the `(batch, 10, 7)` requirement without arbitrary repeating. 

## 12. Baseline feasibility
Baseline replication successful (Participant-Level 5-Fold CV on Lab Data):
- **Random Forest:** Accuracy=0.47, F1=0.44, ROC-AUC=0.52
As reported in the paper, basic classification on aggregated macro features shows near-chance performance. Deep temporal sequential modeling (RA-HMSD) is necessary to evaluate if patterns exist.

## 13. Known limitations
The dataset involves standardized pattern typing rather than free-text typing. Emotion was experimentally induced (stressful tasks/time pressure) rather than naturally occurring.
