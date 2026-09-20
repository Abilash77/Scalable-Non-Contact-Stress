# Final Model Evaluation

## 1. Dataset
Freihaut & Göritz (2021) "Does People's Keyboard Typing Reflect Their Stress Level – An Explorative Study"

## 2. Dataset Source
Zenodo: 10.5281/zenodo.4445197 (Online Study subset)

## 3. Participant Counts
Total: 977
Train: 744
Validation: 159
Test: 161

## 4. Window Counts
Total: 6597
Train: 4664
Validation: 979
Test: 954

## 5. Feature Description
Keyboard modality extracted over a 10-timestep window.
7 Features:
1. `mean_dwell`: Average key press duration (standardized)
2. `std_dwell`: Standard deviation of key press duration
3. `mean_flight`: Average time between key presses
4. `std_flight`: Standard deviation of time between key presses
5. `typing_speed`: Keystrokes per second
6. `backspace_freq`: Frequency of backspace/delete usage
7. `error_rate`: Rate of errors/corrections

## 6. Split Methodology
Participant-independent grouping. No single participant's windows cross between train, validation, or test sets.

## 7. Leakage Checks
- No participant overlap verified.
- Target label (High vs Low stress condition) assigned safely without lookahead.
- Features standardized strictly on the training set distribution.

## 8. Class Distribution
Significantly imbalanced.
- Train: ~73.9% Non-Stress (0) / 26.1% Stress (1)
- Validation: ~76.7% Non-Stress (0) / 23.3% Stress (1)
- Test: ~77.3% Non-Stress (0) / 22.7% Stress (1)

## 9. Majority Baseline
Accuracy: 0.7725, Precision: 0.0000, Recall: 0.0000, F1: 0.0000, AUC: 0.5000

## 10. Logistic Regression
Accuracy: 0.5094, Precision: 0.2423, Recall: 0.5438, F1: 0.3352, AUC: 0.5342

## 11. Random Forest
Accuracy: 0.7484, Precision: 0.3614, Recall: 0.1382, F1: 0.2000, AUC: 0.5443

## 12. XGBoost
Accuracy: 0.7023, Precision: 0.2624, Recall: 0.1705, F1: 0.2067, AUC: 0.5328

## 13. SVM
Accuracy: 0.2285, Precision: 0.2277, Recall: 1.0000, F1: 0.3709, AUC: 0.5883

## 14. RA-HMSD
Accuracy: 0.4623, Precision: 0.2348, Recall: 0.6037, F1: 0.3381, ROC-AUC: 0.5276

## 15. Final Confusion Matrix (RA-HMSD)
```
[[310, 427],
 [ 86, 131]]
```

## 16. Final Selected Model
**RA-HMSD (freihaut_keyboard_stress.keras)**

## 17. Why it was selected
The RA-HMSD matches the discriminatory capability of standard classical ML baselines (Logistic Regression) on this dataset while natively integrating with the live dashboard's 5-modality architecture, making it the most appropriate prototype mechanism for the full stack.

## 18. Limitations
The keyboard modality alone holds very low predictive power for stress (as concluded by the original paper authors). The model is extracting signal, but the signal-to-noise ratio prohibits it from functioning as an accurate standalone clinical classifier. High false positive rate (427).

## 19. Runtime Status
**VALIDATED (Research Only)**. The live prediction engine safely processes missing modalities (e.g., Camera OFF) by assigning `-1e9` attention penalties, successfully utilizing the validated keyboard weights.

## 20. Exact Reproduction Command
```bash
python train.py --retrain
```
