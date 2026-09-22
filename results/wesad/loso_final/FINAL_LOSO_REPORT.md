# FINAL WESAD 15-SUBJECT LOSO REPORT

## 1. Dataset & Scope
WESAD contains strictly physiological, wearable ECG/HRV data. This validates wearable cardiac stress prediction. **It does NOT validate the main non-contact project's modalities (Facial, Eye, Keyboard, Speech).**

## 2. Definitions
- Target: STRESS (1) vs NON-STRESS (0: baseline/amusement/meditation)
- Features: 62 physiological ECG/HRV metrics
- Excluded: Label, Condition, Subject ID, Time, SSSQ

## 3. Methodology
- **Validation**: 15-Fold Leave-One-Subject-Out
- **Training**: 12 subjects | **Validation**: 2 subjects | **Test**: 1 held-out subject
- Preprocessing (Scaler) strictly fit on Training.
- Model and Threshold strictly optimized on Training/Validation.

## 4. Aggregate Macro Results (Averaged across 15 folds)
|                                          |   Accuracy |   Balanced Accuracy |       F1 |   PR-AUC |   Precision |   ROC-AUC |   Recall |   Specificity |
|:-----------------------------------------|-----------:|--------------------:|---------:|---------:|------------:|----------:|---------:|--------------:|
| ('Deep Learning', 'Threshold_0.5')       |   0.846553 |            0.768699 | 0.60271  | 0.806971 |    0.666118 |  0.904042 | 0.631222 |      0.906177 |
| ('Deep Learning', 'Threshold_Val')       |   0.781128 |            0.778344 | 0.621137 | 0.806971 |    0.626209 |  0.904042 | 0.779094 |      0.777593 |
| ('Logistic Regression', 'Threshold_0.5') |   0.837222 |            0.750671 | 0.581608 | 0.844321 |    0.769381 |  0.919399 | 0.596355 |      0.904988 |
| ('Logistic Regression', 'Threshold_Val') |   0.557789 |            0.644139 | 0.460538 | 0.844321 |    0.486307 |  0.919399 | 0.799447 |      0.48883  |
| ('Random Forest', 'Threshold_0.5')       |   0.83972  |            0.723416 | 0.516721 | 0.810011 |    0.769429 |  0.915102 | 0.517017 |      0.929816 |
| ('Random Forest', 'Threshold_Val')       |   0.679545 |            0.776857 | 0.64651  | 0.810011 |    0.535912 |  0.915102 | 0.952309 |      0.601406 |
| ('XGBoost', 'Threshold_0.5')             |   0.829612 |            0.7322   | 0.545893 | 0.864955 |    0.704272 |  0.922855 | 0.558026 |      0.906374 |
| ('XGBoost', 'Threshold_Val')             |   0.644725 |            0.734421 | 0.5845   | 0.864955 |    0.507239 |  0.922855 | 0.896305 |      0.572537 |

## 5. Pooled Predictions Results (Calculated globally over all folds)
| Model               |   Accuracy |   Balanced_Accuracy |   Precision |   Recall |   Specificity |       F1 |   ROC_AUC |   PR_AUC |
|:--------------------|-----------:|--------------------:|------------:|---------:|--------------:|---------:|----------:|---------:|
| Logistic Regression |   0.806643 |            0.743338 |    0.558667 | 0.629153 |      0.857524 | 0.591819 |  0.797198 | 0.562653 |
| Random Forest       |   0.820528 |            0.727637 |    0.605031 | 0.560087 |      0.895187 | 0.581693 |  0.844391 | 0.604756 |
| XGBoost             |   0.805521 |            0.730816 |    0.559671 | 0.596067 |      0.865564 | 0.577296 |  0.848651 | 0.590865 |
| Deep Learning       |   0.831017 |            0.771485 |    0.611134 | 0.664107 |      0.878864 | 0.63652  |  0.838154 | 0.684947 |

## 6. Pilot vs Final Comparison
- **Pilot S7**: DL Accuracy = 96.8%, DL AUC = 0.995
- **15-Subject Final LOSO**: DL Mean Accuracy (Val Threshold) = 78.11%, DL Mean AUC = 0.9040
The exceptionally high accuracy from the pilot was a thresholding artifact. The 15-subject evaluation reveals accurate discrimination (high AUC) but catastrophic inter-subject calibration (failing Accuracy without per-subject centering).

## 7. Scientific Interpretation
**Conclusion: Strong discrimination but calibration/threshold instability remains.**
The model separates stress and non-stress probabilities extremely well for a given subject (AUC). However, baseline heart-rate and HRV variation between human beings means a fixed global threshold chosen on one subject will fail on a new subject. True non-contact/wearable deployments require a dynamic, subject-calibrated baseline.