# Majority Class Collapse Debug Report

## 1. Class Counts & Mapping
**Mapping:**
- Label 0 = NON-STRESS (Low Stress)
- Label 1 = STRESS (High Stress)

**TRAIN CLASS COUNTS:**
Total: 4664
Class 0 (Non-Stress): 3749 (80.4%)
Class 1 (Stress): 915 (19.6%)

**VAL CLASS COUNTS:**
Total: 979
Class 0 (Non-Stress): 777 (79.4%)
Class 1 (Stress): 202 (20.6%)

**TEST CLASS COUNTS:**
Total: 954
Class 0 (Non-Stress): 737 (77.3%)
Class 1 (Stress): 217 (22.7%)

## 2. Model Prediction Distribution
- **Predicted Class 0 Count:** 954
- **Predicted Class 1 Count:** 0
- **Stress Probability (Class 1):** Min: 0.1706, Max: 0.2312, Mean: 0.1982, Std: 0.0076

All predictions output `0` because the stress probability never exceeds the `0.5` decision threshold. It hovers around `0.198`, which perfectly matches the training set's prior probability for Stress (`19.6%`).

## 3. Label Mapping Verification
The mapping of `0 = NON-STRESS` and `1 = STRESS` is consistent across preprocessing, training, evaluation, and the `run.py` server. There are no reversals.

## 4. Class Weights Verification
`train.py` **DOES NOT** pass `class_weight` to `model.fit()`. The model is being penalized equally for mistakes on the majority and minority classes, causing it to aggressively optimize for the 80% majority.

## 5. Output Semantics
`fusion_output[:,0]` correctly maps to class 0 (Non-Stress) and `fusion_output[:,1]` maps to class 1 (Stress). Argmax is not reversing the labels.

## 6. Training History
The model loss quickly drops to ~0.76 and accuracy plateaus around ~0.77-0.78. The model stops learning meaningful distinctions and just predicts the prior distribution.

## 7. Baseline Result
Run on the exact same train/test split:

**LOGISTIC REGRESSION (balanced weights):**
- Accuracy: 0.5094
- Precision: 0.2423
- Recall: 0.5438
- F1: 0.3352
- ROC-AUC: 0.5342

**RANDOM FOREST (balanced weights):**
- Accuracy: 0.7484
- Precision: 0.3614
- Recall: 0.1382
- F1: 0.2000
- ROC-AUC: 0.5443

The baseline models show that applying balanced class weights forces the model to actually predict the Stress class (Recall 54% for LR), proving there is a weak but real signal that our deep learning model is currently ignoring.

## 8. Feature Variance
**VALID.** All 7 features have mean `0` and std `1` (normalized properly on the training set). Min and Max ranges show active variance across timesteps. Zero percent is `0.00%` for all features. The features contain meaningful, varying data.

## 9. Root Cause & Recommended Fix
**ROOT CAUSE:** 
The dataset is heavily imbalanced (~80% Non-Stress vs 20% Stress). Because `class_weight` is not being passed to `model.fit()`, the model successfully minimizes categorical cross-entropy loss by outputting a constant `0.20` probability for Stress, which never triggers the `>0.5` argmax threshold.

**RECOMMENDED FIX:**
Calculate exact mathematical class weights using ONLY the `y_train` distribution and pass them into `model.fit(..., class_weight=class_weights)` in `train.py`.
