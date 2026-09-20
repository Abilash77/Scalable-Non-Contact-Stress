import numpy as np
import json
import os
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import load_data, get_model, custom_fusion_loss

# 1. Print Class Counts & Label Mapping
with open('results/freihaut_preprocessing_metadata.json', 'r') as f:
    meta = json.load(f)

label_mapping = meta.get("label_mapping", {})
print("\n--- 1. CLASS COUNTS & MAPPING ---")
print(f"CLASS MAPPING: {label_mapping}")
print("Label 0 = NON-STRESS (Low Stress)")
print("Label 1 = STRESS (High Stress)")

def print_split_stats(split_name, y):
    count_0 = np.sum(y == 0)
    count_1 = np.sum(y == 1)
    total = len(y)
    print(f"\n{split_name} CLASS COUNTS:")
    print(f"Total: {total}")
    print(f"Class 0 (Non-Stress): {count_0} ({count_0/total*100:.1f}%)")
    print(f"Class 1 (Stress): {count_1} ({count_1/total*100:.1f}%)")

inputs_train, outputs_train = load_data('train')
inputs_val, outputs_val = load_data('val')
inputs_test, outputs_test = load_data('test')

y_train = outputs_train['fusion_output']
y_val = outputs_val['fusion_output']
y_test = outputs_test['fusion_output']

print_split_stats('TRAIN', y_train)
print_split_stats('VAL', y_val)
print_split_stats('TEST', y_test)

# 2. Prediction Distribution
print("\n--- 2. MODEL PREDICTION DISTRIBUTION ---")
# Load model
input_shapes = {
    'audio': (10, 169),
    'face': (10, 12),
    'keystroke': (10, 7),
    'handwriting': (10, 9),
    'eye': (10, 5)
}
model = get_model('fusion', input_shapes=input_shapes, num_classes=2)
model.load_weights('results/checkpoints/freihaut_keyboard_stress.keras')

predictions = model.predict(inputs_test, verbose=0)
y_pred_probs = predictions['fusion_output']
y_pred_classes = np.argmax(y_pred_probs, axis=1)

count_pred_0 = np.sum(y_pred_classes == 0)
count_pred_1 = np.sum(y_pred_classes == 1)
print(f"Predicted Class 0 Count: {count_pred_0}")
print(f"Predicted Class 1 Count: {count_pred_1}")
print(f"Stress Probability (Class 1) - Min: {np.min(y_pred_probs[:,1]):.4f}, Max: {np.max(y_pred_probs[:,1]):.4f}, Mean: {np.mean(y_pred_probs[:,1]):.4f}, Std: {np.std(y_pred_probs[:,1]):.4f}")

print("\nSample Predictions (True | Pred | Stress Prob):")
for i in range(20):
    print(f"{y_test[i]} | {y_pred_classes[i]} | {y_pred_probs[i, 1]:.4f}")

# 6. Training History
print("\n--- 6. TRAINING HISTORY ---")
import pandas as pd
if os.path.exists('results/checkpoints/training_history.csv'):
    hist = pd.read_csv('results/checkpoints/training_history.csv')
    print(hist[['epoch', 'loss', 'fusion_output_accuracy', 'val_loss', 'val_fusion_output_accuracy']].tail(10).to_string())
else:
    print("No training_history.csv found.")

# 7. Baseline Check
print("\n--- 7. BASELINE MODELS ---")
# Flatten sequences for standard ML
X_train_flat = inputs_train['input_keystroke'].reshape(len(y_train), -1)
X_test_flat = inputs_test['input_keystroke'].reshape(len(y_test), -1)

# Logistic Regression
lr = LogisticRegression(class_weight='balanced', max_iter=1000)
lr.fit(X_train_flat, y_train)
lr_preds = lr.predict(X_test_flat)
lr_probs = lr.predict_proba(X_test_flat)[:, 1]

print("LOGISTIC REGRESSION:")
print(f"Accuracy: {accuracy_score(y_test, lr_preds):.4f}")
print(f"Precision: {precision_score(y_test, lr_preds, zero_division=0):.4f}")
print(f"Recall: {recall_score(y_test, lr_preds, zero_division=0):.4f}")
print(f"F1: {f1_score(y_test, lr_preds, zero_division=0):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, lr_probs):.4f}")

# Random Forest
rf = RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42)
rf.fit(X_train_flat, y_train)
rf_preds = rf.predict(X_test_flat)
rf_probs = rf.predict_proba(X_test_flat)[:, 1]

print("\nRANDOM FOREST:")
print(f"Accuracy: {accuracy_score(y_test, rf_preds):.4f}")
print(f"Precision: {precision_score(y_test, rf_preds, zero_division=0):.4f}")
print(f"Recall: {recall_score(y_test, rf_preds, zero_division=0):.4f}")
print(f"F1: {f1_score(y_test, rf_preds, zero_division=0):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, rf_probs):.4f}")

# 8. Feature Variance
print("\n--- 8. FEATURE VARIANCE (TRAIN ONLY) ---")
X_train = inputs_train['input_keystroke']
feature_names = meta.get("feature_names", [])

for i, fn in enumerate(feature_names):
    feats = X_train[:, :, i]
    print(f"\nFeature: {fn}")
    print(f"  Mean: {np.mean(feats):.4f}")
    print(f"  Std: {np.std(feats):.4f}")
    print(f"  Min: {np.min(feats):.4f}")
    print(f"  Max: {np.max(feats):.4f}")
    zero_pct = np.sum(feats == 0) / feats.size * 100
    print(f"  Zero %: {zero_pct:.2f}%")
