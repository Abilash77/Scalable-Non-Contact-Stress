import numpy as np
import json
import os
import sys

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train import load_data

print("Loading Data...")
inputs_train, outputs_train = load_data('train')
inputs_val, outputs_val = load_data('val')
inputs_test, outputs_test = load_data('test')

X_train = inputs_train['input_keystroke'].reshape(len(outputs_train['fusion_output']), -1)
y_train = outputs_train['fusion_output']

X_val = inputs_val['input_keystroke'].reshape(len(outputs_val['fusion_output']), -1)
y_val = outputs_val['fusion_output']

X_test = inputs_test['input_keystroke'].reshape(len(outputs_test['fusion_output']), -1)
y_test = outputs_test['fusion_output']

results = {}

def evaluate(name, y_true, y_pred, y_probs=None):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_probs) if y_probs is not None else 0.5
    cm = confusion_matrix(y_true, y_pred)
    
    results[name] = {
        'Accuracy': float(acc),
        'Precision': float(prec),
        'Recall': float(rec),
        'F1': float(f1),
        'ROC-AUC': float(auc),
        'ConfusionMatrix': cm.tolist()
    }
    
    print(f"\n[{name}]")
    print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

# Majority Baseline
majority_class = np.argmax(np.bincount(y_train))
majority_preds = np.full_like(y_test, majority_class)
majority_probs = np.zeros_like(y_test, dtype=float)
evaluate('Majority Class', y_test, majority_preds, majority_probs)

# Logistic Regression
lr = LogisticRegression(class_weight='balanced', max_iter=2000)
lr.fit(X_train, y_train)
evaluate('Logistic Regression', y_test, lr.predict(X_test), lr.predict_proba(X_test)[:, 1])

# Random Forest
rf = RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
evaluate('Random Forest', y_test, rf.predict(X_test), rf.predict_proba(X_test)[:, 1])

# SVM
svm = SVC(class_weight='balanced', probability=True, max_iter=2000)
svm.fit(X_train, y_train)
evaluate('SVM', y_test, svm.predict(X_test), svm.predict_proba(X_test)[:, 1])

# XGBoost (if available)
try:
    from xgboost import XGBClassifier
    scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)
    xgb = XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=42)
    xgb.fit(X_train, y_train)
    evaluate('XGBoost', y_test, xgb.predict(X_test), xgb.predict_proba(X_test)[:, 1])
except ImportError:
    print("\n[XGBoost] Not available in environment.")

with open('results/baseline_metrics.json', 'w') as f:
    json.dump(results, f, indent=4)
print("\nDone.")
