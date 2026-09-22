import os
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

RESULTS_DIR = "results/wesad/loso_final"

def specificity_score(y_true, y_preds):
    tn, fp, fn, tp = confusion_matrix(y_true, y_preds, labels=[0,1]).ravel()
    return tn / (tn + fp + 1e-10)

def main():
    pooled_df = pd.read_csv(os.path.join(RESULTS_DIR, "pooled_predictions.csv"))
    y_true = pooled_df['True_Label'].values
    
    models = ["Logistic Regression", "Random Forest", "XGBoost", "Deep Learning"]
    
    # Calculate pooled metrics
    pooled_metrics = []
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, model in enumerate(models):
        y_probs = pooled_df[f"{model}_Prob"].values
        y_preds = (y_probs >= 0.5).astype(int)
        
        cm = confusion_matrix(y_true, y_preds)
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[idx], cmap='Blues', 
                    xticklabels=['Non-Stress', 'Stress'], yticklabels=['Non-Stress', 'Stress'])
        axes[idx].set_title(f"{model} (Pooled CM)")
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('True')
        
        pooled_metrics.append({
            'Model': model,
            'Accuracy': accuracy_score(y_true, y_preds),
            'Balanced_Accuracy': balanced_accuracy_score(y_true, y_preds),
            'Precision': precision_score(y_true, y_preds, zero_division=0),
            'Recall': recall_score(y_true, y_preds, zero_division=0),
            'Specificity': specificity_score(y_true, y_preds),
            'F1': f1_score(y_true, y_preds, zero_division=0),
            'ROC_AUC': roc_auc_score(y_true, y_probs),
            'PR_AUC': average_precision_score(y_true, y_probs)
        })
        
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"))
    plt.close()
    
    pooled_metrics_df = pd.DataFrame(pooled_metrics)
    
    macro_df = pd.read_csv(os.path.join(RESULTS_DIR, "model_comparison.csv"))
    
    # Pivot macro table for better readability
    pivoted_macro = macro_df.pivot(index=['Model', 'Threshold_Type'], columns='Metric', values='Mean')
    
    # DL Stats
    dl_val = pivoted_macro.loc[('Deep Learning', 'Threshold_Val')]
    dl_acc = dl_val['Accuracy']
    dl_auc = dl_val['ROC-AUC']
    
    report_md = [
        "# FINAL WESAD 15-SUBJECT LOSO REPORT",
        "",
        "## 1. Dataset & Scope",
        "WESAD contains strictly physiological, wearable ECG/HRV data. This validates wearable cardiac stress prediction. **It does NOT validate the main non-contact project's modalities (Facial, Eye, Keyboard, Speech).**",
        "",
        "## 2. Definitions",
        "- Target: STRESS (1) vs NON-STRESS (0: baseline/amusement/meditation)",
        "- Features: 62 physiological ECG/HRV metrics",
        "- Excluded: Label, Condition, Subject ID, Time, SSSQ",
        "",
        "## 3. Methodology",
        "- **Validation**: 15-Fold Leave-One-Subject-Out",
        "- **Training**: 12 subjects | **Validation**: 2 subjects | **Test**: 1 held-out subject",
        "- Preprocessing (Scaler) strictly fit on Training.",
        "- Model and Threshold strictly optimized on Training/Validation.",
        "",
        "## 4. Aggregate Macro Results (Averaged across 15 folds)",
        pivoted_macro.to_markdown(),
        "",
        "## 5. Pooled Predictions Results (Calculated globally over all folds)",
        pooled_metrics_df.to_markdown(index=False),
        "",
        "## 6. Pilot vs Final Comparison",
        "- **Pilot S7**: DL Accuracy = 96.8%, DL AUC = 0.995",
        f"- **15-Subject Final LOSO**: DL Mean Accuracy (Val Threshold) = {dl_acc:.2%}, DL Mean AUC = {dl_auc:.4f}",
        "The exceptionally high accuracy from the pilot was a thresholding artifact. The 15-subject evaluation reveals accurate discrimination (high AUC) but catastrophic inter-subject calibration (failing Accuracy without per-subject centering).",
        "",
        "## 7. Scientific Interpretation",
        "**Conclusion: Strong discrimination but calibration/threshold instability remains.**",
        "The model separates stress and non-stress probabilities extremely well for a given subject (AUC). However, baseline heart-rate and HRV variation between human beings means a fixed global threshold chosen on one subject will fail on a new subject. True non-contact/wearable deployments require a dynamic, subject-calibrated baseline."
    ]
    
    with open(os.path.join(RESULTS_DIR, "FINAL_LOSO_REPORT.md"), "w") as f:
        f.write("\n".join(report_md))

if __name__ == "__main__":
    main()
