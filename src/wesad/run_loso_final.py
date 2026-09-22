import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, 
                             precision_score, recall_score, f1_score, 
                             roc_auc_score, precision_recall_curve, confusion_matrix,
                             average_precision_score, brier_score_loss)
from sklearn.calibration import calibration_curve
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

TARGET_DIR = "data/wesad/raw"
RESULTS_DIR = "results/wesad/loso_final"
FOLDS_DIR = os.path.join(RESULTS_DIR, "folds")
PREDS_DIR = os.path.join(RESULTS_DIR, "predictions")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FOLDS_DIR, exist_ok=True)
os.makedirs(PREDS_DIR, exist_ok=True)

SUBJECTS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
MODELS = ["Logistic Regression", "Random Forest", "XGBoost", "Deep Learning"]

# Convert numpy to python types for JSON serialization
def to_native(obj):
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    return obj

def binary_label(condition):
    return 1 if str(condition).strip().lower() == 'stress' else 0

def load_data(subjects):
    df_list = []
    for sub in subjects:
        file_path = os.path.join(TARGET_DIR, f"S{sub}", f"S{sub}_features.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df['subject_id_raw'] = sub
            df_list.append(df)
    return pd.concat(df_list, ignore_index=True)

def find_optimal_threshold(y_true, y_probs):
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    fscore = (2 * precision * recall) / (precision + recall + 1e-10)
    ix = np.argmax(fscore)
    thresh = float(thresholds[ix]) if ix < len(thresholds) else 0.5
    # Constrain pathological thresholds
    if thresh < 1e-5: thresh = 1e-5
    if thresh > 1.0 - 1e-5: thresh = 1.0 - 1e-5
    return thresh

def build_dl_model(input_shape):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_shape,)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model

def specificity_score(y_true, y_preds):
    tn, fp, fn, tp = confusion_matrix(y_true, y_preds, labels=[0,1]).ravel()
    return tn / (tn + fp + 1e-10)

def evaluate_predictions(y_true, y_probs, threshold):
    y_preds = (y_probs >= threshold).astype(int)
    
    try: auc_val = roc_auc_score(y_true, y_probs)
    except: auc_val = 0.5
    
    try: pr_auc = average_precision_score(y_true, y_probs)
    except: pr_auc = 0.0
    
    try: brier = brier_score_loss(y_true, y_probs)
    except: brier = 1.0
    
    return {
        'Accuracy': float(accuracy_score(y_true, y_preds)),
        'Balanced Accuracy': float(balanced_accuracy_score(y_true, y_preds)),
        'Precision': float(precision_score(y_true, y_preds, zero_division=0)),
        'Recall': float(recall_score(y_true, y_preds, zero_division=0)),
        'Specificity': float(specificity_score(y_true, y_preds)),
        'F1': float(f1_score(y_true, y_preds, zero_division=0)),
        'ROC-AUC': float(auc_val),
        'PR-AUC': float(pr_auc),
        'Brier_Score': float(brier),
        'Threshold': float(threshold),
        'Pred_Stress': int(y_preds.sum()),
        'Pred_NonStress': int(len(y_preds) - y_preds.sum()),
        'Prob_Min': float(np.min(y_probs)),
        'Prob_Max': float(np.max(y_probs)),
        'Prob_Mean': float(np.mean(y_probs)),
        'Prob_Median': float(np.median(y_probs))
    }

def main():
    print("Loading WESAD data...")
    full_df = load_data(SUBJECTS)
    full_df['binary_label'] = full_df['condition'].apply(binary_label)
    
    cols_to_drop = ['subject id', 'condition', 'SSSQ', 'Time', 'binary_label', 'subject_id_raw']
    feature_cols = [c for c in full_df.columns if c not in cols_to_drop]
    
    print(f"Total rows: {len(full_df)}")
    
    all_fold_results = []
    
    # Check for existing folds to resume
    for test_sub in SUBJECTS:
        fold_json = os.path.join(FOLDS_DIR, f"S{test_sub:02d}.json")
        pred_csv = os.path.join(PREDS_DIR, f"S{test_sub:02d}_predictions.csv")
        
        if os.path.exists(fold_json) and os.path.exists(pred_csv):
            print(f"Skipping Subject {test_sub} (Already completed)")
            with open(fold_json, "r") as f:
                all_fold_results.append(json.load(f))
            continue
            
        print(f"\n--- LOSO Fold: Test Subject {test_sub} ---")
        
        dev_subjects = [s for s in SUBJECTS if s != test_sub]
        val_subjects = dev_subjects[:2]
        train_subjects = dev_subjects[2:]
        
        train_df = full_df[full_df['subject_id_raw'].isin(train_subjects)]
        val_df = full_df[full_df['subject_id_raw'].isin(val_subjects)]
        test_df = full_df[full_df['subject_id_raw'] == test_sub]
        
        # Leakage checks
        assert len(set(train_df['subject_id_raw']).intersection(set(test_df['subject_id_raw']))) == 0, "Train/Test overlap!"
        assert len(set(val_df['subject_id_raw']).intersection(set(test_df['subject_id_raw']))) == 0, "Val/Test overlap!"
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_df[feature_cols].fillna(0))
        X_val = scaler.transform(val_df[feature_cols].fillna(0))
        X_test = scaler.transform(test_df[feature_cols].fillna(0))
        
        y_train = train_df['binary_label'].values
        y_val = val_df['binary_label'].values
        y_test = test_df['binary_label'].values
        
        fold_metrics = {
            "Test_Subject": int(test_sub),
            "True_Stress": int(y_test.sum()),
            "True_NonStress": int(len(y_test) - y_test.sum())
        }
        
        preds_df = pd.DataFrame({
            "Subject_ID": [test_sub] * len(y_test),
            "True_Label": y_test
        })
        
        for m_name in MODELS:
            print(f"  Training {m_name}...")
            if m_name == "Logistic Regression":
                model = LogisticRegression(max_iter=1000, random_state=42)
                model.fit(X_train, y_train)
                val_probs = model.predict_proba(X_val)[:, 1]
                test_probs = model.predict_proba(X_test)[:, 1]
            elif m_name == "Random Forest":
                model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
                model.fit(X_train, y_train)
                val_probs = model.predict_proba(X_val)[:, 1]
                test_probs = model.predict_proba(X_test)[:, 1]
            elif m_name == "XGBoost":
                model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
                model.fit(X_train, y_train)
                val_probs = model.predict_proba(X_val)[:, 1]
                test_probs = model.predict_proba(X_test)[:, 1]
            elif m_name == "Deep Learning":
                model = build_dl_model(X_train.shape[1])
                model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5, batch_size=64, verbose=0)
                val_probs = model.predict(X_val, verbose=0).flatten()
                test_probs = model.predict(X_test, verbose=0).flatten()
            
            val_thresh = find_optimal_threshold(y_val, val_probs)
            
            fold_metrics[m_name] = {
                "Threshold_0.5": evaluate_predictions(y_test, test_probs, 0.5),
                "Threshold_Val": evaluate_predictions(y_test, test_probs, val_thresh)
            }
            
            preds_df[f"{m_name}_Prob"] = test_probs.astype(float)
            
        # Save fold
        all_fold_results.append(fold_metrics)
        with open(fold_json, "w") as f:
            json.dump(fold_metrics, f, indent=4)
            
        preds_df.to_csv(pred_csv, index=False)
        
        # Incremental saving of fold_results.csv
        pd.DataFrame([{
            "Subject": m["Test_Subject"], 
            **{f"{mod}_AUC": m[mod]["Threshold_0.5"]["ROC-AUC"] for mod in MODELS},
            **{f"{mod}_Acc_ValThresh": m[mod]["Threshold_Val"]["Accuracy"] for mod in MODELS}
        } for m in all_fold_results]).to_csv(os.path.join(RESULTS_DIR, "fold_results.csv"), index=False)
        
    print("\n--- Generating Aggregations ---")
    
    # Macro Aggregation
    macro_records = []
    for m_name in MODELS:
        for t_type in ["Threshold_0.5", "Threshold_Val"]:
            for metric in ['Accuracy', 'Balanced Accuracy', 'Precision', 'Recall', 'Specificity', 'F1', 'ROC-AUC', 'PR-AUC']:
                vals = [fold[m_name][t_type][metric] for fold in all_fold_results]
                macro_records.append({
                    "Model": m_name,
                    "Threshold_Type": t_type,
                    "Metric": metric,
                    "Mean": np.mean(vals),
                    "Std": np.std(vals),
                    "Median": np.median(vals),
                    "Min": np.min(vals),
                    "Max": np.max(vals)
                })
    
    macro_df = pd.DataFrame(macro_records)
    macro_df.to_csv(os.path.join(RESULTS_DIR, "model_comparison.csv"), index=False)
    
    # Pooled Predictions
    pooled_dfs = [pd.read_csv(os.path.join(PREDS_DIR, f"S{s:02d}_predictions.csv")) for s in SUBJECTS]
    pooled_df = pd.concat(pooled_dfs, ignore_index=True)
    pooled_df.to_csv(os.path.join(RESULTS_DIR, "pooled_predictions.csv"), index=False)
    
    pooled_y = pooled_df["True_Label"].values
    
    # Calibration Curve Plot
    plt.figure(figsize=(8,8))
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for m_name in MODELS:
        prob_pos = pooled_df[f"{m_name}_Prob"].values
        fraction_of_positives, mean_predicted_value = calibration_curve(pooled_y, prob_pos, n_bins=10)
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", label=f"{m_name}")
    plt.ylabel("Fraction of positives")
    plt.xlabel("Mean predicted value")
    plt.title('Calibration Curve (Reliability Diagram) - Pooled')
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, "calibration_curve.png"))
    plt.close()

    # Create FINAL JSON
    with open(os.path.join(RESULTS_DIR, "FINAL_LOSO_RESULTS.json"), "w") as f:
        json.dump(all_fold_results, f, indent=4)
        
    print("Done. Generating markdown reports via separate script if needed.")

if __name__ == "__main__":
    main()
