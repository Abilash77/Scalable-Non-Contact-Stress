import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve
import tensorflow as tf

# Suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

TARGET_DIR = "data/wesad/raw"
RESULTS_DIR = "results/wesad"

TRAIN_SUBJECTS = [2, 3, 4, 5]
VAL_SUBJECTS = [6]
TEST_SUBJECTS = [7]

def binary_label(condition):
    return 1 if str(condition).strip().lower() == 'stress' else 0

def load_data(subjects):
    df_list = []
    for sub in subjects:
        file_path = os.path.join(TARGET_DIR, f"S{sub}", f"S{sub}_features.csv")
        if os.path.exists(file_path):
            df_list.append(pd.read_csv(file_path))
    return pd.concat(df_list, ignore_index=True)

def find_optimal_threshold(y_true, y_probs):
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    fscore = (2 * precision * recall) / (precision + recall + 1e-10)
    ix = np.argmax(fscore)
    return thresholds[ix] if ix < len(thresholds) else 0.5

def build_dl_model(input_shape):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_shape,)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def evaluate_model(y_true, y_probs, threshold):
    y_preds = (y_probs >= threshold).astype(int)
    return {
        'Accuracy': float(accuracy_score(y_true, y_preds)),
        'Balanced Accuracy': float(balanced_accuracy_score(y_true, y_preds)),
        'Precision': float(precision_score(y_true, y_preds, zero_division=0)),
        'Recall': float(recall_score(y_true, y_preds, zero_division=0)),
        'F1': float(f1_score(y_true, y_preds, zero_division=0)),
        'ROC-AUC': float(roc_auc_score(y_true, y_probs)),
        'Threshold': float(threshold)
    }

def run_pipeline(X_train, y_train, X_val, y_val, X_test, y_test, model_type="RF"):
    if model_type == "RF":
        model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        val_probs = model.predict_proba(X_val)[:, 1]
        test_probs = model.predict_proba(X_test)[:, 1]
    elif model_type == "LR":
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        val_probs = model.predict_proba(X_val)[:, 1]
        test_probs = model.predict_proba(X_test)[:, 1]
    elif model_type == "XGB":
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        model.fit(X_train, y_train)
        val_probs = model.predict_proba(X_val)[:, 1]
        test_probs = model.predict_proba(X_test)[:, 1]
    elif model_type == "DL":
        model = build_dl_model(X_train.shape[1])
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5, batch_size=64, verbose=0)
        val_probs = model.predict(X_val, verbose=0).flatten()
        test_probs = model.predict(X_test, verbose=0).flatten()
    else:
        raise ValueError("Unknown model type")

    opt_thresh = find_optimal_threshold(y_val, val_probs)
    
    res_def = evaluate_model(y_test, test_probs, 0.5)
    res_opt = evaluate_model(y_test, test_probs, opt_thresh)
    
    feature_importances = None
    if model_type == "RF":
        feature_importances = model.feature_importances_
        
    return res_def, res_opt, feature_importances

def main():
    print("Loading data...")
    train_df = load_data(TRAIN_SUBJECTS)
    val_df = load_data(VAL_SUBJECTS)
    test_df = load_data(TEST_SUBJECTS)
    
    for df in [train_df, val_df, test_df]:
        df['binary_label'] = df['condition'].apply(binary_label)
        
    cols_to_drop = ['subject id', 'condition', 'SSSQ', 'Time', 'binary_label']
    feature_cols = [c for c in train_df.columns if c not in cols_to_drop]
    
    scaler = StandardScaler()
    X_train_full = scaler.fit_transform(train_df[feature_cols].fillna(train_df[feature_cols].mean()))
    X_val_full = scaler.transform(val_df[feature_cols].fillna(train_df[feature_cols].mean()))
    X_test_full = scaler.transform(test_df[feature_cols].fillna(train_df[feature_cols].mean()))
    
    y_train = train_df['binary_label'].values
    y_val = val_df['binary_label'].values
    y_test = test_df['binary_label'].values
    
    results = {}
    
    # 3. Model Comparisons
    print("Evaluating models...")
    for mt in ["LR", "RF", "XGB", "DL"]:
        print(f"  Training {mt}...")
        res_def, res_opt, importances = run_pipeline(X_train_full, y_train, X_val_full, y_val, X_test_full, y_test, mt)
        results[mt] = {
            "Default 0.5": res_def,
            "Validation Threshold": res_opt
        }
        if mt == "RF":
            rf_importances = importances
            
    # 5. Permutation Test
    print("Running permutation test (DL)...")
    y_train_permuted = np.random.permutation(y_train)
    res_def_perm, res_opt_perm, _ = run_pipeline(X_train_full, y_train_permuted, X_val_full, y_val, X_test_full, y_test, "DL")
    results["Permutation Test (DL)"] = {
        "Default 0.5": res_def_perm,
        "Validation Threshold": res_opt_perm
    }
    
    # 6. Feature Ablation (Top 10)
    print("Running feature ablation test (Top 10 RF features)...")
    top_10_idx = np.argsort(rf_importances)[-10:]
    top_10_cols = [feature_cols[i] for i in top_10_idx]
    
    X_train_top10 = X_train_full[:, top_10_idx]
    X_val_top10 = X_val_full[:, top_10_idx]
    X_test_top10 = X_test_full[:, top_10_idx]
    
    res_def_top10, res_opt_top10, _ = run_pipeline(X_train_top10, y_train, X_val_top10, y_val, X_test_top10, y_test, "DL")
    results["Ablation (Top 10) (DL)"] = {
        "Top 10 Features": top_10_cols,
        "Default 0.5": res_def_top10,
        "Validation Threshold": res_opt_top10
    }
    
    # Check for defense
    dl_res = results["DL"]["Validation Threshold"]
    perm_res = results["Permutation Test (DL)"]["Validation Threshold"]
    
    is_defensible = True
    reasons = []
    if perm_res['ROC-AUC'] > 0.7:
        is_defensible = False
        reasons.append("Permutation test failed (random labels still yielded high performance).")
    if dl_res['ROC-AUC'] > 0.99 and dl_res['F1'] > 0.98:
        reasons.append("Performance is extraordinarily high, suggesting hidden leakage in features (e.g. physiological overlap).")
        
    validation_status = "CLEARED FOR LOSO" if is_defensible else "NOT CLEARED — ISSUE FOUND"
    
    output_report = {
        "status": validation_status,
        "reasons": reasons,
        "results": results
    }
    
    with open(os.path.join(RESULTS_DIR, "PILOT_FINAL_VALIDATION.json"), "w") as f:
        json.dump(output_report, f, indent=4)
        
    md_lines = [
        "# WESAD Pilot Final Validation",
        "",
        "## Model Comparison",
        "```json",
        json.dumps(results, indent=2),
        "```",
        "",
        "## Conclusion",
        f"**{validation_status}**"
    ]
    if reasons:
        md_lines.append("\n" + "\n".join(["- " + r for r in reasons]))
        
    with open(os.path.join(RESULTS_DIR, "PILOT_FINAL_VALIDATION.md"), "w") as f:
        f.write("\n".join(md_lines))
        
    print("Done. Validation report saved.")

if __name__ == "__main__":
    main()
