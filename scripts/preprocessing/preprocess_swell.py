import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
import json

def load_swell_data(data_dir):
    """
    Attempts to parse SWELL-KW dataset files.
    The SWELL-KW dataset might contain 'PP' (Participant) and 'Condition' (Stress label).
    """
    if not os.path.exists(data_dir):
        print(f"[ERROR] SWELL-KW directory not found at {data_dir}")
        return None
        
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    if not csv_files:
        print(f"[ERROR] No CSV files found in {data_dir}")
        return None
        
    # Attempt to read the first CSV file for demonstration (or a specific known file if available)
    # Ideally, we look for something like 'swell_training_data.csv' or 'dataset.csv'
    target_file = csv_files[0]
    for f in csv_files:
        if 'swell' in f.lower():
            target_file = f
            break
            
    print(f"[INFO] Reading {target_file}...")
    df = pd.read_csv(os.path.join(data_dir, target_file))
    return df

def preprocess_swell(data_dir="data/swell_kw", output_dir="data/processed/swell_kw"):
    print("="*60)
    print("SWELL-KW DATASET PREPROCESSING")
    print("="*60)
    
    df = load_swell_data(data_dir)
    if df is None:
        print("\n[BLOCKED] Cannot proceed with preprocessing. SWELL-KW dataset is missing.")
        print("Please place the legitimate SWELL-KW CSV files in data/swell_kw/")
        return False
        
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n[INFO] Dataset Summary:")
    print(f"Total Rows: {len(df)}")
    print(f"Columns: {list(df.columns)[:10]}...")
    
    # 1. Identify Subject IDs
    subject_col = None
    for col in ['PP', 'subject', 'Participant', 'id']:
        if col in df.columns:
            subject_col = col
            break
            
    if subject_col is None:
        print("[WARNING] No explicit Participant ID found. Subject-wise splitting is COMPROMISED.")
        print("Defaulting to random row-level split (WARNING: Data leakage risk).")
        subjects = np.arange(len(df))
    else:
        subjects = df[subject_col].values
        print(f"[INFO] Found {len(np.unique(subjects))} unique subjects via column '{subject_col}'.")

    # 2. Identify Labels (Stress / No Stress)
    label_col = None
    for col in ['Condition', 'label', 'stress']:
        if col in df.columns:
            label_col = col
            break
            
    if label_col is None:
        print("[ERROR] No Stress label column found. Stopping preprocessing.")
        return False
        
    # Map labels to binary (0 = Non-Stress, 1 = Stress)
    # SWELL 'Condition' might be 'N' (Neutral), 'I' (Interruption), 'T' (Time Pressure)
    def map_label(val):
        val = str(val).lower()
        if 'time' in val or 'interrupt' in val or 'stress' in val or val in ['t', 'i', '1', 1]:
            return 1
        return 0
        
    y = df[label_col].apply(map_label).values
    
    # 3. Extract Modalities
    # Here we map columns to our standard 5 modalities if possible.
    # We create synthetic zero arrays for entirely missing modalities to maintain the architecture,
    # but we will MASK them out during training.
    
    # Note: SWELL-KW typically provides HRV, not 169D Audio or 12D Face. 
    # If SWELL has Keystroke/Face features, we should parse them. For now, we extract 
    # whatever continuous numerical features exist and use them as 'physio' or 'keyboard' if matched.
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feature_cols = [c for c in numeric_cols if c not in [subject_col, label_col]]
    
    print(f"[INFO] Found {len(feature_cols)} continuous feature columns.")
    X_raw = df[feature_cols].values
    
    # Handle NaNs
    if np.isnan(X_raw).any():
        print("[INFO] Filling NaNs with column means...")
        col_means = np.nanmean(X_raw, axis=0)
        inds = np.where(np.isnan(X_raw))
        X_raw[inds] = np.take(col_means, inds[1])
        
    # Reshape for Temporal Window (T=10).
    # Since SWELL rows are typically independent in generic CSVs, we simulate a T=10 sequence
    # by repeating the row (if true time-series isn't available). 
    # WARNING: This is a dimensionality shim, not true temporal dynamics.
    T = 10
    X_temporal = np.repeat(X_raw[:, np.newaxis, :], T, axis=1) # Shape: (N, 10, Features)
    
    # We will map X_temporal to the 'keystroke' branch purely to exercise the architecture,
    # BUT if the feature dim doesn't match exactly 7 (keystroke), the fusion model will need
    # to adapt. We will save it as 'swell_features'.
    
    print("[INFO] Performing Subject-Wise Split (70/15/15)...")
    if subject_col is not None:
        gss1 = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
        train_idx, temp_idx = next(gss1.split(X_temporal, y, groups=subjects))
        
        gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
        val_idx, test_idx = next(gss2.split(X_temporal[temp_idx], y[temp_idx], groups=subjects[temp_idx]))
        
        # map temp_idx back
        val_idx = temp_idx[val_idx]
        test_idx = temp_idx[test_idx]
    else:
        # Random fallback
        indices = np.arange(len(y))
        np.random.shuffle(indices)
        train_end = int(0.7 * len(y))
        val_end = int(0.85 * len(y))
        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]
        
    # Save splits
    np.save(os.path.join(output_dir, "X_train.npy"), X_temporal[train_idx])
    np.save(os.path.join(output_dir, "y_train.npy"), y[train_idx])
    np.save(os.path.join(output_dir, "X_val.npy"), X_temporal[val_idx])
    np.save(os.path.join(output_dir, "y_val.npy"), y[val_idx])
    np.save(os.path.join(output_dir, "X_test.npy"), X_temporal[test_idx])
    np.save(os.path.join(output_dir, "y_test.npy"), y[test_idx])
    
    metadata = {
        "train_samples": len(train_idx),
        "val_samples": len(val_idx),
        "test_samples": len(test_idx),
        "feature_dim": X_temporal.shape[2],
        "subject_split": subject_col is not None
    }
    with open(os.path.join(output_dir, "dataset_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print("\n[SUCCESS] SWELL-KW preprocessing complete.")
    return True

if __name__ == "__main__":
    preprocess_swell()
