import os
import glob
import datetime
import numpy as np
import pandas as pd
import json
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

def parse_pam_logs(pam_file):
    """ Parses PAM logs into a list of dictionaries. """
    pam_records = []
    with open(pam_file, 'r') as f:
        for line in f:
            parts = line.strip().split(', ')
            if not parts or len(parts) < 7:
                continue
                
            record = {}
            for part in parts:
                if ': ' in part:
                    k, v = part.split(': ', 1)
                    if k == 'time':
                        record[k] = datetime.datetime.strptime(v, "%Y-%m-%d %H:%M:%S")
                    else:
                        record[k] = int(v)
            if 'time' in record:
                # STRESS: Negative Affect >= 8 or Arousal >= 3 & Valence <= 2
                if record.get('pam_na', 0) >= 8 or (record.get('arousal', 0) >= 3 and record.get('valence', 4) <= 2):
                    record['stress_label'] = 1
                else:
                    record['stress_label'] = 0
                pam_records.append(record)
    return pam_records

def parse_keystroke_logs(log_file):
    """ Parses raw keystroke logs into a list of events. """
    events = []
    with open(log_file, 'r') as f:
        for line in f:
            if 'KeyHooker' not in line:
                continue
            
            parts = [p.strip() for p in line.split('|')]
            if len(parts) < 4:
                continue
                
            time_str = parts[0]
            try:
                dt = datetime.datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S.%f")
            except ValueError:
                continue
                
            action = 'press' if 'NSKeyDown' in parts[2] else 'release' if 'NSKeyUp' in parts[2] else None
            if not action:
                continue
                
            meta = parts[3]
            key = None
            if 'key=' in meta:
                key_str = meta.split('key=')[1].split(' ')[0]
                key = key_str
                
            if key:
                events.append({
                    'time': dt.timestamp(),
                    'action': action,
                    'key': key
                })
    return events

def extract_7d_features(events):
    from keystroke_utils import extract_keystroke_features
    return extract_keystroke_features(events)

def preprocess(data_dir="data/keystroke-stress/data", output_dir="data/processed/keystroke_stress"):
    print("="*60)
    print("PREPROCESSING KEYSTROKE-STRESS DATASET")
    print("="*60)
    
    os.makedirs(output_dir, exist_ok=True)
    pam_file = os.path.join(data_dir, 'self_report', 'pam_log.txt')
    if not os.path.exists(pam_file):
        print(f"[ERROR] PAM log missing at {pam_file}")
        return False
        
    pam_records = parse_pam_logs(pam_file)
    print(f"[INFO] Parsed {len(pam_records)} PAM self-reports.")
    
    key_logs = glob.glob(os.path.join(data_dir, 'keys_and_mouse', '*.txt'))
    print(f"[INFO] Found {len(key_logs)} keystroke log files.")
    
    all_events = []
    for f in key_logs:
        events = parse_keystroke_logs(f)
        all_events.extend(events)
        
    all_events.sort(key=lambda x: x['time'])
    print(f"[INFO] Extracted {len(all_events)} keystroke events.")
    
    if not all_events:
        print("[ERROR] No keystrokes found.")
        return False
        
    X = []
    y = []
    
    T = 10
    LOOKBACK_MINUTES = 30
    CHUNK_MINUTES = LOOKBACK_MINUTES / T
    
    for pam in pam_records:
        pam_time = pam['time'].timestamp()
        start_time = pam_time - (LOOKBACK_MINUTES * 60)
        
        window_events = [e for e in all_events if start_time <= e['time'] <= pam_time]
        if len(window_events) < 50:
            continue
            
        chunks = []
        for i in range(T):
            c_start = start_time + (i * CHUNK_MINUTES * 60)
            c_end = c_start + (CHUNK_MINUTES * 60)
            c_events = [e for e in window_events if c_start <= e['time'] < c_end]
            feats = extract_7d_features(c_events)
            chunks.append(feats)
            
        X.append(chunks)
        y.append(pam['stress_label'])
        
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    
    print(f"[INFO] Generated {len(X)} true temporal sequences of shape {X.shape}.")
    
    if len(X) < 2:
        print("[ERROR] Not enough data samples generated.")
        return False
        
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    split_idx = int(0.8 * len(X))
    if split_idx == 0 or split_idx == len(X):
        split_idx = len(X) // 2
        
    X_train, y_train = X[indices[:split_idx]], y[indices[:split_idx]]
    X_val, y_val = X[indices[split_idx:]], y[indices[split_idx:]]
    
    np.save(os.path.join(output_dir, "X_train.npy"), X_train)
    np.save(os.path.join(output_dir, "y_train.npy"), y_train)
    np.save(os.path.join(output_dir, "X_val.npy"), X_val)
    np.save(os.path.join(output_dir, "y_val.npy"), y_val)
    
    metadata = {
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "feature_dim": 7,
        "temporal_length": T
    }
    with open(os.path.join(output_dir, "dataset_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print("[SUCCESS] Preprocessing complete.")
    return True

if __name__ == "__main__":
    preprocess()
