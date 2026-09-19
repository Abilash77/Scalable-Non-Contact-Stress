import os
import argparse
import pickle
import pandas as pd
import numpy as np

def preprocess_keystrokes(raw_data_path, output_path):
    print(f"Starting Keystroke preprocessing from {raw_data_path}...")
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    csv_file = os.path.join(raw_data_path, "DSL-StrongPasswordData.csv")
    if not os.path.exists(csv_file):
        print(f"File {csv_file} not found. Ensure dataset is extracted.")
        return
        
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} rows from {csv_file}")
    
    # H columns are Hold (Dwell) times
    h_cols = [c for c in df.columns if c.startswith('H.')]
    # UD columns are Up-Down (Flight) times
    ud_cols = [c for c in df.columns if c.startswith('UD.')]
    
    processed_data = []
    
    for idx, row in df.iterrows():
        subject = row['subject']
        session = row['sessionIndex']
        
        # Calculate features per row
        h_vals = row[h_cols].values.astype(float)
        ud_vals = row[ud_cols].values.astype(float)
        
        dwell_mean = np.mean(h_vals)
        dwell_std = np.std(h_vals)
        flight_mean = np.mean(ud_vals)
        flight_std = np.std(ud_vals)
        
        # Typing speed approx (chars per sec). Total time is sum of dwell + flight.
        total_time = np.sum(h_vals) + np.sum(ud_vals)
        num_chars = len(h_vals)
        speed = num_chars / total_time if total_time > 0 else 0
        
        features = np.array([dwell_mean, dwell_std, flight_mean, flight_std, speed])
        
        processed_data.append({
            'keystroke_features': features,
            'subject': subject,
            'session': session
        })
        
    out_df = pd.DataFrame(processed_data)
    out_file = os.path.join(output_path, 'keystroke_processed.pkl')
    with open(out_file, 'wb') as f:
        pickle.dump(out_df, f)
        
    print(f"Finished. Saved {len(out_df)} samples to {out_file}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Keystroke dataset")
    parser.add_argument("--raw_data_path", type=str, required=True, help="Path to raw Keystroke data")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save processed dataframes")
    
    args = parser.parse_args()
    preprocess_keystrokes(args.raw_data_path, args.output_path)
