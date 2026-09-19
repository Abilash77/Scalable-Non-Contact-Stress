import os
import argparse
import pickle
import pandas as pd
import numpy as np
import glob
import cv2

# Add src to path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from handwriting_utils import extract_handwriting_features

def preprocess_handwriting(raw_data_path, output_path):
    print(f"Starting Handwriting preprocessing from {raw_data_path}...")
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    # The raw data is actually a CSV: handwriting_personality_large_dataset.csv
    csv_file = os.path.join(raw_data_path, 'handwriting_personality_large_dataset.csv')
    if not os.path.exists(csv_file):
        print(f"No CSV file found in {raw_data_path}. Ensure Handwriting dataset is extracted.")
        return
        
    print(f"Loading handwriting data from {csv_file}...")
    
    # Load the CSV
    raw_df = pd.read_csv(csv_file)
    print(f"Found {len(raw_df)} handwriting samples. Processing...")
    
    processed_data = []
    
    for i, row in raw_df.iterrows():
        # The model expects 6 features for handwriting.
        # We will extract Feature_1 through Feature_6 from the CSV.
        features = np.array([
            row['Feature_1'], row['Feature_2'], row['Feature_3'], 
            row['Feature_4'], row['Feature_5'], row['Feature_6']
        ], dtype=np.float32)
        
        # We also need a label and a filename for structural integrity
        filename = row['Handwriting_Sample']
        # The dataset doesn't have stress labels, we'll use Neuroticism > 0.5 as a proxy for stress just for structural mapping
        neuroticism = row['Neuroticism']
        stress_proxy = 1 if neuroticism > 0.5 else 0
        
        processed_data.append({
            'handwriting_features': features,
            'trait_label': stress_proxy,
            'filename': filename
        })
            
        if i % 5000 == 0 and i > 0:
            print(f"Processed {i}/{len(raw_df)} samples...")
        
    df = pd.DataFrame(processed_data)
    out_file = os.path.join(output_path, 'handwriting_processed.pkl')
    with open(out_file, 'wb') as f:
        pickle.dump(df, f)
        
    print(f"Finished. Saved {len(df)} samples to {out_file}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Handwriting dataset")
    parser.add_argument("--raw_data_path", type=str, required=True, help="Path to raw Handwriting data")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save processed dataframes")
    
    args = parser.parse_args()
    preprocess_handwriting(args.raw_data_path, args.output_path)
