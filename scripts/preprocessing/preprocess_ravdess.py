import os
import argparse
import pickle
import pandas as pd
import numpy as np
import glob
from joblib import Parallel, delayed
from tqdm import tqdm

# Add src to path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from audio_utils import extract_audio_features

def process_single_file(file_path):
    filename = os.path.basename(file_path)
    parts = filename.replace('.wav', '').split('-')
    
    if len(parts) != 7:
        # Returning None for malformed files
        return None
        
    modality = int(parts[0])
    vocal_channel = int(parts[1])
    emotion = int(parts[2])
    intensity = int(parts[3])
    statement = int(parts[4])
    repetition = int(parts[5])
    actor = int(parts[6])
    
    features = extract_audio_features(audio_path=file_path)
    
    if features is not None and not np.all(features == 0):
        return {
            'audio_features': features,
            'emotion_label': emotion,
            'intensity': intensity,
            'actor_id': actor,
            'filename': filename
        }
    return None

def preprocess_ravdess(raw_data_path, output_path, workers=4):
    print(f"Starting RAVDESS preprocessing from {raw_data_path}...")
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    wav_files = glob.glob(os.path.join(raw_data_path, "**", "*.wav"), recursive=True)
    if not wav_files:
        print(f"No WAV files found in {raw_data_path}. Ensure RAVDESS is extracted.")
        return
        
    print(f"Found {len(wav_files)} WAV files. Processing with {workers} workers...")
    
    # Process files in parallel
    results = Parallel(n_jobs=workers)(
        delayed(process_single_file)(file_path) 
        for file_path in tqdm(wav_files, desc="Processing RAVDESS")
    )
    
    # Filter out None results (from skipped files or failures)
    processed_data = [r for r in results if r is not None]
    
    df = pd.DataFrame(processed_data)
    out_file = os.path.join(output_path, 'ravdess_processed.pkl')
    with open(out_file, 'wb') as f:
        pickle.dump(df, f)
        
    print(f"Finished. Saved {len(df)} samples to {out_file}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess RAVDESS dataset")
    parser.add_argument("--raw_data_path", type=str, required=True, help="Path to raw RAVDESS data")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save processed dataframes")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers for processing")
    
    args = parser.parse_args()
    preprocess_ravdess(args.raw_data_path, args.output_path, args.workers)
