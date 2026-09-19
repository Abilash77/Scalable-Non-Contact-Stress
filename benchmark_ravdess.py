import os
import glob
import time
import numpy as np
import librosa
from joblib import Parallel, delayed
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from audio_utils import extract_audio_features

def old_extract_audio_features(audio_path, sr=22050):
    # Mimic the old PyIN extraction
    audio_segment, _ = librosa.load(audio_path, sr=sr)
    if audio_segment is None or len(audio_segment) == 0:
        return np.zeros(166)
    
    # Just time the PyIN part since it's the bottleneck
    f0, voiced_flag, voiced_probs = librosa.pyin(audio_segment, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)
    f0_mean = np.nanmean(f0) if np.any(~np.isnan(f0)) else 0.0
    return f0_mean

def benchmark():
    raw_data_path = "data/ravdess/raw/"
    wav_files = glob.glob(os.path.join(raw_data_path, "**", "*.wav"), recursive=True)
    if not wav_files:
        print("No RAVDESS files found.")
        return
        
    # Take a small subset for benchmarking
    sample_files = wav_files[:16]
    print(f"Benchmarking with {len(sample_files)} RAVDESS files...\n")
    
    # 1. OLD: Sequential + PyIN
    print("Running OLD (Sequential + PyIN)...")
    start_old = time.time()
    for file_path in sample_files:
        old_extract_audio_features(file_path)
    end_old = time.time()
    old_time = end_old - start_old
    print(f"OLD time for {len(sample_files)} files: {old_time:.3f} seconds\n")
    
    # 2. NEW: Parallel + YIN
    print("Running NEW (Parallel + YIN)...")
    def new_process(file_path):
        # This uses the updated extract_audio_features which has YIN
        extract_audio_features(audio_path=file_path)
        
    start_new = time.time()
    Parallel(n_jobs=4)(delayed(new_process)(f) for f in sample_files)
    end_new = time.time()
    new_time = end_new - start_new
    print(f"NEW time for {len(sample_files)} files: {new_time:.3f} seconds")
    
    speedup = old_time / new_time if new_time > 0 else 0
    print(f"\nSpeed improvement: {speedup:.1f}x faster!")

if __name__ == "__main__":
    benchmark()
