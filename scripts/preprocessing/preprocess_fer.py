import os
import argparse
import pickle
import pandas as pd
import numpy as np
import cv2
import glob

def preprocess_fer(raw_data_path, output_path):
    print(f"Starting FER2013 preprocessing from {raw_data_path}...")
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    splits = ['train', 'test']
    processed_data = []
    
    emotion_map = {
        'angry': 0,
        'disgust': 1,
        'fear': 2,
        'happy': 3,
        'neutral': 4,
        'sad': 5,
        'surprise': 6
    }
    
    for split in splits:
        split_dir = os.path.join(raw_data_path, split)
        if not os.path.exists(split_dir):
            print(f"Directory {split_dir} not found. Skipping...")
            continue
            
        for emotion, label_idx in emotion_map.items():
            emo_dir = os.path.join(split_dir, emotion)
            if not os.path.exists(emo_dir):
                continue
                
            img_paths = glob.glob(os.path.join(emo_dir, "*.jpg")) + glob.glob(os.path.join(emo_dir, "*.png"))
            print(f"Processing {split}/{emotion}: {len(img_paths)} images")
            
            for img_path in img_paths:
                # Read grayscale 48x48
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                
                # Resize just in case
                if img.shape != (48, 48):
                    img = cv2.resize(img, (48, 48))
                    
                # Normalize 0-1
                img = img.astype(np.float32) / 255.0
                
                # Expand dims to (48, 48, 1)
                img = np.expand_dims(img, axis=-1)
                
                processed_data.append({
                    'face_features': img,
                    'emotion_label': label_idx,
                    'split': split
                })
                
    if not processed_data:
        print("No FER2013 data processed.")
        return
        
    df = pd.DataFrame(processed_data)
    out_file = os.path.join(output_path, 'fer2013_processed.pkl')
    with open(out_file, 'wb') as f:
        pickle.dump(df, f)
        
    print(f"Finished. Saved {len(df)} samples to {out_file}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess FER2013 dataset")
    parser.add_argument("--raw_data_path", type=str, required=True, help="Path to raw FER2013 data")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save processed dataframes")
    
    args = parser.parse_args()
    preprocess_fer(args.raw_data_path, args.output_path)
