import os
import pandas as pd
import numpy as np

class FordigitstressPDAdapter:
    def __init__(self, subject_dir):
        self.subject_dir = subject_dir
        
    def extract_labels(self):
        annotation_file = os.path.join(self.subject_dir, "stress.csv")
        if not os.path.exists(annotation_file):
            raise FileNotFoundError(f"Label parsing blocked. Annotation file not found: {annotation_file}")
            
        try:
            df = pd.read_csv(annotation_file, sep=';')
            if 'stress' not in df.columns:
                raise ValueError("stress.csv missing required column 'stress'")
            return df
        except Exception as e:
            raise Exception(f"Failed to parse label file {annotation_file}: {e}")
            
    def extract_eye(self):
        """
        Extracts pupil features according to the documented pupil_features.csv
        and maps them to the Unimodal PD 5D format: (pd, dx, dy, conf, interp)
        """
        pupil_file = os.path.join(self.subject_dir, "pupil_features.csv")
        if not os.path.exists(pupil_file):
            raise FileNotFoundError(f"Eye extraction blocked. File not found: {pupil_file}")
            
        try:
            df = pd.read_csv(pupil_file, sep=';')
            if 'pupil_diameter' not in df.columns or 'confidence' not in df.columns:
                raise ValueError("pupil_features.csv missing required columns 'pupil_diameter' or 'confidence'")
            
            num_samples = len(df)
            eye_features = np.zeros((num_samples, 5), dtype=np.float32)
            eye_features[:, 0] = df['pupil_diameter'].values
            eye_features[:, 3] = df['confidence'].values
            
            if 'pd_interp' in df.columns:
                eye_features[:, 4] = df['pd_interp'].values
                
            return eye_features
        except Exception as e:
            raise Exception(f"Failed to parse pupil file {pupil_file}: {e}")

class PDSplitter:
    def split(self, subject_ids, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        raise NotImplementedError("Subject splitting blocked. Awaiting valid subject inventory.")

class PDDatasetValidator:
    def validate_directory(self, data_dir="data/fordigitstress/"):
        print("\n" + "="*60)
        print("RESEARCH PD-ONLY FORDIGITSTRESS DATASET VALIDATION REPORT")
        print("="*60)
        
        if not os.path.exists(data_dir):
            print(f"[FAIL] Directory not found: {os.path.abspath(data_dir)}")
            print("\nTRAINING READINESS: BLOCKED")
            print("="*60 + "\n")
            return False
            
        entries = os.listdir(data_dir)
        subject_dirs = sorted([d for d in entries if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('VP')])
        
        print(f"[INFO] Discovered Subjects: {len(subject_dirs)}")
        if len(subject_dirs) == 0:
            print("[FAIL] No valid VPxx subject directories found.")
            print("\nTRAINING READINESS: BLOCKED")
            print("="*60 + "\n")
            return False
            
        valid_labels = 0
        valid_pupil = 0
        
        for vp in subject_dirs:
            vp_dir = os.path.join(data_dir, vp)
            if os.path.exists(os.path.join(vp_dir, "stress.csv")):
                valid_labels += 1
            if os.path.exists(os.path.join(vp_dir, "pupil_features.csv")):
                valid_pupil += 1
                
        print("\nMODALITY READINESS:")
        print(f"- Labels (stress.csv): {valid_labels}/{len(subject_dirs)} subjects")
        print(f"- Eye/Pupil (pupil_features.csv): {valid_pupil}/{len(subject_dirs)} subjects")
        
        print("\nPIPELINE STATUS:")
        print("- Temporal Alignment: PENDING TIMESTAMP DISCOVERY")
        print("- T=10 Construction: PENDING DATA")
        
        print("\nTRAINING READINESS: BLOCKED (Awaiting dataset resolution)")
        print("="*60 + "\n")
        return False
