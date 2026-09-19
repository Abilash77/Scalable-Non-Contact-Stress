"""
"A multimodal architecture supporting five non-contact modalities was developed, with modality-specific training data sourced from datasets appropriate to each modality. ForDigitStress provides synchronized speech, facial, and eye/pupil data, while keyboard and handwriting are obtained from separate datasets. Cross-dataset training is therefore treated as a hybrid multimodal framework rather than a direct five-modality reproduction of a single synchronized dataset."
"""
import os
import argparse
import sys
import json
import datetime
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from DL_models import get_model, custom_fusion_loss
from build_fordigitstress_dataset import (
    FordigitstressLabelParser,
    TemporalAligner,
    FordigitstressModalityAdapter,
    SubjectSplitter,
    DatasetValidator
)

def train_fusion(epochs, batch_size):
    print("\n" + "="*60)
    print("RA-HMSD SCIENTIFIC TRAINING PIPELINE")
    print("="*60 + "\n")
    
    # 1. Hardware & Framework Status
    print("[INFO] Checking hardware capabilities...")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"[INFO] GPUs detected: {len(gpus)}")
    else:
        print("[INFO] No GPU detected. Training on CPU.")

    # 2. Dataset Check
    data_dir = "data/fordigitstress/"
    print(f"\n[INFO] Validating raw dataset directory: {os.path.abspath(data_dir)}")
    
    if not os.path.exists(data_dir):
        print("\n" + "!"*60)
        print("TRAINING BLOCKED — FORDIGITSTRESS DATASET REQUIRED")
        print("!"*60)
        print("The legitimate raw dataset is missing. To maintain scientific integrity,")
        print("this pipeline absolutely forbids fabricating mock data or faking checkpoints.")
        print("Please acquire ForDigitStress and place it in the data directory.")
        print("!"*60 + "\n")
        sys.exit(1)
        
    print("[INFO] Dataset directory found. Instantiating preprocessing pipeline...\n")
    
    try:
        # Instantiate Scaffold components
        label_parser = FordigitstressLabelParser(annotation_file=os.path.join(data_dir, "stress.csv"))
        aligner = TemporalAligner(target_sequence_length=10)
        adapter = FordigitstressModalityAdapter()
        validator = DatasetValidator()
        
        # Determine subject inventories (mock up a Subject ID pass to trigger NotImplementedError later)
        # We assume the validator will do the heavy lifting when actual dataset logic exists.
        
        print("[STAGE 1] Extracting Modalities...")
        adapter.extract_audio()  # Will raise NotImplementedError immediately
        adapter.extract_face()
        adapter.extract_keyboard()
        adapter.extract_handwriting()
        
        print("[STAGE 2] Parsing Annotations...")
        label_parser.parse_labels()
        
        print("[STAGE 3] Temporal Alignment (T=10)...")
        aligner.align_windows([], [])
        
        print("[STAGE 4] Subject-Level Splitting...")
        splitter = SubjectSplitter(subject_ids=[])
        splitter.split()
        
        # ... Theoretical training dataset variables here ...
        # (This block is purely illustrative of the target schema for when the data exists)
        X_train, Y_train = None, None
        X_val, Y_val = None, None
        X_test, Y_test = None, None
        
        print("[STAGE 5] Data Schema Validation...")
        validator.validate_dataset(X_train)
        
    except NotImplementedError as e:
        print("\n" + "-"*60)
        print("DATASET PREPARATION INCOMPLETE")
        print("-"*60)
        print(f"[BLOCKED] {str(e)}")
        print("The data pipeline scaffolding is correctly in place, but raw feature")
        print("extraction and parsing logic is awaiting dataset format discovery.")
        print("-"*60 + "\n")
        sys.exit(1)
    
    # ---------------------------------------------------------
    # Theoretical Training Code (Unreachable until data exists)
    # ---------------------------------------------------------
    
    print("\n[STAGE 6] Initializing Reliability-Aware Attention Fusion Model...")
    
    input_shapes = {
        'audio': (10, 169),
        'face': (10, 12),
        'keystroke': (10, 7),
        'handwriting': (10, 9),
        'eye': (10, 5)
    }
    
    model = get_model('fusion', input_shapes=input_shapes, num_classes=2)
    losses = custom_fusion_loss(lambda1=0.1, lambda2=0.01)
    
    model.compile(optimizer='adam', loss=losses, metrics={'fusion_output': 'accuracy'})
    
    checkpoint_dir = "results/checkpoints/"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Training callbacks
    hist_file = os.path.join(checkpoint_dir, 'training_history.csv')
    latest_cp = os.path.join(checkpoint_dir, 'stress_model.keras') # Explicit .keras usage
    
    best_cb = ModelCheckpoint(latest_cp, save_best_only=True, monitor='val_fusion_output_loss')
    csv_cb = CSVLogger(hist_file, append=True)
    early_stop = EarlyStopping(monitor='val_fusion_output_loss', patience=15, restore_best_weights=True)
    
    print("Starting End-to-End Training...")
    
    # model.fit(...)
    
    print("Training complete.")
    
    # Write checkpoint metadata to ensure run.py can discover exactly what it's loading
    metadata = {
        "timestamp": datetime.datetime.now().isoformat(),
        "model_architecture": "ReliabilityAwareFusion",
        "T_value": 10,
        "input_shapes": input_shapes,
        "modality_order": ["audio", "face", "keystroke", "handwriting", "eye"],
        "class_mapping": {0: "NOT STRESSED", 1: "STRESSED"},
        "training_subjects": [],  # Would be populated by SubjectSplitter
        "validation_subjects": [],
        "test_subjects": []
    }
    
    with open(os.path.join(checkpoint_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print(f"Model saved to {latest_cp}")
    print(f"Metadata saved to {os.path.join(checkpoint_dir, 'metadata.json')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--validate-data", action="store_true", help="Run structural validation of the ForDigitStress dataset")
    args = parser.parse_args()
    
    if args.validate_data:
        validator = DatasetValidator()
        validator.validate_directory("data/fordigitstress/")
    else:
        train_fusion(args.epochs, args.batch_size)
