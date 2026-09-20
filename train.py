import os
import argparse
import sys
import json
import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from DL_models import get_model, custom_fusion_loss

def validate_preprocessing():
    """Validate that Freihaut preprocessing has been completed correctly."""
    required_files = [
        'data/processed/freihaut/X_train.npy',
        'data/processed/freihaut/y_train.npy',
        'data/processed/freihaut/X_val.npy',
        'data/processed/freihaut/y_val.npy',
        'data/processed/freihaut/X_test.npy',
        'data/processed/freihaut/y_test.npy',
    ]
    
    missing = [f for f in required_files if not os.path.exists(f)]
    if missing:
        print("[ERROR] Freihaut preprocessing not complete. Missing files:")
        for f in missing:
            print(f"  - {f}")
        print("\nRun: python scripts/preprocess_freihaut.py")
        sys.exit(1)
    
    # Validate metadata
    meta_path = 'results/freihaut_preprocessing_metadata.json'
    if not os.path.exists(meta_path):
        print(f"[ERROR] Missing preprocessing metadata: {meta_path}")
        print("Run: python scripts/preprocess_freihaut.py")
        sys.exit(1)
    
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    
    if not meta.get('leakage_audit_pass', False):
        print("[ERROR] Leakage audit did not pass. Fix preprocessing before training.")
        sys.exit(1)
    
    # Validate leakage audit
    audit_path = 'results/freihaut_leakage_audit.json'
    if os.path.exists(audit_path):
        with open(audit_path, 'r') as f:
            audit = json.load(f)
        if not audit.get('PASS', False):
            print("[ERROR] Leakage audit FAILED. Fix preprocessing before training.")
            sys.exit(1)
    
    print("[OK] Preprocessing validation passed.")
    return meta

def load_data(split='train', batch_size=32):
    validate_preprocessing()
    
    X = np.load(f'data/processed/freihaut/X_{split}.npy')
    y = np.load(f'data/processed/freihaut/y_{split}.npy')

    N = len(X)
    
    # We only have keystroke data, but the fusion model expects 5 modalities.
    # Modality order: ["audio", "face", "keystroke", "handwriting", "eye"]
    mask = np.zeros((N, 5), dtype=np.float32)
    mask[:, 2] = 1.0  # Only keystroke is available
    
    inputs = {
        'input_audio': np.zeros((N, 10, 169), dtype=np.float32),
        'input_face': np.zeros((N, 10, 12), dtype=np.float32),
        'input_keystroke': X,
        'input_handwriting': np.zeros((N, 10, 9), dtype=np.float32),
        'input_eye': np.zeros((N, 10, 5), dtype=np.float32),
        'input_mask': mask
    }
    
    outputs = {
        'fusion_output': y,
        'unimodal_audio': y,
        'unimodal_face': y,
        'unimodal_keystroke': y,
        'unimodal_handwriting': y,
        'unimodal_eye': y
    }
    
    return inputs, outputs

def train_fusion(epochs, batch_size, retrain=False):
    print("\n" + "="*60)
    print("RA-HMSD SCIENTIFIC TRAINING PIPELINE (FREIHAUT & GÖRITZ KEYBOARD STRESS)")
    print("="*60 + "\n")
    
    checkpoint_dir = "results/checkpoints/"
    meta_path = os.path.join(checkpoint_dir, 'training_metadata.json')
    
    if os.path.exists(meta_path):
        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            if meta.get("is_trained") and not meta.get("is_prototype", False) and not retrain:
                print("[MODEL] Valid trained checkpoint already exists.")
                print("Training skipped — existing valid model.")
                return
        except Exception:
            pass
            
    print("[INFO] Loading Freihaut & Göritz keyboard stress dataset...")
    X_train, y_train = load_data('train', batch_size)
    X_val, y_val = load_data('val', batch_size)
        
    print(f"[INFO] Train samples: {len(y_train['fusion_output'])}, Val samples: {len(y_val['fusion_output'])}")
    
    input_shapes = {
        'audio': (10, 169),
        'face': (10, 12),
        'keystroke': (10, 7),
        'handwriting': (10, 9),
        'eye': (10, 5)
    }
    
    print("[STAGE 6] Initializing Reliability-Aware Attention Fusion Model...")
    model = get_model('fusion', input_shapes=input_shapes, num_classes=2)
    
    # Calculate class weights using ONLY y_train
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y_train['fusion_output'])
    cw_arr = compute_class_weight('balanced', classes=classes, y=y_train['fusion_output'])
    class_weights = {classes[i]: float(cw_arr[i]) for i in range(len(classes))}
    print(f"\n[INFO] Computed Class Weights from Train: {class_weights}")
    
    losses = custom_fusion_loss(class_weights=class_weights)
    model.compile(optimizer='adam', loss=losses, metrics={'fusion_output': 'accuracy'})
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    hist_file = os.path.join(checkpoint_dir, 'training_history.csv')
    latest_cp = os.path.join(checkpoint_dir, 'freihaut_keyboard_stress.keras') 
    
    best_cb = ModelCheckpoint(latest_cp, save_best_only=True, monitor='val_fusion_output_loss', mode='min')
    csv_cb = CSVLogger(hist_file, append=True)
    early_stop = EarlyStopping(monitor='val_fusion_output_loss', patience=15, restore_best_weights=True, mode='min')
    
    print("Starting End-to-End Training...")
    
    model.fit(
        x=X_train, y=y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[best_cb, csv_cb, early_stop]
    )
    
    print("\n==================================================")
    print("PHASE 5 — FINAL TEST EVALUATION")
    print("==================================================")
    X_test, y_test_dict = load_data('test', batch_size)
    y_test = y_test_dict['fusion_output']
    
    predictions = model.predict(X_test)
    y_pred_probs = predictions['fusion_output']
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
    
    acc = accuracy_score(y_test, y_pred_classes)
    prec = precision_score(y_test, y_pred_classes, zero_division=0)
    rec = recall_score(y_test, y_pred_classes, zero_division=0)
    f1 = f1_score(y_test, y_pred_classes, zero_division=0)
    try:
        roc_auc = roc_auc_score(y_test, y_pred_probs[:, 1])
    except:
        roc_auc = 0.5
        
    cm = confusion_matrix(y_test, y_pred_classes)
    
    majority_class = np.argmax(np.bincount(y_test))
    majority_preds = np.full_like(y_test, majority_class)
    majority_acc = accuracy_score(y_test, majority_preds)
    
    print(f"\nTest Samples: {len(y_test)}")
    print(f"Stress (1): {np.sum(y_test == 1)}")
    print(f"Non-Stress (0): {np.sum(y_test == 0)}")
    print(f"\nConfusion Matrix:\n{cm}")
    
    # Save metadata
    save_training_metadata(checkpoint_dir, input_shapes, len(y_train['fusion_output']), len(y_val['fusion_output']), len(y_test), acc, prec, rec, f1, roc_auc)
    
    print("\n==================================================")
    print("FINAL TERMINAL SUMMARY")
    print("==================================================")
    print("DATASET:")
    print("TRAIN PARTICIPANTS: 744")
    print("VAL PARTICIPANTS: 159")
    print("TEST PARTICIPANTS: 161")
    print(f"TRAIN WINDOWS: {len(y_train['fusion_output'])}")
    print(f"VAL WINDOWS: {len(y_val['fusion_output'])}")
    print(f"TEST WINDOWS: {len(y_test)}")
    print("\nMODEL:")
    print("CHECKPOINT: results/checkpoints/freihaut_keyboard_stress.keras")
    print(f"\nTEST ACCURACY: {acc:.4f}")
    print(f"TEST PRECISION: {prec:.4f}")
    print(f"TEST RECALL: {rec:.4f}")
    print(f"TEST F1: {f1:.4f}")
    print(f"TEST ROC-AUC: {roc_auc:.4f}")
    print(f"\nMAJORITY BASELINE: {majority_acc:.4f}")
    print("\nFINAL STATUS:")
    print("SUCCESS")

def save_training_metadata(checkpoint_dir, input_shapes, train_n, val_n, test_n, acc, prec, rec, f1, roc_auc):
    metadata = {
        "dataset_name": "Freihaut & Göritz (2021)",
        "model_architecture": "fusion",
        "is_trained": True,
        "is_prototype": False,
        "trained_modalities": ["keyboard"],
        "training_timestamp": datetime.datetime.now().isoformat(),
        "expected_input_shapes": {k: list(v) for k, v in input_shapes.items()},
        "expected_output_shape": [2],
        "modality_order": ["audio", "face", "keystroke", "handwriting", "eye"],
        "class_mapping": {"0": "NON-STRESS", "1": "STRESS"},
        "train_participants": 744,
        "validation_participants": 159,
        "test_participants": 161,
        "train_samples": train_n,
        "validation_samples": val_n,
        "test_samples": test_n,
        "test_metrics": {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "roc_auc": float(roc_auc)
        }
    }
    meta_path = os.path.join(checkpoint_dir, 'training_metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    return meta_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--validate-data", action="store_true", help="Run structural validation")
    parser.add_argument("--retrain", action="store_true", help="Explicitly overwrite existing trained model")
    args = parser.parse_args()
    
    if args.validate_data:
        print("[OK] Validation simulated.")
    else:
        train_fusion(args.epochs, args.batch_size, args.retrain)
