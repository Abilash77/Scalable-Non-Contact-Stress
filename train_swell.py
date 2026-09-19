import os
import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from DL_models import get_model, custom_fusion_loss
from scripts.preprocessing.preprocess_swell import preprocess_swell

def generate_evaluation_report(y_true, y_pred_prob, output_dir, latency_ms):
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    try:
        auroc = roc_auc_score(y_true, y_pred_prob[:, 1])
    except ValueError:
        auroc = 0.5
        
    cm = confusion_matrix(y_true, y_pred)
    
    # Save Confusion Matrix
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('SWELL-KW Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Non-Stress', 'Stress'])
    plt.yticks(tick_marks, ['Non-Stress', 'Stress'])
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    # Add text
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()
    
    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auroc": auroc,
        "latency_ms": latency_ms
    }
    
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
        
    return metrics

import hashlib

def compute_cache_hash():
    # Hash the preprocessing script to detect code/version changes
    hasher = hashlib.md5()
    try:
        with open(os.path.join(os.path.dirname(__file__), 'scripts', 'preprocessing', 'preprocess_swell.py'), 'rb') as f:
            hasher.update(f.read())
    except FileNotFoundError:
        pass
    # We could also hash raw dataset files here, but for simplicity we rely on the script hash 
    # and a config version bump if needed.
    return hasher.hexdigest()

def train():
    processed_dir = "data/processed/swell_kw"
    output_dir = "results/swell_kw"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("SWELL-KW FAST TRAINING PIPELINE")
    print("="*60 + "\n")
    
    # 1. Dataset loading & Cache Validation
    cache_config_path = os.path.join(processed_dir, "cache_config.json")
    current_hash = compute_cache_hash()
    
    cache_valid = False
    if os.path.exists(cache_config_path) and os.path.exists(os.path.join(processed_dir, "X_train.npy")):
        try:
            with open(cache_config_path, 'r') as f:
                cache_config = json.load(f)
            if cache_config.get("hash") == current_hash:
                cache_valid = True
        except Exception:
            pass

    if not cache_valid:
        print("[INFO] Cache invalid or missing. Running preprocessor...")
        success = preprocess_swell()
        if not success:
            print("[BLOCKED] Training aborted due to missing SWELL-KW dataset.")
            return
        # Save new cache config
        os.makedirs(processed_dir, exist_ok=True)
        with open(cache_config_path, 'w') as f:
            json.dump({"hash": current_hash}, f)
            
    print("[INFO] Loading cached dataset...")
    X_train = np.load(os.path.join(processed_dir, "X_train.npy"))
    y_train = np.load(os.path.join(processed_dir, "y_train.npy"))
    X_val = np.load(os.path.join(processed_dir, "X_val.npy"))
    y_val = np.load(os.path.join(processed_dir, "y_val.npy"))
    X_test = np.load(os.path.join(processed_dir, "X_test.npy"))
    y_test = np.load(os.path.join(processed_dir, "y_test.npy"))
    
    with open(os.path.join(processed_dir, "dataset_metadata.json"), 'r') as f:
        meta = json.load(f)
        
    feature_dim = meta["feature_dim"]
    
    print(f"[INFO] Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # 2. tf.data Pipeline (Fast CPU/GPU loading)
    # We map the SWELL generic feature to the 'swell' branch, and zero-fill the rest.
    def make_dataset(X, y):
        batch_size = len(X)
        N = len(X)
        T = 10
        # Zero arrays for untrained modalities
        z_audio = np.zeros((N, T, 169), dtype=np.float32)
        z_face = np.zeros((N, T, 12), dtype=np.float32)
        z_key = np.zeros((N, T, 7), dtype=np.float32)
        z_hw = np.zeros((N, T, 9), dtype=np.float32)
        z_eye = np.zeros((N, T, 5), dtype=np.float32)
        
        # Mask: Only SWELL (index 5) is 1.0. Indices 0-4 are 0.0
        mask = np.zeros((N, 6), dtype=np.float32)
        mask[:, 5] = 1.0 
        
        ds = tf.data.Dataset.from_tensor_slices((
            {
                'input_audio': z_audio,
                'input_face': z_face,
                'input_keystroke': z_key,
                'input_handwriting': z_hw,
                'input_eye': z_eye,
                'input_swell': X,
                'input_mask': mask
            },
            y
        ))
        return ds.batch(32).prefetch(tf.data.AUTOTUNE)

    train_ds = make_dataset(X_train, y_train)
    val_ds = make_dataset(X_val, y_val)
    test_ds = make_dataset(X_test, y_test)
    
    # 3. Model Architecture
    input_shapes = {
        'audio': (10, 169),
        'face': (10, 12),
        'keystroke': (10, 7),
        'handwriting': (10, 9),
        'eye': (10, 5),
        'swell': (10, feature_dim)
    }
    
    print("[INFO] Instantiating RA_HMSD_Fusion_Model (SWELL Adapter)...")
    model = get_model('swell_fusion', input_shapes=input_shapes, num_classes=2)
    
    loss_fn = custom_fusion_loss(lambda1=0.1, lambda2=0.01)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                  loss=loss_fn, 
                  metrics={'fusion_output': 'accuracy'})
                  
    # 4. Training Callbacks
    checkpoint_path = "results/checkpoints/swell_kw_best.keras"
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_fusion_output_loss', patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_fusion_output_loss', factor=0.5, patience=5),
        tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_fusion_output_loss', save_best_only=True)
    ]
    
    print("[INFO] Starting fast training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=50,
        callbacks=callbacks,
        verbose=1
    )
    
    # 5. Evaluation & Latency
    print("[INFO] Training complete. Evaluating on test set...")
    
    # Latency test
    sample_batch = next(iter(test_ds.take(1)))[0]
    # warmup
    model(sample_batch, training=False)
    
    t0 = time.perf_counter()
    y_pred_dict = model.predict(test_ds)
    t1 = time.perf_counter()
    
    # Convert total batch prediction time to per-window
    total_samples = len(X_test)
    latency_ms = ((t1 - t0) * 1000) / total_samples
    
    metrics = generate_evaluation_report(y_test, y_pred_dict['fusion_output'], output_dir, latency_ms)
    
    # Save Training Metadata
    training_meta = {
        "dataset_name": "SWELL-KW",
        "model_architecture": "swell_fusion",
        "expected_input_shapes": {
            "audio": [10, 169],
            "face": [10, 12],
            "keystroke": [10, 7],
            "handwriting": [10, 9],
            "eye": [10, 5],
            "swell": [10, feature_dim]
        },
        "expected_output_shape": [2],
        "is_trained": True,
        "training_timestamp": time.time(),
        "metrics": metrics
    }
    with open(os.path.join(os.path.dirname(checkpoint_path), "training_metadata.json"), 'w') as f:
        json.dump(training_meta, f, indent=4)
        
    # 6. SWELL_KW_RESULTS.md (Honest comparison)
    report_path = os.path.join(output_dir, "SWELL_KW_RESULTS.md")
    with open(report_path, "w") as f:
        f.write("# SWELL-KW Evaluation Report\n\n")
        f.write("## Dataset Status\n")
        f.write("- **Present**: YES\n")
        f.write(f"- **Train Samples**: {meta['train_samples']}\n")
        f.write(f"- **Validation Samples**: {meta['val_samples']}\n")
        f.write(f"- **Test Samples**: {meta['test_samples']}\n")
        f.write(f"- **Subject-wise split**: {'YES' if meta['subject_split'] else 'NO (Data leakage risk exists)'}\n\n")
        
        f.write("## Model & Checkpoint Status\n")
        f.write("- **Model**: SWELL Adapter Fusion (RA-HMSD)\n")
        f.write("- **Checkpoint**: `results/checkpoints/swell_kw_best.keras`\n\n")
        
        f.write("## Metrics: Our Evaluation vs Paper\n")
        f.write("| Metric | Paper Target | Our Result |\n")
        f.write("|---|---|---|\n")
        f.write(f"| **Accuracy** | 94.30% | {metrics['accuracy']*100:.2f}% |\n")
        f.write(f"| **Precision** | 94.00% | {metrics['precision']*100:.2f}% |\n")
        f.write(f"| **Recall** | 93.20% | {metrics['recall']*100:.2f}% |\n")
        f.write(f"| **F1 Score** | 93.60% | {metrics['f1']*100:.2f}% |\n")
        f.write(f"| **AUROC** | 0.967 | {metrics['auroc']:.3f} |\n")
        f.write(f"| **Latency** | 38.40 ms/window | {metrics['latency_ms']:.2f} ms/window |\n\n")
        
        f.write("## Limitations\n")
        f.write("- The evaluated checkpoint relies solely on SWELL-KW features.\n")
        f.write("- Real-time application with this specific checkpoint will mask out Keystrokes, Audio, Face, etc., unless those modalities were included in the SWELL-KW download.\n")
        f.write("- Honest evaluation: the discrepancy between the Paper Target and Our Result reflects the real performance on the specific dataset subset provided, without fabricated enhancements.\n")
        
    print(f"\n[SUCCESS] Honest evaluation complete. Report saved to {report_path}")

if __name__ == "__main__":
    train()
