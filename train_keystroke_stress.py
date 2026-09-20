import os
import sys
import json
import time
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from DL_models import get_model, custom_fusion_loss

def train():
    processed_dir = "data/processed/keystroke_stress"
    output_dir = "results/keystroke_stress"
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("KEYSTROKE-STRESS TRAINING PIPELINE")
    print("="*60)
    
    X_train = np.load(os.path.join(processed_dir, "X_train.npy"))
    y_train = np.load(os.path.join(processed_dir, "y_train.npy"))
    X_val = np.load(os.path.join(processed_dir, "X_val.npy"))
    y_val = np.load(os.path.join(processed_dir, "y_val.npy"))
    
    print(f"[INFO] Train: {X_train.shape}, Val: {X_val.shape}")
    
    def make_dataset(X, y):
        N = len(X)
        T = 10
        # Untrained modalities (zeros)
        z_audio = np.zeros((N, T, 169), dtype=np.float32)
        z_face = np.zeros((N, T, 12), dtype=np.float32)
        z_hw = np.zeros((N, T, 9), dtype=np.float32)
        z_eye = np.zeros((N, T, 5), dtype=np.float32)
        
        # We DO NOT feed keystroke into physiology, we feed it into KEYSTROKE.
        # The mask for Keystroke is index 2.
        mask = np.zeros((N, 5), dtype=np.float32)
        mask[:, 2] = 1.0 
        
        y_dict = {
            'fusion_output': y,
            'unimodal_audio': y,
            'unimodal_face': y,
            'unimodal_keystroke': y,
            'unimodal_handwriting': y,
            'unimodal_eye': y,
            'reliabilities': np.zeros((N, 5), dtype=np.float32),
            'alphas': np.zeros((N, 5), dtype=np.float32)
        }
        
        ds = tf.data.Dataset.from_tensor_slices((
            {
                'input_audio': z_audio,
                'input_face': z_face,
                'input_keystroke': X,
                'input_handwriting': z_hw,
                'input_eye': z_eye,
                'input_mask': mask
            },
            y_dict
        ))
        return ds.batch(2).prefetch(tf.data.AUTOTUNE)

    train_ds = make_dataset(X_train, y_train)
    val_ds = make_dataset(X_val, y_val)
    
    input_shapes = {
        'audio': (10, 169),
        'face': (10, 12),
        'keystroke': (10, 7),
        'handwriting': (10, 9),
        'eye': (10, 5)
    }
    
    print("[INFO] Instantiating RA_HMSD_Fusion_Model (Standard)...")
    model = get_model('fusion', input_shapes=input_shapes, num_classes=2)
    loss_fn = custom_fusion_loss(lambda1=0.1, lambda2=0.01)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                  loss=loss_fn, 
                  metrics={'fusion_output': 'accuracy'})
                  
    checkpoint_path = "results/checkpoints/keystroke_stress.keras"
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_fusion_output_loss', save_best_only=True, mode='min')
    ]
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
        callbacks=callbacks,
        verbose=1
    )
    
    print("[INFO] Training complete. Saving metadata...")
    
    # Save Training Metadata
    training_meta = {
        "dataset_name": "Keystroke-Stress",
        "model_architecture": "fusion",
        "expected_input_shapes": input_shapes,
        "expected_output_shape": [2],
        "is_trained": True,
        "training_timestamp": time.time(),
        "trained_modalities": ["keystroke"],  # Only keystroke is trained!
        "modality_order": ["audio", "face", "keystroke", "handwriting", "eye"],
        "class_mapping": {"0": "NON-STRESS", "1": "STRESS"}
    }
    with open(os.path.join(os.path.dirname(checkpoint_path), "training_metadata.json"), 'w') as f:
        json.dump(training_meta, f, indent=4)
        
    print(f"[SUCCESS] Checkpoint saved to {checkpoint_path}")

if __name__ == "__main__":
    train()
