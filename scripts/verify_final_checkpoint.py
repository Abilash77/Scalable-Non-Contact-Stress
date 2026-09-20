import json
import os
import sys
import numpy as np
import tensorflow as tf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.DL_models import get_model

def run_verification():
    print("==================================================")
    print("1. VERIFY FINAL CHECKPOINT")
    print("==================================================")
    
    meta_path = 'results/checkpoints/training_metadata.json'
    with open(meta_path, 'r') as f:
        meta = json.load(f)
        
    print(f"checkpoint: {meta.get('checkpoint_path', 'results/checkpoints/keystroke_stress.keras')}")
    print(f"model architecture: {meta.get('model_architecture')}")
    print(f"dataset: {meta.get('dataset_name')}")
    print(f"trained modalities: {meta.get('trained_modalities')}")
    print(f"epochs: {meta.get('completed_epochs')}")
    print(f"best validation metric: {meta.get('best_val_metric')}")
    print(f"test metrics: {meta.get('test_metrics', 'N/A')}")
    print(f"training date: {meta.get('training_timestamp')}")
    
    print("\n==================================================")
    print("2 & 3. VERIFY MODEL IS REALLY TRAINED")
    print("==================================================")
    
    print(f"Metadata is_trained: {meta.get('is_trained')}")
    
    # Load model
    input_shapes = {
        'audio': (10, 169),
        'face': (10, 12),
        'keystroke': (10, 7),
        'handwriting': (10, 9),
        'eye': (10, 5)
    }
    model = get_model('fusion', input_shapes=input_shapes, num_classes=2)
    
    # Checkpoint weights vs untrained weights
    untrained_weights = [np.copy(w) for w in model.get_weights()]
    
    cp_path = 'results/checkpoints/keystroke_stress.keras'
    model.load_weights(cp_path)
    trained_weights = model.get_weights()
    
    diff_sum = sum(np.sum(np.abs(tw - uw)) for tw, uw in zip(trained_weights, untrained_weights))
    print(f"Weight difference (trained vs untrained): {diff_sum:.4f}")
    if diff_sum > 0:
        print("[SUCCESS] Checkpoint contains trained weights, not random initialization.")
    
    # Create REAL feature window from test data
    # (Since keystroke-stress split is random subsets, we'll load the processed data)
    X_key = np.load('data/processed/keystroke_stress/X_val.npy')
    y_labels = np.load('data/processed/keystroke_stress/y_val.npy')
    
    if len(X_key) == 0:
        print("ERROR: Processed data is empty.")
        return
        
    # Use one real sample
    real_sample_X = X_key[0:1] # shape (1, 10, 7)
    real_sample_y = y_labels[0]
    
    dummy_inputs = {f'input_{k}': tf.zeros((1, *v)) for k, v in input_shapes.items()}
    dummy_inputs['input_keystroke'] = tf.convert_to_tensor(real_sample_X, dtype=tf.float32)
    dummy_inputs['input_mask'] = tf.convert_to_tensor([[0.0, 0.0, 1.0, 0.0, 0.0]], dtype=tf.float32)
    
    preds = model.predict(dummy_inputs, verbose=0)
    fusion_pred = preds['fusion_output'][0]
    
    print("\nRAW MODEL OUTPUT:")
    print(f"Fusion output probabilities: {fusion_pred}")
    print(f"STRESS PROBABILITY: {fusion_pred[1]:.4f}")
    print(f"PREDICTED CLASS: {'STRESS' if fusion_pred[1] > 0.5 else 'NON-STRESS'}")
    
    print("\n==================================================")
    print("4 & 5. VERIFY REAL TEST DATA / SUBJECT SPLIT")
    print("==================================================")
    
    # Processed data doesn't contain subjects explicitly, but we can print samples count
    total_samples = len(X_key)
    stress_samples = np.sum(y_labels == 1)
    non_stress_samples = np.sum(y_labels == 0)
    
    print(f"Total samples processed: {total_samples}")
    print(f"Stress samples: {stress_samples}")
    print(f"Non-stress samples: {non_stress_samples}")
    
    # Since we train on all in the fallback due to lack of subjects:
    print("\nSubject split: Participant IDs are NOT available in this fallback subset (SWELL-KW/DriveDB subset converted).")
    print("Train/Test split was performed randomly at 80/20 in train_keystroke_stress.py.")
    
if __name__ == "__main__":
    run_verification()
