import numpy as np
import tensorflow as tf
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from DL_models import get_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import json

def run_audit():
    print("="*60)
    print("SCIENTIFIC AUDIT: KEYSTROKE STRESS CHECKPOINT")
    print("="*60)
    
    checkpoint_path = 'results/checkpoints/keystroke_stress.keras'
    model = get_model('fusion', input_shapes={'audio':(10,169),'face':(10,12),'keystroke':(10,7),'handwriting':(10,9),'eye':(10,5)}, num_classes=2)
    model.load_weights(checkpoint_path)
    
    X_val = np.load('data/processed/keystroke_stress/X_val.npy')
    y_val = np.load('data/processed/keystroke_stress/y_val.npy')
    
    N = len(X_val)
    z_audio = np.zeros((N, 10, 169), dtype=np.float32)
    z_face = np.zeros((N, 10, 12), dtype=np.float32)
    z_hw = np.zeros((N, 10, 9), dtype=np.float32)
    z_eye = np.zeros((N, 10, 5), dtype=np.float32)
    mask = np.zeros((N, 5), dtype=np.float32)
    mask[:, 2] = 1.0
    
    inputs = {
        'input_audio': z_audio,
        'input_face': z_face,
        'input_keystroke': X_val,
        'input_handwriting': z_hw,
        'input_eye': z_eye,
        'input_mask': mask
    }
    
    print("\n--- HELD-OUT TEST EVALUATION ---")
    preds = model.predict(inputs, verbose=0)
    fusion_probs = preds['fusion_output']
    y_pred = np.argmax(fusion_probs, axis=1)
    
    acc = accuracy_score(y_val, y_pred)
    try:
        roc = roc_auc_score(y_val, fusion_probs[:, 1])
    except:
        roc = "N/A"
    
    print(f"Test Sample Count: {N}")
    print(f"Accuracy: {acc:.4f}")
    
    print("\n--- PERTURBATION TEST ---")
    sample_idx = 0
    orig_X = X_val[sample_idx:sample_idx+1]
    orig_prob = fusion_probs[sample_idx, 1]
    
    pert_X = orig_X.copy()
    pert_X *= 0.0 # Perturb by zeroing out the keystrokes
    inputs['input_keystroke'] = pert_X
    pert_preds = model.predict({k: v[sample_idx:sample_idx+1] for k, v in inputs.items()}, verbose=0)
    pert_prob = pert_preds['fusion_output'][0, 1]
    
    print(f"True Label: {y_val[sample_idx]}")
    print(f"Original Keystroke Input Prob (Stress): {orig_prob:.4f}")
    print(f"Perturbed (Zeroed) Keystroke Input Prob (Stress): {pert_prob:.4f}")
    print(f"Absolute Prob Shift: {abs(orig_prob - pert_prob):.4f}")

if __name__ == '__main__':
    run_audit()
