import os
import sys
import tensorflow as tf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from DL_models import get_model

def verify():
    print("==================================================")
    print("CHECKPOINT VALIDATION")
    print("==================================================")
    cp_file = 'results/checkpoints/stress_model.keras'
    
    if not os.path.exists(cp_file):
        print("LIVE CHECKPOINT: NOT FOUND")
        print("CHECKPOINT LOAD: NOT TESTABLE")
        print("LIVE MODEL CHECKPOINT NOT AVAILABLE — TRAINING REQUIRED")
        return
        
    print("LIVE CHECKPOINT: FOUND")
    
    input_shapes = {
        'audio': (10, 169),
        'face': (10, 12),
        'keystroke': (10, 7),
        'handwriting': (10, 9),
        'eye': (10, 5)
    }
    
    try:
        model = get_model('fusion', input_shapes=input_shapes, num_classes=2)
        model.load_weights(cp_file)
        print("CHECKPOINT LOAD: PASS")
        print("Architecture: T=10, 5-modality fusion model.")
    except Exception as e:
        print("CHECKPOINT LOAD: FAIL")
        print(f"Error: {e}")

if __name__ == '__main__':
    verify()
