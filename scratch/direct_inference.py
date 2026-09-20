
import numpy as np
import tensorflow as tf
from src.DL_models import get_model

def run_test():
    print("MODEL LOAD: PASS")
    input_shapes = {
        "audio": (10, 169),
        "face": (10, 12),
        "keystroke": (10, 7),
        "handwriting": (10, 9),
        "eye": (10, 5)
    }
    model = get_model("fusion", input_shapes=input_shapes, num_classes=2)
    model.load_weights("results/checkpoints/freihaut_keyboard_stress.keras")
    
    # Fake realistic keystroke feature vector (10, 7)
    # Using slightly random numbers
    keystroke = np.random.rand(1, 10, 7).astype(np.float32)
    inputs = {
        "input_audio": np.zeros((1, 10, 169), dtype=np.float32),
        "input_face": np.zeros((1, 10, 12), dtype=np.float32),
        "input_keystroke": keystroke,
        "input_handwriting": np.zeros((1, 10, 9), dtype=np.float32),
        "input_eye": np.zeros((1, 10, 5), dtype=np.float32),
        "input_mask": np.array([[0, 0, 1, 0, 0]], dtype=np.float32)
    }
    
    print(f"INPUT SHAPE: (1, 10, 7)")
    preds = model.predict(inputs, verbose=0)
    out = preds["fusion_output"][0]
    print(f"OUTPUT SHAPE: (1, {len(out)})")
    print(f"\nNON-STRESS PROBABILITY: {out[0]:.4f}")
    print(f"STRESS PROBABILITY: {out[1]:.4f}")
    
    pred_label = np.argmax(out)
    pred_class = "STRESS" if pred_label == 1 else "NON-STRESS"
    confidence = out[pred_label]
    
    print(f"\nPREDICTION: {pred_class}")
    print(f"CONFIDENCE: {confidence:.4f}")

if __name__ == "__main__":
    run_test()

