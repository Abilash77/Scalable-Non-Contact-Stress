import os
import sys
import json
import tensorflow as tf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from DL_models import get_model

INPUT_SHAPES = {
    'audio': (10, 169),
    'face': (10, 12),
    'keystroke': (10, 7),
    'handwriting': (10, 9),
    'eye': (10, 5),
}


def verify():
    print("==================================================")
    print("CHECKPOINT VALIDATION")
    print("==================================================")

    cp_dir = 'results/checkpoints'
    cp_file = os.path.join(cp_dir, 'stress_model.keras')
    meta_file = os.path.join(cp_dir, 'training_metadata.json')

    if not os.path.exists(cp_file):
        print("LIVE CHECKPOINT: NOT FOUND")
        print("CHECKPOINT LOAD: NOT TESTABLE")
        print("LIVE MODEL CHECKPOINT NOT AVAILABLE — TRAINING REQUIRED")
        return False

    print("LIVE CHECKPOINT: FOUND")

    if not os.path.exists(meta_file):
        print("METADATA: NOT FOUND")
        print("CHECKPOINT LOAD: FAIL (missing training_metadata.json)")
        return False

    try:
        with open(meta_file, 'r') as f:
            meta = json.load(f)
    except Exception as e:
        print("METADATA: INVALID")
        print(f"CHECKPOINT LOAD: FAIL ({e})")
        return False

    if not meta.get('is_trained'):
        print("METADATA: is_trained=false")
        print("CHECKPOINT LOAD: FAIL (checkpoint marked untrained)")
        return False

    expected_arch = meta.get('model_architecture')
    if expected_arch not in ('fusion', 'swell_fusion'):
        print(f"METADATA: unsupported architecture '{expected_arch}'")
        print("CHECKPOINT LOAD: FAIL")
        return False

    model_name = expected_arch
    input_shapes = dict(INPUT_SHAPES)
    if model_name == 'swell_fusion':
        swell_dim = meta.get('expected_input_shapes', {}).get('swell', [10, 9])[1]
        input_shapes['swell'] = (10, swell_dim)

    try:
        model = get_model(model_name, input_shapes=input_shapes, num_classes=2)
        model.load_weights(cp_file)
        dummy = {f"input_{k}": tf.zeros((1, *v)) for k, v in input_shapes.items()}
        dummy['input_mask'] = tf.ones((1, len(input_shapes)))
        out = model(dummy, training=False)
        assert out['fusion_output'].shape == (1, 2)
        print("CHECKPOINT LOAD: PASS")
        print(f"Architecture: {model_name}, T=10, {len(input_shapes)}-modality fusion model.")
        print(f"Dataset: {meta.get('dataset_name', 'UNKNOWN')}")
        return True
    except Exception as e:
        print("CHECKPOINT LOAD: FAIL")
        print(f"Error: {e}")
        return False


if __name__ == '__main__':
    verify()
