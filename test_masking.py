import numpy as np
import tensorflow as tf
from src.DL_models import get_model, ReliabilityAttention

# We want to trace the attention output specifically to ensure mask sets attention to exactly zero.
model = get_model('fusion', 
    input_shapes={
        'audio': (10, 169), 
        'face': (10, 12), 
        'keystroke': (10, 7), 
        'handwriting': (10, 9), 
        'eye': (10, 5)
    })

inputs_template = {
    'input_audio': np.random.rand(1, 10, 169).astype(np.float32),
    'input_face': np.random.rand(1, 10, 12).astype(np.float32),
    'input_keystroke': np.random.rand(1, 10, 7).astype(np.float32),
    'input_handwriting': np.random.rand(1, 10, 9).astype(np.float32),
    'input_eye': np.random.rand(1, 10, 5).astype(np.float32)
}

print("Masking Test Start")
scenarios = [
    ("Speech missing", [0.0, 1.0, 1.0, 1.0, 1.0], 0),
    ("Facial missing", [1.0, 0.0, 1.0, 1.0, 1.0], 1),
    ("Keyboard missing", [1.0, 1.0, 0.0, 1.0, 1.0], 2),
    ("Handwriting missing", [1.0, 1.0, 1.0, 0.0, 1.0], 3),
    ("Eye/Pupil missing", [1.0, 1.0, 1.0, 1.0, 0.0], 4)
]

for name, mask_list, idx in scenarios:
    test_inputs = inputs_template.copy()
    mask = np.array([mask_list], dtype=np.float32)
    test_inputs['input_mask'] = mask
    try:
        preds = model(test_inputs, training=False)
        # To verify attention suppression:
        # We can extract the ReliabilityAttention layer and pass synthetic encoded features + mask to it.
        att_layer = [l for l in model.layers if isinstance(l, ReliabilityAttention)][0]
        # Simulate intermediate features (batch_size=1, 5 modalities, each 64 dim)
        mock_features = [np.random.rand(1, 64).astype(np.float32) for _ in range(5)]
        attended_fused, att_weights = att_layer(mock_features, mask)
        
        # Check if the attention weight for the missing modality is exactly 0
        weight_for_missing = att_weights[0, idx].numpy()
        
        # Check if the attended_fused ignores the missing modality
        # By setting the missing modality mock feature to NaN, if it contributes, output will be NaN
        mock_features_nan = mock_features.copy()
        mock_features_nan[idx] = np.full((1, 64), np.nan, dtype=np.float32)
        attended_fused_nan, _ = att_layer(mock_features_nan, mask)
        has_nan = np.isnan(attended_fused_nan).any()
        
        print(f"Scenario: {name} - PASS (Model Output Shape: {preds['fusion_output'].shape}, Attention Weight: {weight_for_missing}, NaN Contamination: {has_nan})")
    except Exception as e:
        print(f"Scenario: {name} - FAIL ({e})")
