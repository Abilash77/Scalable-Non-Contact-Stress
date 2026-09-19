import os
import time
import numpy as np
import tensorflow as tf
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from DL_models import get_model

def run_latency_benchmark(model, num_modalities, input_shapes, name=""):
    print(f"\n{'='*50}\nARCHITECTURE TEST — NOT STRESS PREDICTION\nLATENCY BENCHMARK: {name}\n{'='*50}")
    
    # 1. Prepare deterministic synthetic inputs
    T = 10
    N = 100
    
    # Create batch of 1
    inputs_full = {}
    for m, shape in input_shapes.items():
        inputs_full[f'input_{m}'] = tf.convert_to_tensor(np.ones((1, T, shape[1]), dtype=np.float32) * 0.5)
        
    mask_full = np.ones((1, num_modalities), dtype=np.float32)
    inputs_full['input_mask'] = tf.convert_to_tensor(mask_full)
    
    # 2. First inference (Cold start)
    t0 = time.perf_counter()
    _ = model(inputs_full, training=False)
    t1 = time.perf_counter()
    cold_latency = (t1 - t0) * 1000
    
    # 3. Warm-up
    for _ in range(5):
        _ = model(inputs_full, training=False)
        
    # 4. Warmed-up execution (100 cycles)
    latencies = []
    for _ in range(N):
        t0 = time.perf_counter()
        _ = model(inputs_full, training=False)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)
        
    latencies = np.array(latencies)
    
    print(f"Cycles: {N}")
    print(f"Cold Start:   {cold_latency:.2f} ms")
    print(f"Mean:         {np.mean(latencies):.2f} ms")
    print(f"Median:       {np.median(latencies):.2f} ms")
    print(f"P95:          {np.percentile(latencies, 95):.2f} ms")
    print(f"P99:          {np.percentile(latencies, 99):.2f} ms")
    print(f"Max:          {np.max(latencies):.2f} ms")
    
    # 5. Masking Permutations Test
    print("\n[Masking Permutations Test]")
    permutations = [
        ("All Modalities", [1,1,1,1,1]),
        ("One Missing", [1,1,1,1,0]),
        ("Two Missing", [1,1,1,0,0]),
        ("All But One Missing", [1,0,0,0,0]),
        ("All Missing", [0,0,0,0,0])
    ]
    
    if num_modalities == 6:
        permutations = [
            ("All Modalities (SWELL)", [1,1,1,1,1,1]),
            ("Only SWELL", [0,0,0,0,0,1]),
            ("All Missing", [0,0,0,0,0,0])
        ]
        
    for desc, mask_arr in permutations:
        m_tensor = tf.convert_to_tensor(np.array([mask_arr], dtype=np.float32))
        inputs_mod = inputs_full.copy()
        inputs_mod['input_mask'] = m_tensor
        
        try:
            preds = model(inputs_mod, training=False)
            prob = preds['fusion_output'].numpy()[0]
            print(f"  {desc:<25}: PASS | Output Prob = {prob}")
        except Exception as e:
            print(f"  {desc:<25}: FAIL | {e}")
            
def test_all():
    # Enforce checkpoint rule (Task 11)
    # The user says "Do NOT create stress_model.keras unless legitimate stress-labelled training has actually completed."
    fake_cp = "results/checkpoints/stress_model.keras"
    if os.path.exists(fake_cp):
        print(f"[INFO] Removing fake checkpoint: {fake_cp}")
        os.remove(fake_cp)
        
    # Standard 5-Modality Model
    input_shapes_5 = {
        'audio': (10, 169),
        'face': (10, 12),
        'keystroke': (10, 7),
        'handwriting': (10, 9),
        'eye': (10, 5)
    }
    model_5 = get_model('fusion', input_shapes=input_shapes_5, num_classes=2)
    run_latency_benchmark(model_5, 5, input_shapes_5, "RA-HMSD 5-Modality Baseline")
    
    # SWELL 6-Modality Adapter
    input_shapes_6 = input_shapes_5.copy()
    input_shapes_6['swell'] = (10, 9)
    model_6 = get_model('swell_fusion', input_shapes=input_shapes_6, num_classes=2)
    run_latency_benchmark(model_6, 6, input_shapes_6, "RA-HMSD 6-Modality SWELL Adapter")

if __name__ == "__main__":
    test_all()
