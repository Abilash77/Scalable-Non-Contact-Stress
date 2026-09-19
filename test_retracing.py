import os
import time
import numpy as np
import tensorflow as tf
from src.DL_models import get_model

def run_retracing_test():
    tf.config.set_visible_devices([], 'GPU')
    model = get_model('fusion')
    
    @tf.function(reduce_retracing=True)
    def fast_inference(inputs):
        return model(inputs, training=False)
        
    T = 10
    
    def make_inputs(mask_val):
        return {
            'input_audio': tf.convert_to_tensor(np.zeros((1, T, 169), dtype=np.float32)),
            'input_face': tf.convert_to_tensor(np.zeros((1, T, 12), dtype=np.float32)),
            'input_keystroke': tf.convert_to_tensor(np.zeros((1, T, 7), dtype=np.float32)),
            'input_handwriting': tf.convert_to_tensor(np.zeros((1, T, 9), dtype=np.float32)),
            'input_eye': tf.convert_to_tensor(np.zeros((1, T, 5), dtype=np.float32)),
            'input_mask': tf.convert_to_tensor(np.array([mask_val], dtype=np.float32))
        }

    cases = {
        'All available': [1.0, 1.0, 1.0, 1.0, 1.0],
        'Speech missing': [0.0, 1.0, 1.0, 1.0, 1.0],
        'Facial missing': [1.0, 0.0, 1.0, 1.0, 1.0],
        'Keyboard missing': [1.0, 1.0, 0.0, 1.0, 1.0],
        'Handwriting missing': [1.0, 1.0, 1.0, 0.0, 1.0],
        'Eye missing': [1.0, 1.0, 1.0, 1.0, 0.0],
        'All available again': [1.0, 1.0, 1.0, 1.0, 1.0]
    }
    
    # Pre-flight call to force initial tracing
    print("Initial tracing call...")
    _ = fast_inference(make_inputs([1.0]*5))
    
    print("\n--- RETRACING GATE TEST ---")
    
    for case_name, mask_val in cases.items():
        inputs = make_inputs(mask_val)
        
        # Verify shapes and dtypes
        print(f"\nCase: {case_name}")
        for k, v in inputs.items():
            print(f"  {k} shape={v.shape} dtype={v.dtype}")
            
        trace_before = fast_inference._get_tracing_count()
        
        # 10 calls
        for _ in range(10):
            _ = fast_inference(inputs)
            
        trace_after = fast_inference._get_tracing_count()
        retraced = "YES" if trace_after > trace_before else "NO"
        
        print(f"  Trace Before: {trace_before} | Trace After: {trace_after} | Retraced: {retraced}")
        
    print("\n--- LATENCY BENCHMARK ---")
    for case_name, mask_val in cases.items():
        inputs = make_inputs(mask_val)
        
        # warmup
        for _ in range(5):
            _ = fast_inference(inputs)
            
        lats = []
        for _ in range(100):
            t0 = time.perf_counter()
            _ = fast_inference(inputs)
            lats.append((time.perf_counter() - t0) * 1000)
            
        print(f"{case_name:22} | Mean: {np.mean(lats):.2f} ms | Median: {np.median(lats):.2f} ms | P95: {np.percentile(lats, 95):.2f} ms | Min: {np.min(lats):.2f} ms | Max: {np.max(lats):.2f} ms")

if __name__ == '__main__':
    run_retracing_test()
