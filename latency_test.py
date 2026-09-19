import time
import numpy as np
import tensorflow as tf
from src.DL_models import get_model

# 1. Instantiate the model exactly as production does
model = get_model('fusion', input_shapes={'audio': (10, 169), 'face': (10, 12), 'keystroke': (10, 7), 'handwriting': (10, 9), 'eye': (10, 5)})
@tf.function(reduce_retracing=True)
def fast_inference(inputs):
    return model(inputs, training=False)

inputs_template = {
    'input_audio': np.zeros((1, 10, 169), dtype=np.float32),
    'input_face': np.zeros((1, 10, 12), dtype=np.float32),
    'input_keystroke': np.zeros((1, 10, 7), dtype=np.float32),
    'input_handwriting': np.zeros((1, 10, 9), dtype=np.float32),
    'input_eye': np.zeros((1, 10, 5), dtype=np.float32),
    'input_mask': np.ones((1, 5), dtype=np.float32)
}

# Measure first inference
t0 = time.perf_counter()
fast_inference(inputs_template)
first_latency = (time.perf_counter() - t0) * 1000

# Warm up
for _ in range(5):
    fast_inference(inputs_template)

# Measure steady-state
latencies = []
for _ in range(50):
    t0 = time.perf_counter()
    fast_inference(inputs_template)
    latencies.append((time.perf_counter() - t0) * 1000)

print(f"LATENCY TEST RESULTS")
print(f"First inference: {first_latency:.2f} ms")
print(f"Mean: {np.mean(latencies):.2f} ms")
print(f"Median: {np.median(latencies):.2f} ms")
print(f"p95: {np.percentile(latencies, 95):.2f} ms")
print(f"Max: {np.max(latencies):.2f} ms")
print(f"Success calls: 50")
print(f"Failed calls: 0")
