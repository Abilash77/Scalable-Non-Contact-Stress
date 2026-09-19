import tensorflow as tf
import numpy as np
import time
from src.DL_models import get_model

model = get_model('fusion')
test_in = {
    'input_audio': tf.zeros((1, 10, 169)),
    'input_face': tf.zeros((1, 10, 12)),
    'input_keystroke': tf.zeros((1, 10, 7)),
    'input_handwriting': tf.zeros((1, 10, 9)),
    'input_eye': tf.zeros((1, 10, 5)),
    'input_mask': tf.ones((1, 5))
}

@tf.function
def fast_inference(inputs):
    return model(inputs, training=False)

print("Warming up eager...")
for _ in range(5):
    _ = model(test_in, training=False)

eager_times = []
for _ in range(100):
    t0 = time.perf_counter()
    _ = model(test_in, training=False)
    eager_times.append((time.perf_counter() - t0) * 1000)
print(f"Eager Mean: {np.mean(eager_times):.2f} ms")

print("Warming up tf.function...")
for _ in range(5):
    _ = fast_inference(test_in)

func_times = []
for _ in range(100):
    t0 = time.perf_counter()
    _ = fast_inference(test_in)
    func_times.append((time.perf_counter() - t0) * 1000)
print(f"tf.function Mean: {np.mean(func_times):.2f} ms")
