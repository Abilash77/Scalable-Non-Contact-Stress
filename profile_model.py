import tensorflow as tf
import numpy as np
import cProfile
import pstats
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

# Warmup to trace
for _ in range(5):
    _ = model(test_in, training=False)

def run_inference():
    for _ in range(10):
        _ = model(test_in, training=False)

print("Starting cProfile...")
cProfile.run('run_inference()', 'profile.stats')
p = pstats.Stats('profile.stats')
p.strip_dirs().sort_stats('cumtime').print_stats(30)
