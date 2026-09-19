import time
import multiprocessing
import numpy as np
import tensorflow as tf
from src.DL_models import get_model

def mock_inference_cadence_test():
    tf.config.set_visible_devices([], 'GPU')
    model = get_model('fusion')
    
    @tf.function(reduce_retracing=True)
    def fast_inference(inputs):
        return model(inputs, training=False)
    
    T = 10
    buffers = {
        'audio': np.zeros((T, 169), dtype=np.float32),
        'face': np.zeros((T, 12), dtype=np.float32),
        'keystroke': np.zeros((T, 7), dtype=np.float32),
        'handwriting': np.zeros((T, 9), dtype=np.float32),
        'eye': np.zeros((T, 5), dtype=np.float32)
    }
    
    print("Warming up model...")
    mask = np.ones((1,5), dtype=np.float32)
    inputs = {
        'input_audio': tf.convert_to_tensor(np.expand_dims(buffers['audio'], axis=0)),
        'input_face': tf.convert_to_tensor(np.expand_dims(buffers['face'], axis=0)),
        'input_keystroke': tf.convert_to_tensor(np.expand_dims(buffers['keystroke'], axis=0)),
        'input_handwriting': tf.convert_to_tensor(np.expand_dims(buffers['handwriting'], axis=0)),
        'input_eye': tf.convert_to_tensor(np.expand_dims(buffers['eye'], axis=0)),
        'input_mask': tf.convert_to_tensor(mask)
    }
    for _ in range(3):
        fast_inference(inputs)
        
    print("Starting cadence test...")
    target_interval = 0.5  # 500 ms
    num_cycles = 100
    
    cycle_intervals = []
    inference_durations = []
    missed_cycles = 0
    
    last_start_time = time.perf_counter()
    next_wake_time = time.perf_counter() + target_interval
    
    for i in range(num_cycles):
        now = time.perf_counter()
        
        # Enforce target cadence
        sleep_time = next_wake_time - now
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            missed_cycles += 1
            # If we fell behind, schedule from *now* to prevent backlog death spiral?
            # Or schedule strictly from the previous wake time (strict cadence)?
            # run.py uses `time.sleep(0.5)` which is a relative wait. We will simulate strict relative wake.
            pass
            
        cycle_start = time.perf_counter()
        cycle_intervals.append((cycle_start - last_start_time) * 1000.0)
        last_start_time = cycle_start
        
        # Inference step
        t0 = time.perf_counter()
        out = fast_inference(inputs)
        t1 = time.perf_counter()
        
        inference_durations.append((t1 - t0) * 1000.0)
        
        next_wake_time = cycle_start + target_interval
        
    cycle_intervals = cycle_intervals[1:] # drop first one
    
    print("\n--- CADENCE TEST RESULTS ---")
    print(f"Target cadence: {target_interval * 1000:.1f} ms")
    print(f"Actual mean cycle interval: {np.mean(cycle_intervals):.2f} ms")
    print(f"Actual p95 cycle interval: {np.percentile(cycle_intervals, 95):.2f} ms")
    print(f"Max cycle interval: {np.max(cycle_intervals):.2f} ms")
    print(f"Inference mean: {np.mean(inference_durations):.2f} ms")
    print(f"Missed cycles: {missed_cycles}")
    
    backlog = np.sum([max(0, x - 500.0) for x in cycle_intervals])
    print(f"Backlog: {backlog:.2f} ms")

if __name__ == '__main__':
    mock_inference_cadence_test()
