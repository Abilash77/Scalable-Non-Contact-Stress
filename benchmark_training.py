import os
import sys
import time
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from DL_models import get_model, custom_fusion_loss

def benchmark():
    print("="*50)
    print("RA-HMSD ARCHITECTURE BENCHMARK")
    print("="*50)

    gpus = tf.config.list_physical_devices('GPU')
    print(f"Hardware: {'GPU' if gpus else 'CPU'}")

    input_shapes = {
        'audio': (10, 169),
        'face': (10, 12),
        'keystroke': (10, 7),
        'handwriting': (10, 9),
        'eye': (10, 5)
    }
    
    batch_size = 32
    print("\n1. Measuring Model Initialization...")
    start_t = time.time()
    model = get_model('fusion', input_shapes=input_shapes, num_classes=2)
    losses = custom_fusion_loss(lambda1=0.1, lambda2=0.01)
    
    # Filter losses to only match model outputs to avoid KeyError
    model_output_names = [out.name.split('/')[0] if hasattr(out, 'name') else out for out in model.outputs]
    filtered_losses = {k: v for k, v in losses.items() if k in model.output_names}
    
    model.compile(optimizer='adam', loss=filtered_losses)
    init_time = time.time() - start_t
    print(f"   Initialization Time: {init_time*1000:.2f} ms")

    # Generate dummy data
    def generate_dummy_data(num_samples):
        X = {
            'input_audio': np.random.rand(num_samples, 10, 169).astype(np.float32),
            'input_face': np.random.rand(num_samples, 10, 12).astype(np.float32),
            'input_keystroke': np.random.rand(num_samples, 10, 7).astype(np.float32),
            'input_handwriting': np.random.rand(num_samples, 10, 9).astype(np.float32),
            'input_eye': np.random.rand(num_samples, 10, 5).astype(np.float32),
            'input_mask': np.ones((num_samples, 5)).astype(np.float32),
        }
        y = {
            'fusion_output': np.random.randint(0, 2, size=(num_samples, 1)).astype(np.float32),
            'unimodal_audio': np.random.randint(0, 2, size=(num_samples, 1)).astype(np.float32),
            'unimodal_face': np.random.randint(0, 2, size=(num_samples, 1)).astype(np.float32),
            'unimodal_keystroke': np.random.randint(0, 2, size=(num_samples, 1)).astype(np.float32),
            'unimodal_handwriting': np.random.randint(0, 2, size=(num_samples, 1)).astype(np.float32),
            'unimodal_eye': np.random.randint(0, 2, size=(num_samples, 1)).astype(np.float32),
        }
        return X, y

    # Warmup
    X_warm, y_warm = generate_dummy_data(batch_size)
    model.train_on_batch(X_warm, y_warm)

    print("\n2. Measuring One Training Batch...")
    start_t = time.time()
    model.train_on_batch(X_warm, y_warm)
    one_batch_time = time.time() - start_t
    print(f"   One Batch (size {batch_size}): {one_batch_time*1000:.2f} ms")

    print("\n3. Measuring 10 Training Batches...")
    start_t = time.time()
    for _ in range(10):
        model.train_on_batch(X_warm, y_warm)
    ten_batch_time = time.time() - start_t
    print(f"   10 Batches Time: {ten_batch_time*1000:.2f} ms")
    avg_batch_time = ten_batch_time / 10
    print(f"   Average Batch Time: {avg_batch_time*1000:.2f} ms")

    print("\n4. Measuring Validation Batch...")
    start_t = time.time()
    model.test_on_batch(X_warm, y_warm)
    val_batch_time = time.time() - start_t
    print(f"   Validation Batch (size {batch_size}): {val_batch_time*1000:.2f} ms")

    print("\n5. Measuring Inference Batch (Real-time Simulation)...")
    X_inf = {k: v[:1] for k, v in X_warm.items()} # Batch size 1
    # Warmup inference
    model.predict(X_inf, verbose=0)
    
    start_t = time.time()
    for _ in range(50):
        model.predict(X_inf, verbose=0)
    inf_time = (time.time() - start_t) / 50
    print(f"   Inference Time (batch_size=1): {inf_time*1000:.2f} ms")

    # Time Estimation
    print("\n" + "="*50)
    print("ESTIMATED TOTAL TRAINING TIME")
    print("="*50)
    
    # Assumptions for ForDigitStress (12 subjects, ~30 mins per subject @ 1Hz = 21,600 samples)
    # Split: Train (~15k), Val (~3k), Test (~3k)
    assumed_train_samples = 15000
    epochs = 100
    steps_per_epoch = assumed_train_samples // batch_size
    
    time_per_epoch = steps_per_epoch * avg_batch_time
    total_train_time = time_per_epoch * epochs
    
    print(f"Assumed Train Samples: {assumed_train_samples}")
    print(f"Batch Size: {batch_size}")
    print(f"Steps per Epoch: {steps_per_epoch}")
    print(f"Epochs Configured: {epochs}")
    print("-" * 30)
    print(f"Estimated Time per Epoch: {time_per_epoch:.2f} seconds")
    print(f"Estimated Total Training Time: {total_train_time / 60:.2f} minutes")
    print("="*50)

if __name__ == '__main__':
    benchmark()
