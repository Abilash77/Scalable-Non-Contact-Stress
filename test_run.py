import sys
import os
import json
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run
import threading

def run_test(camera_status, keyboard_status):
    shared_state = {
        'running': True,
        'total_cycles': 0,
        'successful_cycles': 0,
        'failed_cycles': 0,
        'is_live_monitoring': True,
        'latency': {},
        'keyboard_status': keyboard_status,
        'camera_connected': camera_status == 'ACTIVE',
        'face_status': 'DETECTED' if camera_status == 'ACTIVE' else 'HARDWARE_UNAVAILABLE',
        'trained_modalities': '["keyboard"]',
        'keystroke_features': [0]*7, # mock some data so it registers as active
        'face_features': [0]*12 if camera_status == 'ACTIVE' else None
    }

    thread = threading.Thread(target=run.inference_worker, args=(shared_state,))
    thread.start()

    time.sleep(10)
    shared_state['running'] = False
    thread.join()
    
    print(f"\n--- TEST: Camera {camera_status}, Keyboard {keyboard_status} ---")
    print("Model Status:", shared_state.get('model_status'))
    print("Dataset Name:", shared_state.get('dataset_name'))
    print("Trained Modalities:", shared_state.get('trained_modalities'))
    print("Fusion Mask:", shared_state.get('fusion_mask'))
    print("Predictions Available:", shared_state.get('predictions_available'))
    print("Successful Cycles:", shared_state.get('successful_cycles'))

print("Running Live Tests...")
run_test(camera_status="HARDWARE_UNAVAILABLE", keyboard_status="ACTIVE")
run_test(camera_status="ACTIVE", keyboard_status="ACTIVE")
