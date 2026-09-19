import os
import sys
import time
import multiprocessing
import numpy as np
import tensorflow as tf
import cv2
import queue
import glob
import base64

from flask import Flask, jsonify, request, Response, render_template

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from DL_models import get_model
from audio_utils import extract_audio_features
from keystroke_utils import extract_keystroke_features
from face_utils import extract_face_features
from eye_utils import extract_eye_features
from handwriting_utils import extract_handwriting_features

try:
    import pyaudio
except ImportError:
    pyaudio = None
    
try:
    from pynput import keyboard
except ImportError:
    keyboard = None

def keyboard_worker(shared_state):
    if keyboard is None:
        shared_state['keyboard_status'] = "HARDWARE_UNAVAILABLE"
        return
        
    shared_state['keyboard_status'] = "ACTIVE"
    events = []
    total_event_count = 0
    
    def on_press(key):
        nonlocal total_event_count
        events.append({'key': str(key), 'action': 'press', 'time': time.time()})
        total_event_count += 1
    def on_release(key):
        nonlocal total_event_count
        events.append({'key': str(key), 'action': 'release', 'time': time.time()})
        total_event_count += 1
        
    try:
        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.start()
    except Exception as e:
        shared_state['keyboard_status'] = f"ERROR: {e}"
        return
    
    while shared_state['running']:
        time.sleep(0.1)
        if not shared_state.get('is_live_monitoring', False):
            continue
            
        current_time = time.time()
        recent_events = [e for e in events if current_time - e['time'] < 10.0]
        events[:] = recent_events
        
        t0 = time.perf_counter()
        feats = extract_keystroke_features(recent_events)
        lat = (time.perf_counter() - t0) * 1000
        
        shared_state['keystroke_features'] = feats.tolist()
        shared_state['keyboard_event_count'] = total_event_count
        latencies = shared_state['latency']
        latencies['keystroke'] = lat
        shared_state['latency'] = latencies

def audio_worker(shared_state):
    if pyaudio is None:
        shared_state['mic_status'] = "HARDWARE_UNAVAILABLE"
        return
        
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    RECORD_SECONDS = 3
    
    p = pyaudio.PyAudio()
    try:
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
        shared_state['mic_status'] = "ACTIVE"
    except Exception as e:
        shared_state['mic_status'] = f"HARDWARE_UNAVAILABLE"
        return
    
    sample_count = 0
    while shared_state['running']:
        time.sleep(0.1)
        if not shared_state.get('is_live_monitoring', False):
            continue
            
        frames = []
        for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            try:
                frames.append(stream.read(CHUNK, exception_on_overflow=False))
            except Exception:
                pass
        audio_data = np.frombuffer(b''.join(frames), dtype=np.int16).astype(np.float32)
        sample_count += len(audio_data)
        
        t0 = time.perf_counter()
        feats = extract_audio_features(audio_segment=audio_data, sr=RATE)
        lat = (time.perf_counter() - t0) * 1000
        
        shared_state['audio_features'] = feats.tolist()
        shared_state['audio_sample_count'] = sample_count
        latencies = shared_state['latency']
        latencies['audio'] = lat
        shared_state['latency'] = latencies

def webcam_worker(shared_state):
    cap = None
    camera_idx = -1
    for i in range(4):
        temp_cap = cv2.VideoCapture(i)
        if temp_cap.isOpened():
            ret, _ = temp_cap.read()
            if ret:
                cap = temp_cap
                camera_idx = i
                break
            temp_cap.release()
            
    if cap is None:
        shared_state['camera_connected'] = False
        shared_state['face_status'] = "HARDWARE_UNAVAILABLE"
        shared_state['eye_status'] = "HARDWARE_UNAVAILABLE"
        return
        
    shared_state['camera_connected'] = True
    shared_state['camera_index'] = camera_idx
    
    frame_count = 0
    face_detection_count = 0
    eye_detection_count = 0
    fps_start_time = time.time()
    
    while shared_state['running']:
        if not shared_state.get('is_live_monitoring', False):
            time.sleep(0.1)
            continue
            
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.033)
            continue
            
        frame_timestamp = time.time()
        shared_state['last_frame_timestamp'] = frame_timestamp
        h_orig, w_orig = frame.shape[:2]
        shared_state['camera_width'] = w_orig
        shared_state['camera_height'] = h_orig
            
        frame_count += 1
        shared_state['camera_frame_count'] = frame_count
        
        if frame_count % 30 == 0:
            now = time.time()
            fps = 30.0 / (now - fps_start_time + 1e-6)
            shared_state['camera_fps'] = round(fps, 1)
            fps_start_time = now
            
        if frame_count % 3 != 0:
            ret_enc, buffer = cv2.imencode('.jpg', frame)
            if ret_enc:
                shared_state['latest_frame_jpg'] = buffer.tobytes()
            continue
            
        # Scale for extraction performance
        frame_small = cv2.resize(frame, (320, 240))
        
        t0 = time.perf_counter()
        face_feats, face_status, face_meta = extract_face_features(frame_small)
        eye_feats, eye_status = extract_eye_features(frame_small)
        lat = (time.perf_counter() - t0) * 1000
        
        shared_state['face_status'] = face_status
        shared_state['eye_status'] = eye_status
        
        if face_status == "DETECTED":
            face_detection_count += 1
            shared_state['face_last_valid_time'] = time.time()
            
            bbox = face_meta.get('bbox', [0,0,0,0])
            scale_x = w_orig / 320.0
            scale_y = h_orig / 240.0
            
            x = int(bbox[0] * scale_x)
            y = int(bbox[1] * scale_y)
            w = int(bbox[2] * scale_x)
            h = int(bbox[3] * scale_y)
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "FACE DETECTED", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Faces: {face_meta.get('count', 1)}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            shared_state['face_bbox'] = [x, y, w, h]
            shared_state['face_count'] = face_meta.get('count', 1)
            shared_state['face_confidence'] = face_meta.get('confidence', 'N/A')
            shared_state['face_detected'] = True
        else:
            cv2.putText(frame, "NO FACE DETECTED", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            shared_state['face_detected'] = False
            shared_state['face_bbox'] = None
            shared_state['face_count'] = 0
            
        if eye_status == "DETECTED":
            eye_detection_count += 1
            shared_state['eye_last_valid_time'] = time.time()
            
        ret_enc, buffer = cv2.imencode('.jpg', frame)
        if ret_enc:
            shared_state['latest_frame_jpg'] = buffer.tobytes()
            
        shared_state['face_features'] = face_feats.tolist()
        shared_state['eye_features'] = eye_feats.tolist()
        shared_state['face_detection_count'] = face_detection_count
        shared_state['eye_detection_count'] = eye_detection_count
        
        latencies = shared_state['latency']
        latencies['face'] = lat
        latencies['eye'] = lat
        shared_state['latency'] = latencies

def handwriting_worker(shared_state):
    shared_state['handwriting_status'] = "WAITING_FOR_INPUT"
    submission_count = 0
    while shared_state['running']:
        time.sleep(0.1)
        if not shared_state.get('is_live_monitoring', False):
            continue
            
        b64_img = shared_state.get('live_handwriting_b64')
        if b64_img:
            t0 = time.perf_counter()
            try:
                img_data = base64.b64decode(b64_img)
                nparr = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
                
                feats = extract_handwriting_features(img_array=img)
                submission_count += 1
                shared_state['handwriting_status'] = "ACTIVE"
                shared_state['handwriting_features'] = feats.tolist()
                shared_state['handwriting_submission_count'] = submission_count
                shared_state['handwriting_last_valid_time'] = time.time()
            except Exception as e:
                shared_state['handwriting_status'] = f"ERROR: {e}"
                
            lat = (time.perf_counter() - t0) * 1000
            latencies = shared_state['latency']
            latencies['handwriting'] = lat
            shared_state['latency'] = latencies
            
            shared_state['live_handwriting_b64'] = None

def inference_worker(shared_state):
    import datetime
    import json as _json
    tf.config.set_visible_devices([], 'GPU')
    
    cp_dir = 'results/checkpoints'
    cp_swell = os.path.join(cp_dir, 'swell_kw_best.keras')
    cp_orig = os.path.join(cp_dir, 'stress_model.keras')
    meta_path = os.path.join(cp_dir, 'training_metadata.json')
    
    cp_file = None
    model_name = 'fusion'
    num_mods = 5
    feature_dim = 9
    is_trained = False
    
    # Validation
    if os.path.exists(meta_path):
        try:
            with open(meta_path, 'r') as f:
                meta = _json.load(f)
            
            if meta.get("is_trained"):
                expected_arch = meta.get("model_architecture")
                if expected_arch == "swell_fusion" and os.path.exists(cp_swell):
                    cp_file = cp_swell
                    model_name = "swell_fusion"
                    num_mods = 6
                    feature_dim = meta.get("expected_input_shapes", {}).get("swell", [10, 9])[1]
                    is_trained = True
                elif expected_arch == "fusion" and os.path.exists(cp_orig):
                    cp_file = cp_orig
                    model_name = "fusion"
                    num_mods = 5
                    is_trained = True
        except Exception as e:
            print(f"Metadata validation error: {e}")
            
    if not is_trained:
        print("[WARNING] No valid trained checkpoint found via metadata. System in ARCHITECTURE TEST ONLY mode.")
        shared_state['model_status'] = "UNTRAINED"
        shared_state['dataset_name'] = "NONE"
    else:
        shared_state['dataset_name'] = meta.get("dataset_name", "UNKNOWN")
    
    input_shapes = {
        'audio': (10, 169),
        'face': (10, 12),
        'keystroke': (10, 7),
        'handwriting': (10, 9),
        'eye': (10, 5)
    }
    
    if model_name == 'swell_fusion':
        input_shapes['swell'] = (10, feature_dim)

    # Measure model load time
    t_load_start = time.perf_counter()
    model = get_model(model_name, input_shapes=input_shapes, num_classes=2)
    
    if is_trained and cp_file:
        try:
            model.load_weights(cp_file)
            shared_state['model_status'] = "TRAINED"
            shared_state['checkpoint_name'] = os.path.basename(cp_file)
            print(f"Successfully loaded trained weights from: {cp_file}")
        except Exception as e:
            print(f"Failed to load trained weights: {e}")
            shared_state['model_status'] = "UNTRAINED"
            is_trained = False
            
    t_load_end = time.perf_counter()
    shared_state['model_load_time_ms'] = round((t_load_end - t_load_start) * 1000, 1)
    
    # Warmup
    print("[INFO] Warming up model...")
    dummy_inputs = {k: tf.zeros((1, *v)) for k, v in input_shapes.items()}
    dummy_inputs['input_mask'] = tf.zeros((1, num_mods))
    # Correct names for model inputs:
    dummy_inputs = {
        'input_audio': dummy_inputs['audio'],
        'input_face': dummy_inputs['face'],
        'input_keystroke': dummy_inputs['keystroke'],
        'input_handwriting': dummy_inputs['handwriting'],
        'input_eye': dummy_inputs['eye'],
        'input_mask': dummy_inputs['input_mask']
    }
    if model_name == 'swell_fusion':
         dummy_inputs['input_swell'] = tf.zeros((1, 10, feature_dim))
    model(dummy_inputs, training=False)
    print("[INFO] Model warmup complete.")
            
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
    
    # Per-modality buffer fill counts (how many REAL non-zero vectors have been pushed)
    buffer_fills = {'audio': 0, 'face': 0, 'keystroke': 0, 'handwriting': 0, 'eye': 0}
    
    # Sticky availability — track last time each modality produced valid (non-zero) features
    AVAILABILITY_TIMEOUT = 10.0  # seconds before declaring a modality unavailable
    last_valid_time = {'audio': 0.0, 'face': 0.0, 'keystroke': 0.0, 'handwriting': 0.0, 'eye': 0.0}
    
    # Temporal smoothing state — Exponential Moving Average (EMA)
    # NOTE: This is an engineering convenience for display stability,
    # NOT a component of the original paper's architecture.
    SMOOTHING_ALPHA = 0.3  # weight for the newest observation
    smoothed_stress_prob = 0.0
    
    # Prediction history ring buffer (last 20 windows)
    MAX_HISTORY = 20
    prediction_history = []
    
    modality_names_ordered = ['keyboard', 'speech', 'facial', 'eye_pupil', 'handwriting']
    
    os.makedirs('results', exist_ok=True)
    log_file_path = 'results/realtime_session.log'
    
    while shared_state['running']:
        time.sleep(0.1)
        if not shared_state.get('is_live_monitoring', False):
            continue
            
        t_start = time.perf_counter()
        now = time.time()
        
        shared_state['total_cycles'] += 1
        mask = np.zeros(num_mods, dtype=np.float32)
        
        aud_f = shared_state.get('audio_features')
        fac_f = shared_state.get('face_features')
        key_f = shared_state.get('keystroke_features')
        hw_f = shared_state.get('handwriting_features')
        eye_f = shared_state.get('eye_features')
        
        # --- Audio / Speech (mask index 0) ---
        if aud_f is not None:
            arr = np.array(aud_f)
            buffers['audio'] = np.roll(buffers['audio'], -1, axis=0)
            buffers['audio'][-1] = arr
            if np.count_nonzero(arr) > 0:
                last_valid_time['audio'] = now
                buffer_fills['audio'] = min(buffer_fills['audio'] + 1, T)
        if (now - last_valid_time['audio']) < AVAILABILITY_TIMEOUT and last_valid_time['audio'] > 0:
            mask[0] = 1.0
        
        # --- Facial (mask index 1) ---
        if fac_f is not None:
            arr = np.array(fac_f)
            buffers['face'] = np.roll(buffers['face'], -1, axis=0)
            buffers['face'][-1] = arr
            if np.count_nonzero(arr) > 0:
                last_valid_time['face'] = now
                buffer_fills['face'] = min(buffer_fills['face'] + 1, T)
        if (now - last_valid_time['face']) < AVAILABILITY_TIMEOUT and last_valid_time['face'] > 0:
            mask[1] = 1.0
        
        # --- Keyboard (mask index 2) ---
        # Keyboard is special: zero features from idle typing are legitimate data.
        # Available whenever the keyboard worker is alive and has provided features.
        if key_f is not None:
            arr = np.array(key_f)
            buffers['keystroke'] = np.roll(buffers['keystroke'], -1, axis=0)
            buffers['keystroke'][-1] = arr
            buffer_fills['keystroke'] = min(buffer_fills['keystroke'] + 1, T)
            last_valid_time['keystroke'] = now  # Always valid when worker is alive
        kbd_status = shared_state.get('keyboard_status', '')
        if kbd_status == 'ACTIVE' and key_f is not None:
            mask[2] = 1.0
        
        # --- Handwriting (mask index 3) ---
        if hw_f is not None:
            arr = np.array(hw_f)
            buffers['handwriting'] = np.roll(buffers['handwriting'], -1, axis=0)
            buffers['handwriting'][-1] = arr
            if np.count_nonzero(arr) > 0:
                last_valid_time['handwriting'] = now
                buffer_fills['handwriting'] = min(buffer_fills['handwriting'] + 1, T)
        if (now - last_valid_time['handwriting']) < AVAILABILITY_TIMEOUT and last_valid_time['handwriting'] > 0:
            mask[3] = 1.0
        
        # --- Eye/Pupil (mask index 4) ---
        if eye_f is not None:
            arr = np.array(eye_f)
            buffers['eye'] = np.roll(buffers['eye'], -1, axis=0)
            buffers['eye'][-1] = arr
            if np.count_nonzero(arr) > 0:
                last_valid_time['eye'] = now
                buffer_fills['eye'] = min(buffer_fills['eye'] + 1, T)
        if (now - last_valid_time['eye']) < AVAILABILITY_TIMEOUT and last_valid_time['eye'] > 0:
            mask[4] = 1.0
            
        inputs = {
            'input_audio': tf.convert_to_tensor(np.expand_dims(buffers['audio'], axis=0), dtype=tf.float32),
            'input_face': tf.convert_to_tensor(np.expand_dims(buffers['face'], axis=0), dtype=tf.float32),
            'input_keystroke': tf.convert_to_tensor(np.expand_dims(buffers['keystroke'], axis=0), dtype=tf.float32),
            'input_handwriting': tf.convert_to_tensor(np.expand_dims(buffers['handwriting'], axis=0), dtype=tf.float32),
            'input_eye': tf.convert_to_tensor(np.expand_dims(buffers['eye'], axis=0), dtype=tf.float32),
            'input_mask': tf.convert_to_tensor(np.expand_dims(mask, axis=0), dtype=tf.float32)
        }
        
        if model_name == 'swell_fusion':
            # Create a zero array for the SWELL features (since we have no real-time SWELL extractor)
            inputs['input_swell'] = tf.convert_to_tensor(np.zeros((1, T, feature_dim), dtype=np.float32))
        
        try:
            preds = fast_inference(inputs)
            
            raw_stress_prob = float(preds['fusion_output'].numpy()[0][1])
            raw_nonstress_prob = float(preds['fusion_output'].numpy()[0][0])
            reliabilities = preds['reliabilities'].numpy()[0].tolist()
            alphas = preds['alphas'].numpy()[0].tolist()
            shared_state['successful_cycles'] += 1
        except Exception as e:
            shared_state['failed_cycles'] += 1
            print(f"Inference error: {e}")
            continue
        
        t_end = time.perf_counter()
        lat_fusion = (t_end - t_start) * 1000
        
        # Temporal smoothing (EMA)
        smoothed_stress_prob = SMOOTHING_ALPHA * raw_stress_prob + (1.0 - SMOOTHING_ALPHA) * smoothed_stress_prob
        smoothed_nonstress_prob = 1.0 - smoothed_stress_prob
        
        pred_label = 1 if smoothed_stress_prob > 0.5 else 0
        confidence = smoothed_stress_prob if pred_label == 1 else smoothed_nonstress_prob
        
        now_ts = datetime.datetime.now().strftime("%H:%M:%S")
        
        # Build modality weights dict (order: audio/speech, face/facial, keystroke/keyboard, handwriting, eye/pupil)
        modality_weights = {
            'keyboard': round(alphas[2], 4),
            'speech': round(alphas[0], 4),
            'facial': round(alphas[1], 4),
            'eye_pupil': round(alphas[4], 4),
            'handwriting': round(alphas[3], 4)
        }
        
        # Determine unavailable modalities from mask
        unavailable = []
        mask_to_name = {0: 'speech', 1: 'facial', 2: 'keyboard', 3: 'handwriting', 4: 'eye_pupil'}
        for idx, name in mask_to_name.items():
            if mask[idx] < 0.5:
                unavailable.append(name)
        
        # Build prediction history entry
        history_entry = {
            'timestamp': now_ts,
            'prediction': 'STRESS' if pred_label == 1 else 'NON-STRESS',
            'stress_prob_pct': round(smoothed_stress_prob * 100, 1)
        }
        prediction_history.append(history_entry)
        if len(prediction_history) > MAX_HISTORY:
            prediction_history.pop(0)
        
        # Write all fields to shared state
        shared_state['fusion_prob'] = round(smoothed_stress_prob, 4)
        shared_state['fusion_pred'] = pred_label
        shared_state['raw_stress_prob'] = round(raw_stress_prob, 4)
        shared_state['raw_nonstress_prob'] = round(raw_nonstress_prob, 4)
        shared_state['smoothed_stress_prob'] = round(smoothed_stress_prob, 4)
        shared_state['smoothed_nonstress_prob'] = round(smoothed_nonstress_prob, 4)
        shared_state['confidence'] = round(confidence, 4)
        shared_state['modality_weights'] = _json.dumps(modality_weights)
        shared_state['reliabilities'] = _json.dumps({
            'keyboard': round(reliabilities[2], 4),
            'speech': round(reliabilities[0], 4),
            'facial': round(reliabilities[1], 4),
            'eye_pupil': round(reliabilities[4], 4),
            'handwriting': round(reliabilities[3], 4)
        })
        shared_state['unavailable_modalities'] = _json.dumps(unavailable)
        shared_state['prediction_timestamp'] = now_ts
        shared_state['prediction_history'] = _json.dumps(prediction_history)
        
        # Latency calculations
        latencies = shared_state.get('latency', {})
        active_lats = []
        for i, mod in enumerate(['audio', 'face', 'keystroke', 'handwriting', 'eye']):
            if mask[i] > 0.5 and mod in latencies:
                active_lats.append(latencies[mod])
        max_ext_ms = max(active_lats) if active_lats else 0.0
        
        shared_state['fusion_latency_ms'] = round(lat_fusion, 1)
        shared_state['max_extraction_ms'] = round(max_ext_ms, 1)
        shared_state['pipeline_latency_ms'] = round(lat_fusion + max_ext_ms, 1)
        
        shared_state['fusion_mask'] = _json.dumps(mask.tolist())
        shared_state['buffer_fills'] = _json.dumps({
            'audio': buffer_fills['audio'],
            'face': buffer_fills['face'],
            'keystroke': buffer_fills['keystroke'],
            'handwriting': buffer_fills['handwriting'],
            'eye': buffer_fills['eye']
        })
        
        latencies['fusion'] = lat_fusion
        latencies['pipeline'] = lat_fusion + max_ext_ms
        shared_state['latency'] = latencies
        
        # Real-time logging
        try:
            with open(log_file_path, 'a') as lf:
                log_line = f"{now_ts} | Mask: {mask.tolist()} | Prob: {smoothed_stress_prob:.4f} | Attn: {[round(a, 3) for a in alphas]} | Ext: {max_ext_ms:.1f}ms | Inf: {lat_fusion:.1f}ms | Total: {(lat_fusion + max_ext_ms):.1f}ms\n"
                lf.write(log_line)
        except Exception:
            pass

# FLASK SERVER
app = Flask(__name__)
manager_dict = {
    'modalities': {}, 'latency': {}, 'fusion_prob': 0.0, 'fusion_pred': 0,
    'model_status': 'LOADING', 'face_status': 'PENDING', 'eye_status': 'PENDING',
    'mic_status': 'PENDING', 'keyboard_status': 'PENDING', 'handwriting_status': 'WAITING_FOR_INPUT',
    'live_handwriting_b64': None, 'latest_frame_jpg': None
}

def init_app_state(shared_state):
    global manager_dict
    manager_dict = shared_state

import threading
local_cache = {}

def update_local_cache():
    import json as _json
    def _parse_json(key, default):
        raw = manager_dict.get(key)
        if raw and isinstance(raw, str):
            try:
                return _json.loads(raw)
            except Exception:
                return default
        return default
        
    while True:
        try:
            if not manager_dict.get('running', True):
                break
            lat = manager_dict.get('latency', {})
            new_cache = {
                'model_status': manager_dict.get('model_status', 'UNKNOWN'),
                'dataset_name': manager_dict.get('dataset_name', 'UNKNOWN'),
                'checkpoint_name': manager_dict.get('checkpoint_name', ''),
                'model_load_time_ms': manager_dict.get('model_load_time_ms', 0.0),
                'max_extraction_ms': manager_dict.get('max_extraction_ms', 0.0),
                'pipeline_latency_ms': manager_dict.get('pipeline_latency_ms', 0.0),
                'face_status': manager_dict.get('face_status', 'UNKNOWN'),
                'eye_status': manager_dict.get('eye_status', 'UNKNOWN'),
                'mic_status': manager_dict.get('mic_status', 'UNKNOWN'),
                'keyboard_status': manager_dict.get('keyboard_status', 'UNKNOWN'),
                'handwriting_status': manager_dict.get('handwriting_status', 'UNKNOWN'),
                'fusion_prob': manager_dict.get('fusion_prob', 0.0),
                'fusion_pred': manager_dict.get('fusion_pred', 0),
                'latency': dict(lat) if lat else {},
                'prediction': 'STRESS' if manager_dict.get('fusion_pred', 0) == 1 else 'NON-STRESS',
                'stress_probability': manager_dict.get('smoothed_stress_prob', 0.0),
                'non_stress_probability': manager_dict.get('smoothed_nonstress_prob', 1.0),
                'confidence': manager_dict.get('confidence', 0.0),
                'modality_weights': _parse_json('modality_weights', {}),
                'reliabilities': _parse_json('reliabilities', {}),
                'unavailable_modalities': _parse_json('unavailable_modalities', []),
                'prediction_timestamp': manager_dict.get('prediction_timestamp', ''),
                'prediction_history': _parse_json('prediction_history', []),
                'processing_latency_ms': manager_dict.get('fusion_latency_ms', 0.0),
                'keyboard_event_count': manager_dict.get('keyboard_event_count', 0),
                'audio_sample_count': manager_dict.get('audio_sample_count', 0),
                'camera_frame_count': manager_dict.get('camera_frame_count', 0),
                'face_detection_count': manager_dict.get('face_detection_count', 0),
                'eye_detection_count': manager_dict.get('eye_detection_count', 0),
                'handwriting_submission_count': manager_dict.get('handwriting_submission_count', 0),
                'fusion_mask': _parse_json('fusion_mask', [0,0,0,0,0]),
                'buffer_fills': _parse_json('buffer_fills', {}),
                'is_live_monitoring': manager_dict.get('is_live_monitoring', False),
                'session_start_time': manager_dict.get('session_start_time', 0.0),
                'total_cycles': manager_dict.get('total_cycles', 0),
                'successful_cycles': manager_dict.get('successful_cycles', 0),
                'failed_cycles': manager_dict.get('failed_cycles', 0),
                'dropped_cycles': manager_dict.get('dropped_cycles', 0),
                'app_start_time': manager_dict.get('app_start_time', 0.0),
                # Camera Diagnostics
                'camera_connected': manager_dict.get('camera_connected', False),
                'camera_index': manager_dict.get('camera_index', -1),
                'camera_fps': manager_dict.get('camera_fps', 0.0),
                'last_frame_timestamp': manager_dict.get('last_frame_timestamp', 0.0),
                'face_detected': manager_dict.get('face_detected', False),
                'face_count': manager_dict.get('face_count', 0),
                'face_confidence': manager_dict.get('face_confidence', 'N/A'),
                'face_bbox': manager_dict.get('face_bbox', None)
            }
            
            # Atomic update of the global cache
            global local_cache
            local_cache = new_cache
        except Exception as e:
            pass
        time.sleep(0.05)



@app.route('/status')
def status():
    return jsonify(local_cache)

@app.route('/camera/status')
def camera_status():
    import time
    last_frame_ts = local_cache.get('last_frame_timestamp', 0.0)
    frame_age_ms = (time.time() - last_frame_ts) * 1000 if last_frame_ts > 0 else None
    
    return jsonify({
        'camera_connected': local_cache.get('camera_connected', False),
        'camera_index': local_cache.get('camera_index', -1),
        'frame_age_ms': frame_age_ms,
        'fps': local_cache.get('camera_fps', 0.0),
        'face_detected': local_cache.get('face_detected', False),
        'face_count': local_cache.get('face_count', 0),
        'face_confidence': local_cache.get('face_confidence', 'N/A'),
        'face_bbox': local_cache.get('face_bbox', None),
        'last_frame_timestamp': last_frame_ts
    })

@app.route('/upload_handwriting', methods=['POST'])
def upload_handwriting():
    data = request.json
    if not data or 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400
    b64_str = data['image']
    if b64_str.startswith('data:image/png;base64,'):
        b64_str = b64_str.replace('data:image/png;base64,', '')
    manager_dict['live_handwriting_b64'] = b64_str
    return jsonify({'status': 'success'})

@app.route('/api/control', methods=['POST'])
def api_control():
    data = request.json
    action = data.get('action')
    if action == 'start':
        if not manager_dict['is_live_monitoring']:
            manager_dict['session_start_time'] = time.time()
            manager_dict['is_live_monitoring'] = True
    elif action == 'stop':
        manager_dict['is_live_monitoring'] = False
    elif action == 'reset':
        manager_dict['total_cycles'] = 0
        manager_dict['successful_cycles'] = 0
        manager_dict['failed_cycles'] = 0
        manager_dict['dropped_cycles'] = 0
        manager_dict['session_start_time'] = time.time() if manager_dict['is_live_monitoring'] else 0.0
    return jsonify({'status': 'success', 'is_live_monitoring': manager_dict['is_live_monitoring']})

def gen_frames():
    while manager_dict.get('running', True):
        frame_bytes = manager_dict.get('latest_frame_jpg')
        if frame_bytes is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.05)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    
    man = multiprocessing.Manager()
    main_shared_state = man.dict({
        'running': True,
        'audio_features': None,
        'face_features': None,
        'keystroke_features': None,
        'handwriting_features': None,
        'eye_features': None,
        'fusion_prob': 0.0,
        'fusion_pred': 0,
        'modalities': man.dict(),
        'latency': man.dict(),
        'model_status': 'LOADING',
        'checkpoint_name': '',
        'face_status': 'PENDING',
        'eye_status': 'PENDING',
        'mic_status': 'PENDING',
        'keyboard_status': 'PENDING',
        'handwriting_status': 'WAITING_FOR_INPUT',
        'live_handwriting_b64': None,
        'latest_frame_jpg': None,
        # Prediction output fields
        'raw_stress_prob': 0.0,
        'raw_nonstress_prob': 1.0,
        'smoothed_stress_prob': 0.0,
        'smoothed_nonstress_prob': 1.0,
        'confidence': 0.0,
        'modality_weights': '{}',
        'reliabilities': '{}',
        'unavailable_modalities': '[]',
        'prediction_timestamp': '',
        'prediction_history': '[]',
        'fusion_latency_ms': 0.0,
        'fusion_mask': '[0,0,0,0,0]',
        'buffer_fills': '{}',
        # Diagnostic counters
        'keyboard_event_count': 0,
        'audio_sample_count': 0,
        'camera_frame_count': 0,
        'face_detection_count': 0,
        'eye_detection_count': 0,
        'handwriting_submission_count': 0,
        'face_last_valid_time': 0.0,
        'eye_last_valid_time': 0.0,
        'handwriting_last_valid_time': 0.0,
        # Real-time session state
        'is_live_monitoring': False,
        'session_start_time': 0.0,
        'total_cycles': 0,
        'successful_cycles': 0,
        'failed_cycles': 0,
        'dropped_cycles': 0,
        'app_start_time': time.time()
    })
    
    manager_dict = main_shared_state
    init_app_state(main_shared_state)
    
    workers = [
        multiprocessing.Process(target=keyboard_worker, args=(main_shared_state,), daemon=True),
        multiprocessing.Process(target=audio_worker, args=(main_shared_state,), daemon=True),
        multiprocessing.Process(target=webcam_worker, args=(main_shared_state,), daemon=True),
        multiprocessing.Process(target=handwriting_worker, args=(main_shared_state,), daemon=True),
        multiprocessing.Process(target=inference_worker, args=(main_shared_state,), daemon=True)
    ]
    
    for w in workers:
        w.start()
        
    print("========================================")
    print("RA-HMSD REAL-TIME MULTIMODAL SERVER")
    print("========================================")
    print("Dashboard URL: http://localhost:5000")
    print("API Endpoint:  http://localhost:5000/status")
    print("========================================\n")
    
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    
    try:
        cache_thread = threading.Thread(target=update_local_cache, daemon=True)
        cache_thread.start()
        app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
    except KeyboardInterrupt:
        pass
    finally:
        main_shared_state['running'] = False
        for w in workers:
            w.terminate()
            w.join()

