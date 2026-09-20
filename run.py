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

def webcam_worker(shared_state):
    cap = None
    camera_idx = -1
    for i in range(5):
        temp_cap = cv2.VideoCapture(i)
        if temp_cap.isOpened():
            ret, frame = temp_cap.read()
            if ret and frame is not None and frame.size > 0:
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
    
    try:
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

                bbox = face_meta.get('bbox', [0, 0, 0, 0])
                scale_x = w_orig / 320.0
                scale_y = h_orig / 240.0

                x = int(bbox[0] * scale_x)
                y = int(bbox[1] * scale_y)
                w = int(bbox[2] * scale_x)
                h = int(bbox[3] * scale_y)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
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
    finally:
        if cap is not None:
            cap.release()

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
                img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
                
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
            
            cp_freihaut = os.path.join(cp_dir, 'freihaut_keyboard_stress.keras')
            if meta.get("is_trained") and not meta.get("is_prototype", False) and os.path.exists(cp_freihaut):
                expected_arch = meta.get("model_architecture")
                if expected_arch == "fusion":
                    cp_file = cp_freihaut
                    model_name = "fusion"
                    num_mods = 5
                    is_trained = True
                    shared_state['trained_modalities'] = _json.dumps(meta.get('trained_modalities', []))
            else:
                is_trained = False
        except Exception as e:
            print(f"Metadata validation error: {e}")
            is_trained = False
            
    if not is_trained:
        print("\n" + "="*50)
        print("MODEL STATUS: READY FOR TRAINING")
        print("DATA STATUS: FREIHAUT & GÖRITZ KEYBOARD STRESS")
        print("MODE: ARCHITECTURE TEST ONLY")
        print("MESSAGE: \"No valid trained stress model found. Run `python train.py` after preprocessing Freihaut data.\"")
        print("="*50 + "\n")
        shared_state['model_status'] = "READY FOR TRAINING"
        shared_state['dataset_name'] = "FREIHAUT & GÖRITZ KEYBOARD STRESS"
    else:
        shared_state['dataset_name'] = meta.get("dataset_name", "UNKNOWN")
        print(f"\n[MODEL] Valid trained checkpoint found")
        print(f"[MODEL] Dataset: {shared_state['dataset_name']}")
        print(f"[MODEL] Training date: {meta.get('training_timestamp', 'UNKNOWN')}")
        print(f"[MODEL] Loading checkpoint...")
    
    input_shapes = {
        'audio': (10, 169),
        'face': (10, 12),
        'keystroke': (10, 7),
        'handwriting': (10, 9),
        'eye': (10, 5)
    }

    # Measure model load time
    t_load_start = time.perf_counter()
    model = get_model(model_name, input_shapes=input_shapes, num_classes=2)
    
    if is_trained and cp_file:
        try:
            model.load_weights(cp_file)
            shared_state['model_status'] = "TRAINED"
            shared_state['checkpoint_name'] = os.path.basename(cp_file)
            print(f"[MODEL] Checkpoint loaded successfully")
            print(f"[MODEL] Training skipped — existing valid model")
            print(f"[MODEL] Real-time inference starting...")
        except Exception as e:
            print(f"Failed to load trained weights: {e}")
            shared_state['model_status'] = "READY FOR TRAINING"
            shared_state['dataset_name'] = "FREIHAUT & GÖRITZ KEYBOARD STRESS"
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

    # Rolling latency samples for percentile reporting (last 100 cycles)
    LATENCY_WINDOW = 100
    feat_lat_samples = []
    inf_lat_samples = []
    total_lat_samples = []
    cycle_timestamps = []
    
    os.makedirs('results', exist_ok=True)
    log_file_path = 'results/realtime_session.log'
    
    while shared_state['running']:
        time.sleep(0.1)
        if not shared_state.get('is_live_monitoring', False):
            continue
            
        t_start = time.perf_counter()
        now = time.time()
        
        shared_state['total_cycles'] += 1
        ui_mask = np.zeros(5, dtype=np.float32)
        
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
            ui_mask[0] = 1.0
        
        # --- Facial (mask index 1) ---
        if fac_f is not None:
            arr = np.array(fac_f)
            buffers['face'] = np.roll(buffers['face'], -1, axis=0)
            buffers['face'][-1] = arr
            if np.count_nonzero(arr) > 0:
                last_valid_time['face'] = now
                buffer_fills['face'] = min(buffer_fills['face'] + 1, T)
        if (now - last_valid_time['face']) < AVAILABILITY_TIMEOUT and last_valid_time['face'] > 0:
            ui_mask[1] = 1.0
        
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
            ui_mask[2] = 1.0
        
        # --- Handwriting (mask index 3) ---
        if shared_state.get('handwriting_clear_flag', False):
            buffers['handwriting'] = np.zeros((T, 9), dtype=np.float32)
            buffer_fills['handwriting'] = 0
            last_valid_time['handwriting'] = 0
            shared_state['handwriting_clear_flag'] = False
            shared_state['handwriting_features'] = None
            hw_f = None

        if hw_f is not None:
            arr = np.array(hw_f)
            # The architecture intentionally uses the static handwriting feature over the entire 10-step window
            buffers['handwriting'][:] = arr
            if np.count_nonzero(arr) > 0:
                last_valid_time['handwriting'] = now
                buffer_fills['handwriting'] = T
            shared_state['handwriting_features'] = None # consume it so we don't process it repeatedly
            
        if (now - last_valid_time['handwriting']) < AVAILABILITY_TIMEOUT and last_valid_time['handwriting'] > 0:
            ui_mask[3] = 1.0
        
        # --- Eye/Pupil (mask index 4) ---
        if eye_f is not None:
            arr = np.array(eye_f)
            buffers['eye'] = np.roll(buffers['eye'], -1, axis=0)
            buffers['eye'][-1] = arr
            if np.count_nonzero(arr) > 0:
                last_valid_time['eye'] = now
                buffer_fills['eye'] = min(buffer_fills['eye'] + 1, T)
        if (now - last_valid_time['eye']) < AVAILABILITY_TIMEOUT and last_valid_time['eye'] > 0:
            ui_mask[4] = 1.0
            
        # Enforce trained modalities mask
        trained_modalities_str = shared_state.get('trained_modalities', '[]')
        import json as _json
        try:
            trained_mods = _json.loads(trained_modalities_str)
        except Exception:
            trained_mods = []
            
        trained_modalities_mapping = {
            0: ['audio', 'speech'],
            1: ['face', 'facial'],
            2: ['keystroke', 'keyboard'],
            3: ['handwriting'],
            4: ['eye', 'eye_pupil']
        }
        
        for i in range(5):
            if ui_mask[i] > 0.5:
                is_trained_mod = any(m in trained_mods for m in trained_modalities_mapping[i])
                if not is_trained_mod:
                    ui_mask[i] = 0.0
            
        inputs = {
            'input_audio': tf.convert_to_tensor(np.expand_dims(buffers['audio'], axis=0), dtype=tf.float32),
            'input_face': tf.convert_to_tensor(np.expand_dims(buffers['face'], axis=0), dtype=tf.float32),
            'input_keystroke': tf.convert_to_tensor(np.expand_dims(buffers['keystroke'], axis=0), dtype=tf.float32),
            'input_handwriting': tf.convert_to_tensor(np.expand_dims(buffers['handwriting'], axis=0), dtype=tf.float32),
            'input_eye': tf.convert_to_tensor(np.expand_dims(buffers['eye'], axis=0), dtype=tf.float32),
            'input_mask': tf.convert_to_tensor(np.expand_dims(ui_mask, axis=0), dtype=tf.float32)
        }
        
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
            if ui_mask[idx] < 0.5:
                unavailable.append(name)
        
        # Build prediction history entry (architecture debug when untrained)
        if is_trained:
            history_entry = {
                'timestamp': now_ts,
                'prediction': 'STRESS' if pred_label == 1 else 'NON-STRESS',
                'stress_prob_pct': round(smoothed_stress_prob * 100, 1),
                'source': 'trained_model',
            }
        else:
            history_entry = {
                'timestamp': now_ts,
                'prediction': 'ARCHITECTURE_TEST',
                'stress_prob_pct': round(raw_stress_prob * 100, 1),
                'source': 'untrained_debug',
            }
        prediction_history.append(history_entry)
        if len(prediction_history) > MAX_HISTORY:
            prediction_history.pop(0)
        
        # Write all fields to shared state
        shared_state['predictions_available'] = is_trained
        shared_state['fusion_prob'] = round(smoothed_stress_prob, 4) if is_trained else None
        shared_state['fusion_pred'] = pred_label if is_trained else -1
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
            if ui_mask[i] > 0.5 and mod in latencies:
                active_lats.append(latencies[mod])
        max_ext_ms = max(active_lats) if active_lats else 0.0
        
        pipeline_ms = lat_fusion + max_ext_ms
        shared_state['fusion_latency_ms'] = round(lat_fusion, 1)
        shared_state['max_extraction_ms'] = round(max_ext_ms, 1)
        shared_state['pipeline_latency_ms'] = round(pipeline_ms, 1)

        feat_lat_samples.append(max_ext_ms)
        inf_lat_samples.append(lat_fusion)
        total_lat_samples.append(pipeline_ms)
        cycle_timestamps.append(now)
        if len(feat_lat_samples) > LATENCY_WINDOW:
            feat_lat_samples.pop(0)
            inf_lat_samples.pop(0)
            total_lat_samples.pop(0)
            cycle_timestamps.pop(0)

        def _percentiles(samples):
            if not samples:
                return {'p50': 0.0, 'p95': 0.0, 'max': 0.0}
            ordered = sorted(samples)
            n = len(ordered)
            p50 = ordered[n // 2]
            p95 = ordered[min(n - 1, int(n * 0.95))]
            return {
                'p50': round(p50, 1),
                'p95': round(p95, 1),
                'max': round(max(ordered), 1),
            }

        scheduler_hz = 0.0
        if len(cycle_timestamps) >= 2:
            span = cycle_timestamps[-1] - cycle_timestamps[0]
            if span > 0:
                scheduler_hz = round((len(cycle_timestamps) - 1) / span, 2)

        shared_state['latency_percentiles'] = _json.dumps({
            'feature_extraction_ms': _percentiles(feat_lat_samples),
            'model_inference_ms': _percentiles(inf_lat_samples),
            'total_pipeline_ms': _percentiles(total_lat_samples),
            'scheduler_frequency_hz': scheduler_hz,
        })
        
        shared_state['fusion_mask'] = _json.dumps(ui_mask.tolist())
        shared_state['buffer_fills'] = _json.dumps({
            'audio': buffer_fills['audio'],
            'face': buffer_fills['face'],
            'keystroke': buffer_fills['keystroke'],
            'handwriting': buffer_fills['handwriting'],
            'eye': buffer_fills['eye']
        })
        
        latencies['fusion'] = lat_fusion
        latencies['pipeline'] = lat_fusion + max_ext_ms
        
        # Real-time logging
        try:
            with open(log_file_path, 'a') as lf:
                log_line = f"{now_ts} | Mask: {ui_mask.tolist()} | Prob: {smoothed_stress_prob:.4f} | Attn: {[round(a, 3) for a in alphas]} | Ext: {max_ext_ms:.1f}ms | Inf: {lat_fusion:.1f}ms | Total: {(lat_fusion + max_ext_ms):.1f}ms\n"
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
import datetime as _dt
local_cache = {}

MODALITY_API_ORDER = ['keyboard', 'speech', 'facial', 'eye_pupil', 'handwriting']
# Internal model mask order: [audio/speech, face/facial, keystroke/keyboard, handwriting, eye/pupil]
MASK_INDEX_TO_MODALITY = {
    0: 'speech',
    1: 'facial',
    2: 'keyboard',
    3: 'handwriting',
    4: 'eye_pupil',
}


def _parse_json_field(raw, default):
    import json as _json
    if raw and isinstance(raw, str):
        try:
            return _json.loads(raw)
        except Exception:
            return default
    return default


def build_status_payload(state):
    """Build the canonical /status JSON contract for frontend and tests."""
    lat = dict(state.get('latency', {}) or {})
    model_status = state.get('model_status', 'LOADING')
    is_trained = model_status == 'TRAINED'
    is_live = bool(state.get('is_live_monitoring', False))

    reliabilities = _parse_json_field(state.get('reliabilities'), {})
    modality_attention = _parse_json_field(state.get('modality_weights'), {})
    fusion_mask = _parse_json_field(state.get('fusion_mask'), [0, 0, 0, 0, 0])
    unavailable = _parse_json_field(state.get('unavailable_modalities'), [])
    available = [m for m in MODALITY_API_ORDER if m not in unavailable]

    buffer_fills = _parse_json_field(state.get('buffer_fills'), {})
    keyboard_fills = buffer_fills.get('keystroke', 0)
    keyboard_window_ready = keyboard_fills >= 10

    if model_status == 'LOADING':
        status = 'LOADING'
    elif not is_trained:
        status = 'ARCHITECTURE_TEST_ONLY' if is_live else 'READY'
    elif is_live:
        status = 'LIVE'
    else:
        status = 'READY'

    fusion_pred = state.get('fusion_pred', 0)
    if is_trained and is_live:
        if not keyboard_window_ready:
            prediction = 'WAITING FOR KEYBOARD WINDOW'
            stress_probability = None
            non_stress_probability = None
            confidence = None
        else:
            prediction = 'STRESS' if fusion_pred == 1 else 'NON-STRESS'
            stress_probability = state.get('smoothed_stress_prob', 0.0)
            non_stress_probability = state.get('smoothed_nonstress_prob', 1.0)
            confidence = state.get('confidence', 0.0)
    else:
        prediction = 'NOT AVAILABLE'
        stress_probability = None
        non_stress_probability = None
        confidence = None

    feature_extraction_ms = state.get('max_extraction_ms', 0.0)
    model_inference_ms = state.get('fusion_latency_ms', 0.0)
    total_pipeline_ms = state.get('pipeline_latency_ms', 0.0)

    payload = {
        'status': status,
        'model_status': model_status,
        'predictions_available': is_trained and is_live,
        'prediction': prediction,
        'stress_probability': stress_probability,
        'non_stress_probability': non_stress_probability,
        'confidence': confidence,
        'keyboard_window_ready': keyboard_window_ready,
        'modality_reliability': reliabilities,
        'modality_attention': modality_attention,
        'available_modalities': available,
        'unavailable_modalities': unavailable,
        'trained_modalities': _parse_json_field(state.get('trained_modalities'), []),
        'usable_modalities': [m for m in available if m in _parse_json_field(state.get('trained_modalities'), []) or (m == 'speech' and 'audio' in _parse_json_field(state.get('trained_modalities'), [])) or (m == 'facial' and 'face' in _parse_json_field(state.get('trained_modalities'), [])) or (m == 'keyboard' and 'keystroke' in _parse_json_field(state.get('trained_modalities'), [])) or (m == 'eye_pupil' and 'eye' in _parse_json_field(state.get('trained_modalities'), []))],
        'prediction_source': 'trained_model' if is_trained else 'untrained_debug',
        'window_size': 10,
        'timestamp': _dt.datetime.now().isoformat(timespec='seconds'),
        'latency': {
            'feature_extraction_ms': feature_extraction_ms,
            'model_inference_ms': model_inference_ms,
            'total_pipeline_ms': total_pipeline_ms,
        },
        'latency_percentiles': _parse_json_field(state.get('latency_percentiles'), {}),
        # Legacy / extended fields consumed by dashboard
        'dataset_name': state.get('dataset_name', 'UNKNOWN'),
        'checkpoint_name': state.get('checkpoint_name', ''),
        'model_load_time_ms': state.get('model_load_time_ms', 0.0),
        'face_status': state.get('face_status', 'UNKNOWN'),
        'eye_status': state.get('eye_status', 'UNKNOWN'),
        'mic_status': state.get('mic_status', 'UNKNOWN'),
        'keyboard_status': state.get('keyboard_status', 'UNKNOWN'),
        'handwriting_status': state.get('handwriting_status', 'UNKNOWN'),
        'fusion_prob': state.get('fusion_prob', 0.0),
        'fusion_pred': fusion_pred,
        'latency_detail': lat,
        'modality_weights': modality_attention,
        'reliabilities': reliabilities,
        'prediction_timestamp': state.get('prediction_timestamp', ''),
        'prediction_history': _parse_json_field(state.get('prediction_history'), []),
        'processing_latency_ms': model_inference_ms,
        'pipeline_latency_ms': total_pipeline_ms,
        'max_extraction_ms': feature_extraction_ms,
        'keyboard_event_count': state.get('keyboard_event_count', 0),
        'audio_sample_count': state.get('audio_sample_count', 0),
        'camera_frame_count': state.get('camera_frame_count', 0),
        'face_detection_count': state.get('face_detection_count', 0),
        'eye_detection_count': state.get('eye_detection_count', 0),
        'handwriting_submission_count': state.get('handwriting_submission_count', 0),
        'fusion_mask': fusion_mask,
        'buffer_fills': _parse_json_field(state.get('buffer_fills'), {}),
        'is_live_monitoring': is_live,
        'session_start_time': state.get('session_start_time', 0.0),
        'total_cycles': state.get('total_cycles', 0),
        'successful_cycles': state.get('successful_cycles', 0),
        'failed_cycles': state.get('failed_cycles', 0),
        'dropped_cycles': state.get('dropped_cycles', 0),
        'app_start_time': state.get('app_start_time', 0.0),
        'camera_connected': state.get('camera_connected', False),
        'camera_index': state.get('camera_index', -1),
        'camera_fps': state.get('camera_fps', 0.0),
        'last_frame_timestamp': state.get('last_frame_timestamp', 0.0),
        'face_detected': state.get('face_detected', False),
        'face_count': state.get('face_count', 0),
        'face_confidence': state.get('face_confidence', 'N/A'),
        'face_bbox': state.get('face_bbox', None),
    }

    if model_status == 'UNTRAINED' and is_live:
        payload['architecture_debug'] = {
            'note': 'Synthetic/untrained forward-pass values — NOT valid stress detection',
            'raw_stress_probability': state.get('raw_stress_prob', 0.0),
            'raw_non_stress_probability': state.get('raw_nonstress_prob', 1.0),
            'smoothed_stress_probability': state.get('smoothed_stress_prob', 0.0),
            'modality_attention': modality_attention,
            'modality_reliability': reliabilities,
        }

    return payload


def update_local_cache():
    while True:
        try:
            if not manager_dict.get('running', True):
                break
            # Do NOT use dict(manager_dict) as it copies large video frames and causes IPC deadlocks.
            # Instead, let build_status_payload pull exactly the scalar keys it needs.
            new_cache = build_status_payload(manager_dict)
            global local_cache
            local_cache = new_cache
        except Exception:
            pass
        time.sleep(0.1)



@app.route('/status')
def status():
    return jsonify(local_cache if local_cache else {})

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
            manager_dict['status'] = 'LIVE'
    elif action == 'stop':
        manager_dict['is_live_monitoring'] = False
        manager_dict['status'] = 'READY'
    elif action == 'reset':
        manager_dict['total_cycles'] = 0
        manager_dict['successful_cycles'] = 0
        manager_dict['failed_cycles'] = 0
        manager_dict['dropped_cycles'] = 0
        manager_dict['session_start_time'] = time.time() if manager_dict['is_live_monitoring'] else 0.0
        manager_dict['status'] = 'READY'
    elif action == 'clear_handwriting':
        manager_dict['handwriting_status'] = 'WAITING_FOR_INPUT'
        manager_dict['handwriting_clear_flag'] = True
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
        'latency_percentiles': '{}',
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
        app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False, threaded=True)
    except KeyboardInterrupt:
        pass
    finally:
        main_shared_state['running'] = False
        for w in workers:
            w.terminate()
            w.join()

