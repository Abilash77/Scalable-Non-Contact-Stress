import os
import sys
import time
import multiprocessing
import numpy as np
import cv2
import queue
import glob
import base64
import json
import urllib.error
import urllib.request
import uuid
import datetime
import matplotlib
matplotlib.use('Agg')

from flask import Flask, jsonify, request, Response, render_template, send_from_directory


def _load_local_env():
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if not os.path.exists(env_path):
        return
    try:
        with open(env_path, 'r', encoding='utf-8') as env_file:
            for line in env_file:
                line = line.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                name, value = line.split('=', 1)
                os.environ.setdefault(name.strip(), value.strip().strip('"\''))
    except OSError:
        pass


_load_local_env()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from audio_utils import extract_audio_features
from face_utils import extract_face_features
from eye_utils import extract_eye_features
from handwriting_utils import extract_handwriting_features
from keyboard_stress_features import (
    KEYSTROKES_PER_SUBWINDOW, NUM_TIMESTEPS, KeyboardStressPredictor,
    browser_events_to_raw, compute_subwindow_features, events_to_pairs,
)
from pdf_generator import generate_session_pdf

try:
    import pyaudio
except ImportError:
    pyaudio = None
    
try:
    from pynput import keyboard
except ImportError:
    keyboard = None

import threading
KEY_EVENTS_LOCK = threading.Lock()


def keyboard_worker(shared_state):
    """
    Real browser key events -> keystroke pairs -> 4-keystroke sub-windows -> 7D
    features (identical code path to the Freihaut training data). Each complete
    sub-window is appended to `keyboard_window` (last 10 = one model sample).
    """
    import json
    shared_state['keyboard_status'] = "WAITING_FOR_INPUT"
    raw_events = []            # dataset-format KeyDown/KeyUp events (ms)
    seen_event_ids = set()
    consumed_down = None       # down_time of the last keystroke already turned into a sub-window
    pending = []               # completed keystrokes not yet in a sub-window
    total_press_count = 0
    session_marker = None

    while shared_state['running']:
        time.sleep(0.05)
        if not shared_state.get('is_live_monitoring', False):
            shared_state['keyboard_status'] = "WAITING_FOR_INPUT"
            continue

        marker = shared_state.get('keyboard_session_marker')
        if marker != session_marker:  # new session / reset -> drop old typing
            session_marker = marker
            raw_events, pending, consumed_down = [], [], None
            seen_event_ids.clear()
            total_press_count = 0

        with KEY_EVENTS_LOCK:
            remote_events_json = shared_state.get('remote_keystroke_events')
            shared_state['remote_keystroke_events'] = None
        if not remote_events_json:
            continue
        try:
            new_events = [e for e in json.loads(remote_events_json)
                          if e.get('event_id') is not None and e['event_id'] not in seen_event_ids]
        except Exception:
            continue
        for e in new_events:
            seen_event_ids.add(e['event_id'])
            if e.get('action') == 'press' and not e.get('repeat'):
                total_press_count += 1
        if not new_events:
            continue

        t0 = time.perf_counter()
        raw_events.extend(browser_events_to_raw(new_events))
        raw_events.sort(key=lambda ev: ev['time'])
        raw_events = raw_events[-600:]
        pairs = events_to_pairs(raw_events)
        # Only the newest keystroke still being held can be incomplete; everything
        # older is final, so hold back pairs whose KeyUp may still be reordered.
        fresh = [p for p in pairs if consumed_down is None or p['down_time'] > consumed_down]
        pending = fresh
        new_vectors = []
        while len(pending) >= KEYSTROKES_PER_SUBWINDOW:
            chunk, pending = pending[:KEYSTROKES_PER_SUBWINDOW], pending[KEYSTROKES_PER_SUBWINDOW:]
            consumed_down = chunk[-1]['down_time']
            feats = compute_subwindow_features(chunk)
            if feats is not None:  # same rule as training: unusable sub-windows are skipped
                new_vectors.append(feats.tolist())
        lat = (time.perf_counter() - t0) * 1000

        shared_state['keyboard_event_count'] = total_press_count
        shared_state['keyboard_pending_keystrokes'] = len(pending)
        shared_state['keyboard_status'] = "RECEIVING"
        if not new_vectors:
            continue

        window = list(shared_state.get('keyboard_window') or [])
        window = (window + new_vectors)[-NUM_TIMESTEPS:]
        shared_state['keyboard_window'] = window
        shared_state['keystroke_features'] = new_vectors[-1]
        shared_state['keystroke_features_display'] = [round(v, 3) for v in new_vectors[-1]]
        shared_state['keyboard_sample_id'] = shared_state.get('keyboard_sample_id', 0) + len(new_vectors)
        shared_state['keyboard_last_sample_time'] = time.time()
        shared_state['keyboard_sample_count'] = len(window)
        shared_state['latency']['keystroke'] = lat

def audio_worker(shared_state):
    shared_state['mic_status'] = "WAITING_FOR_MIC"
    RATE = 16000
    sample_id = 0
    session_marker = None
    # Warm up librosa/numba once at startup so the first real microphone chunk
    # is not delayed by JIT compilation.
    try:
        extract_audio_features(audio_segment=(0.01 * np.random.RandomState(0).randn(RATE)).astype(np.float32), sr=RATE)
    except Exception as e:
        print(f"Audio warm-up failed: {e}")

    while shared_state['running']:
        time.sleep(0.5)  # Poll twice a second
        marker = shared_state.get('keyboard_session_marker')
        if marker != session_marker:
            session_marker = marker
            sample_id = 0
            shared_state['audio_features'] = None
            shared_state['audio_sample_id'] = 0
            shared_state['audio_sample_count'] = 0
        if not shared_state.get('is_live_monitoring', False):
            shared_state['mic_status'] = "WAITING_FOR_MIC"
            continue
            
        b64_audio = shared_state.get('remote_audio_b64')
        if b64_audio:
            try:
                import base64
                audio_bytes = base64.b64decode(b64_audio)
                audio_data = np.frombuffer(audio_bytes, dtype=np.float32)
                
                # Diagnostic variables
                audio_recv = audio_data.size > 0
                audio_samples = audio_data.size
                audio_duration = round(audio_samples / RATE, 2) if RATE else 0
                
                if not audio_recv or float(np.sqrt(np.mean(audio_data ** 2))) <= 1e-5:
                    shared_state['audio_features'] = None
                    continue
                
                t0 = time.perf_counter()
                feats = extract_audio_features(audio_segment=audio_data, sr=RATE)
                lat = (time.perf_counter() - t0) * 1000
                
                feat_valid = np.count_nonzero(feats) > 0
                finite_feats = np.all(np.isfinite(feats))
                
                print("SPEECH DEBUG")
                print(f"audio_received = {audio_recv}")
                print(f"audio_samples = {audio_samples}")
                print(f"audio_duration = {audio_duration}")
                print(f"feature_extraction = {'PASS' if feat_valid else 'FAIL'}")
                print(f"feature_shape = {feats.shape}")
                print(f"finite_features = {finite_feats}")
                print("speech_model_available = False")
                print("speech_prediction = NONE")
                
                if not feat_valid:
                    shared_state['audio_features'] = None
                    continue

                sample_id += 1
                shared_state['audio_features'] = feats.tolist()
                shared_state['audio_features_summary'] = {
                    'mfcc_1_4': [round(float(v), 2) for v in feats[:4]],
                    'rms_dbfs': round(float(20 * np.log10(np.sqrt(np.mean(audio_data ** 2)) + 1e-9)), 1),
                    'seconds': audio_duration,
                }
                shared_state['audio_sample_id'] = sample_id
                shared_state['audio_last_sample_time'] = time.time()
                shared_state['audio_sample_count'] = sample_id
                shared_state['mic_status'] = "RECEIVING_AUDIO"
                latencies = shared_state['latency']
                latencies['audio'] = lat
            except Exception as e:
                print(f"Audio processing error: {e}")
            finally:
                shared_state['remote_audio_b64'] = None


def webcam_worker(shared_state):
    shared_state['camera_connected'] = False
    shared_state['face_status'] = "WAITING_FOR_CAMERA"
    shared_state['eye_status'] = "WAITING_FOR_CAMERA"
    
    frame_count = 0
    face_detection_count = 0
    eye_detection_count = 0
    fps_start_time = time.time()
    session_marker = None
    
    fer_model = None
    fer_classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    try:
        from tensorflow.keras.models import load_model
        import os
        model_path = os.path.join('results', 'checkpoints', 'fer2013_facial_emotion.keras')
        if os.path.exists(model_path):
            fer_model = load_model(model_path)
            shared_state['face_model_loaded'] = True
            print("[MODEL] FER-2013 Facial Emotion Model Loaded in Webcam Worker")
    except Exception as e:
        print(f"FER model load error: {e}")
    
    while shared_state['running']:
        time.sleep(0.1)
        marker = shared_state.get('keyboard_session_marker')
        if marker != session_marker:
            session_marker = marker
            frame_count = 0
            face_detection_count = 0
            eye_detection_count = 0
            shared_state['face_features'] = None
            shared_state['eye_features'] = None
            shared_state['camera_frame_count'] = 0
            shared_state['face_detection_count'] = 0
            shared_state['eye_detection_count'] = 0
            shared_state['face_sample_id'] = 0
            shared_state['eye_sample_id'] = 0
            fps_start_time = time.time()
        if not shared_state.get('is_live_monitoring', False):
            shared_state['camera_connected'] = False
            shared_state['face_status'] = "WAITING_FOR_CAMERA"
            shared_state['eye_status'] = "WAITING_FOR_CAMERA"
            continue

        b64_frame = shared_state.get('remote_frame_b64')
        if not b64_frame:
            last_ts = shared_state.get('last_frame_timestamp', 0)
            if last_ts > 0 and (time.time() - last_ts) > 3.0:
                shared_state['camera_connected'] = False
                shared_state['face_status'] = "OFFLINE"
                shared_state['eye_status'] = "OFFLINE"
            continue
            
        try:
            import base64
            img_data = base64.b64decode(b64_frame.split(',')[1] if ',' in b64_frame else b64_frame)
            nparr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                continue
                
            print(f"CAMERA_FRAME_RECEIVED | timestamp={time.time()} | frame_size={len(img_data)} | decoded_width={frame.shape[1]} | decoded_height={frame.shape[0]}", flush=True)
                
            shared_state['camera_connected'] = True
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

            frame_small = cv2.resize(frame, (320, 240))

            t0 = time.perf_counter()
            face_feats, face_status, face_meta = extract_face_features(frame_small)
            eye_feats, eye_status = extract_eye_features(frame_small)
            lat = (time.perf_counter() - t0) * 1000

            shared_state['face_status'] = face_status
            shared_state['eye_status'] = eye_status

            if face_status == "DETECTED":
                print(f"FACE_DETECTION | faces={face_meta.get('count', 1)}", flush=True)
                face_detection_count += 1
                shared_state['face_last_valid_time'] = time.time()
                shared_state['face_sample_id'] = face_detection_count
                shared_state['face_last_sample_time'] = shared_state['face_last_valid_time']

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

                if fer_model is not None:
                    # face_meta.get('bbox') is based on frame_small (320x240)
                    bx, by, bw, bh = bbox
                    face_crop = frame_small[max(0, by):min(240, by+bh), max(0, bx):min(320, bx+bw)]
                    if face_crop.shape[0] > 0 and face_crop.shape[1] > 0:
                        face_resized = cv2.resize(face_crop, (48, 48))
                        face_input = np.expand_dims(face_resized, axis=0)
                        preds = fer_model(face_input, training=False).numpy()[0]
                        probs = {fer_classes[i]: float(preds[i]) for i in range(len(fer_classes))}
                        top_emotion = fer_classes[np.argmax(preds)].upper()
                        print(f"FACIAL_EXPRESSION | expression={top_emotion} | confidence={preds[np.argmax(preds)]:.4f}", flush=True)
                        shared_state['face_emotion_probs'] = probs
                        shared_state['face_emotion_label'] = top_emotion
                        shared_state['face_fer_inference_count'] = shared_state.get('face_fer_inference_count', 0) + 1

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
                print(f"EYE_FEATURES | dimension={len(eye_feats)} | timestamp={time.time()}", flush=True)
                eye_detection_count += 1
                shared_state['eye_last_valid_time'] = time.time()
                shared_state['eye_sample_id'] = eye_detection_count
                shared_state['eye_last_sample_time'] = shared_state['eye_last_valid_time']

            ret_enc, buffer = cv2.imencode('.jpg', frame)
            if ret_enc:
                shared_state['latest_frame_jpg'] = buffer.tobytes()

            shared_state['face_features'] = face_feats.tolist() if face_status == "DETECTED" else None
            shared_state['eye_features'] = eye_feats.tolist() if eye_status == "DETECTED" else None
            shared_state['face_features_display'] = [round(float(v), 3) for v in face_feats] if face_status == "DETECTED" else None
            shared_state['eye_features_display'] = [round(float(v), 3) for v in eye_feats] if eye_status == "DETECTED" else None
            shared_state['face_detection_count'] = face_detection_count
            shared_state['eye_detection_count'] = eye_detection_count

            latencies = shared_state['latency']
            latencies['face'] = lat
            latencies['eye'] = lat
        except Exception as e:
            print(f"Webcam processing error: {e}")
        finally:
            shared_state['remote_frame_b64'] = None

def handwriting_worker(shared_state):
    shared_state['handwriting_status'] = "WAITING_FOR_INPUT"
    sample_id = 0
    session_marker = None
    while shared_state['running']:
        time.sleep(0.1)
        marker = shared_state.get('keyboard_session_marker')
        if marker != session_marker:
            session_marker = marker
            sample_id = 0
            shared_state['handwriting_features'] = None
            shared_state['handwriting_sample_id'] = 0
            shared_state['handwriting_submission_count'] = 0
        if not shared_state.get('is_live_monitoring', False):
            continue
            
        b64_img = shared_state.get('live_handwriting_b64')
        if b64_img:
            t0 = time.perf_counter()
            try:
                img_data = base64.b64decode(b64_img)
                nparr = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
                
                strokes = None
                strokes_json = shared_state.get('live_handwriting_strokes')
                if strokes_json:
                    import json as _json
                    try:
                        strokes = _json.loads(strokes_json)
                    except Exception:
                        strokes = None
                feats = extract_handwriting_features(img_array=img, strokes=strokes)
                if np.count_nonzero(feats[:7]) == 0:  # blank canvas
                    shared_state['handwriting_status'] = "WAITING_FOR_INPUT"
                    shared_state['live_handwriting_b64'] = None
                    shared_state['live_handwriting_strokes'] = None
                    continue

                sample_id += 1
                shared_state['handwriting_status'] = "RECEIVING"
                shared_state['handwriting_features'] = feats.tolist()
                shared_state['handwriting_features_display'] = [round(float(v), 3) for v in feats]
                shared_state['handwriting_sample_id'] = sample_id
                shared_state['handwriting_submission_count'] = sample_id
                shared_state['handwriting_last_sample_time'] = time.time()
                shared_state['handwriting_last_valid_time'] = shared_state['handwriting_last_sample_time']
            except Exception as e:
                shared_state['handwriting_status'] = f"ERROR: {e}"
                
            lat = (time.perf_counter() - t0) * 1000
            latencies = shared_state['latency']
            latencies['handwriting'] = lat
            
            shared_state['live_handwriting_b64'] = None
            shared_state['live_handwriting_strokes'] = None

FEATURE_ONLY_BUFFERS = {'audio': 169, 'face': 12, 'handwriting': 9, 'eye': 5}


def load_keyboard_predictor(shared_state):
    """Load the validated keyboard stress model once. No retraining, no fallback."""
    import json as _json
    meta_path = os.path.join('results', 'checkpoints', 'keyboard_stress_model_metadata.json')
    t0 = time.perf_counter()
    try:
        predictor = KeyboardStressPredictor(meta_path)
        # Real forward pass on a real-shaped input to fail fast on incompatible files.
        predictor.predict_proba(np.zeros((1, NUM_TIMESTEPS, 7), dtype=np.float32))
    except Exception as exc:
        print("\n" + "=" * 50)
        print("MODEL STATUS: NO VALID KEYBOARD STRESS MODEL")
        print(f"REASON: {exc}")
        print("ACTION: python scripts/preprocess_freihaut.py && python train.py --retrain")
        print("=" * 50 + "\n")
        shared_state['model_status'] = 'READY FOR TRAINING'
        shared_state['dataset_name'] = 'FREIHAUT & GOERITZ KEYBOARD STRESS'
        shared_state['trained_modalities'] = _json.dumps([])
        shared_state['model_load_time_ms'] = round((time.perf_counter() - t0) * 1000, 1)
        return None

    meta = predictor.meta
    shared_state['model_status'] = 'TRAINED'
    shared_state['checkpoint_name'] = meta['model_file']
    shared_state['model_name'] = meta.get('model_name', '')
    shared_state['dataset_name'] = meta.get('dataset_name', 'Freihaut & Goeritz (2021)')
    
    t_mods = ['keyboard']
    # We will update t_mods in inference_worker once both models are loaded.
    shared_state['trained_modalities'] = _json.dumps(t_mods)
    
    shared_state['decision_threshold'] = predictor.threshold
    shared_state['model_test_metrics'] = _json.dumps({
        k: round(v, 4) for k, v in meta.get('test_metrics', {}).items() if isinstance(v, float)})
    shared_state['model_load_time_ms'] = round((time.perf_counter() - t0) * 1000, 1)
    print(f"[MODEL] Keyboard stress model loaded: {meta['model_file']} ({meta.get('model_name')})")
    print(f"[MODEL] Dataset: {shared_state['dataset_name']} | trained {meta.get('training_timestamp')}")
    print(f"[MODEL] Test balanced accuracy {meta['test_metrics']['balanced_accuracy']:.3f}, "
          f"ROC-AUC {meta['test_metrics']['roc_auc']:.3f} | threshold {predictor.threshold:.3f}")
    print("[MODEL] Face / eye / speech / handwriting: FEATURE EXTRACTION ONLY (no stress-trained model)")
    return predictor


def load_speech_predictor(shared_state):
    import os, joblib, json
    meta_path = os.path.join('results', 'checkpoints', 'speech_model_metadata.json')
    model_path = os.path.join('results', 'checkpoints', 'speech_stress_model.joblib')
    scaler_path = os.path.join('results', 'checkpoints', 'speech_scaler.joblib')
    if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
        return None, None
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        print(f"[MODEL] Speech model loaded: {meta.get('model_type')} (RAVDESS Emotion -> Stress proxy)")
        return model, scaler
    except Exception as e:
        print(f"Error loading speech model: {e}")
        return None, None


def inference_worker(shared_state):
    import datetime
    import json as _json

    predictor = load_keyboard_predictor(shared_state)
    is_trained = predictor is not None
    
    speech_model, speech_scaler = load_speech_predictor(shared_state)
    speech_is_trained = speech_model is not None
    
    import json as _json
    t_mods = []
    if is_trained: t_mods.append('keyboard')
    if speech_is_trained: t_mods.append('speech')
    shared_state['trained_modalities'] = _json.dumps(t_mods)

    T = NUM_TIMESTEPS
    buffers = {m: np.zeros((T, d), dtype=np.float32) for m, d in FEATURE_ONLY_BUFFERS.items()}
    buffer_fills = {'audio': 0, 'face': 0, 'keystroke': 0, 'handwriting': 0, 'eye': 0}
    AVAILABILITY_TIMEOUT = 10.0
    last_valid_time = {'audio': 0.0, 'face': 0.0, 'keystroke': 0.0, 'handwriting': 0.0, 'eye': 0.0}
    last_sample_ids = {'audio': 0, 'face': 0, 'keystroke': 0, 'handwriting': 0, 'eye': 0}
    MAX_HISTORY = 20
    prediction_history = []
    LATENCY_WINDOW = 100
    feat_lat_samples, inf_lat_samples, total_lat_samples, cycle_timestamps = [], [], [], []
    session_marker = None
    last_pred_latency = 0.0

    os.makedirs('results', exist_ok=True)
    log_file_path = 'results/realtime_session.log'
    mod_order = ['audio', 'face', 'keystroke', 'handwriting', 'eye']  # mask order

    def _percentiles(samples):
        if not samples:
            return {'p50': 0.0, 'p95': 0.0, 'max': 0.0}
        ordered = sorted(samples)
        n = len(ordered)
        return {'p50': round(ordered[n // 2], 1), 'p95': round(ordered[min(n - 1, int(n * 0.95))], 1),
                'max': round(ordered[-1], 1)}

    def _clear_prediction():
        shared_state['predictions_available'] = False
        shared_state['fusion_prob'] = None
        shared_state['fusion_pred'] = -1
        shared_state['raw_stress_prob'] = None
        shared_state['raw_nonstress_prob'] = None
        shared_state['smoothed_stress_prob'] = None
        shared_state['smoothed_nonstress_prob'] = None
        shared_state['confidence'] = None

    _clear_prediction()

    while shared_state['running']:
        time.sleep(0.1)
        if not shared_state.get('is_live_monitoring', False):
            continue

        marker = shared_state.get('keyboard_session_marker')
        if marker != session_marker:  # new session / reset: clear buffers and stale prediction
            session_marker = marker
            for m in buffers:
                buffers[m][:] = 0
            buffer_fills = {k: 0 for k in buffer_fills}
            last_valid_time = {k: 0.0 for k in last_valid_time}
            last_sample_ids = {k: 0 for k in last_sample_ids}
            prediction_history = []
            _clear_prediction()
            shared_state['unique_samples'] = {k: 0 for k in mod_order}
            shared_state['inference_counts'] = {k: 0 for k in mod_order}
            shared_state['prediction_history'] = '[]'

        t_start = time.perf_counter()
        now = time.time()
        shared_state['total_cycles'] += 1
        runtime_mask = np.zeros(5, dtype=np.float32)
        us = dict(shared_state.get('unique_samples') or {k: 0 for k in mod_order})
        ic = dict(shared_state.get('inference_counts') or {k: 0 for k in mod_order})

        # ---- feature-only modalities: consume new real samples -------------
        sources = {'audio': ('audio_features', 'audio_sample_id'),
                   'face': ('face_features', 'face_sample_id'),
                   'handwriting': ('handwriting_features', 'handwriting_sample_id'),
                   'eye': ('eye_features', 'eye_sample_id')}
        if shared_state.get('handwriting_clear_flag', False):
            buffers['handwriting'][:] = 0
            buffer_fills['handwriting'] = 0
            last_valid_time['handwriting'] = 0.0
            shared_state['handwriting_clear_flag'] = False
        new_audio_sample = False
        for m, (feat_key, id_key) in sources.items():
            feats = shared_state.get(feat_key)
            sid = int(shared_state.get(id_key, 0) or 0)
            if feats is not None and sid > last_sample_ids[m]:
                if m == 'audio':
                    new_audio_sample = True
                arr = np.asarray(feats, dtype=np.float32)
                last_sample_ids[m] = sid
                if arr.shape == (FEATURE_ONLY_BUFFERS[m],) and np.count_nonzero(arr) > 0:
                    buffers[m] = np.roll(buffers[m], -1, axis=0)
                    buffers[m][-1] = arr
                    buffer_fills[m] = min(buffer_fills[m] + 1, T)
                    last_valid_time[m] = now
                    us[m] = us.get(m, 0) + 1
            if last_valid_time[m] > 0 and (now - last_valid_time[m]) < AVAILABILITY_TIMEOUT:
                runtime_mask[mod_order.index(m)] = 1.0
                ic[m] = ic.get(m, 0) + 1

        # ---- keyboard: the only stress-trained modality --------------------
        window = list(shared_state.get('keyboard_window') or [])
        buffer_fills['keystroke'] = len(window)
        keyboard_id = int(shared_state.get('keyboard_sample_id', 0) or 0)
        new_keyboard_sample = keyboard_id > last_sample_ids['keystroke']
        if new_keyboard_sample:
            us['keystroke'] = us.get('keystroke', 0) + (keyboard_id - last_sample_ids['keystroke'])
            last_sample_ids['keystroke'] = keyboard_id
            last_valid_time['keystroke'] = now
        if (last_valid_time['keystroke'] > 0 and (now - last_valid_time['keystroke']) < 30.0) or shared_state.get('keyboard_status') == 'RECEIVING':
            runtime_mask[2] = 1.0
            if (now - float(shared_state.get('keyboard_last_sample_time', 0) or 0)) >= 30.0:
                shared_state['keyboard_status'] = 'IDLE'

        predicted_now = False
        camera_predicted_now = False
        speech_predicted_now = False
        
        if speech_is_trained and new_audio_sample and buffer_fills['audio'] > 0:
            try:
                a_feats = np.asarray(shared_state['audio_features'], dtype=np.float32).reshape(1, -1)
                a_feats_166 = a_feats[:, :166]  # Match the 166 features from the RAVDESS processed dataset
                a_feats_scaled = speech_scaler.transform(a_feats_166)
                s_probs = speech_model.predict_proba(a_feats_scaled)[0]
                s_stress_prob = float(s_probs[1])
                s_nonstress_prob = float(s_probs[0])
                s_pred_label = 1 if s_stress_prob >= 0.5 else 0
                s_confidence = s_stress_prob if s_pred_label == 1 else s_nonstress_prob
                
                shared_state['speech_stress_prob'] = round(s_stress_prob, 4)
                shared_state['speech_nonstress_prob'] = round(s_nonstress_prob, 4)
                shared_state['speech_confidence'] = round(s_confidence, 4)
                shared_state['speech_pred'] = s_pred_label
                shared_state['speech_predictions_available'] = True
                speech_predicted_now = True
            except Exception as e:
                with open('error.log', 'a') as f:
                    f.write(f"Speech prediction error: {e}\n")
                print(f"Speech prediction error: {e}")
        
        if is_trained and new_keyboard_sample and len(window) >= T:
            X = np.asarray(window[-T:], dtype=np.float32)[None]
            t_inf = time.perf_counter()
            try:
                k_stress_prob = float(predictor.predict_proba(X)[0])
                shared_state['successful_cycles'] += 1
            except Exception as e:
                shared_state['failed_cycles'] += 1
                print(f"Inference error: {e}")
                continue
            last_pred_latency = (time.perf_counter() - t_inf) * 1000
            ic['keystroke'] = ic.get('keystroke', 0) + 1
            k_pred_label = 1 if k_stress_prob >= predictor.threshold else 0
            k_nonstress_prob = 1.0 - k_stress_prob
            k_confidence = k_stress_prob if k_pred_label == 1 else k_nonstress_prob
            
            print(f"KEYBOARD_WINDOW_READY | dimension=7", flush=True)
            print(f"KEYBOARD_PREDICTION | stress_probability={k_stress_prob:.4f} | nonstress_probability={k_nonstress_prob:.4f} | confidence={k_confidence:.4f}", flush=True)
            
            shared_state['keyboard_stress_prob'] = round(k_stress_prob, 4)
            shared_state['keyboard_nonstress_prob'] = round(k_nonstress_prob, 4)
            shared_state['keyboard_confidence'] = round(k_confidence, 4)
            shared_state['keyboard_pred'] = k_pred_label
            shared_state['keyboard_predictions_available'] = True
            predicted_now = True

        face_probs = shared_state.get('face_emotion_probs')
        if face_probs and shared_state.get('face_detected', False):
            # Facial expression stress probability (FER-2013 labels)
            f_stress_prob = (
                face_probs.get('angry', 0) + 
                face_probs.get('disgust', 0) + 
                face_probs.get('fear', 0) + 
                face_probs.get('sad', 0)
            )
            f_nonstress_prob = (
                face_probs.get('happy', 0) + 
                face_probs.get('neutral', 0) + 
                face_probs.get('surprise', 0)
            )
            total_f = f_stress_prob + f_nonstress_prob
            if total_f > 0:
                f_stress_prob /= total_f
                f_nonstress_prob /= total_f
            else:
                f_stress_prob = 0.5
                f_nonstress_prob = 0.5
                
            # Eye feature based heuristic (Research)
            eye_feats = shared_state.get('eye_features')
            eye_stress = 0.0
            if eye_feats is not None and len(eye_feats) == 5:
                # Features: pupil_size_left, pupil_size_right, gaze_x, gaze_y, eye_closure
                pupil_avg = (eye_feats[0] + eye_feats[1]) / 2.0
                ear = eye_feats[4]
                
                # Heuristic logic: larger pupil size and higher EAR (widened eyes) increase stress score
                if ear > 0.3: eye_stress += (ear - 0.3) * 2.0
                if pupil_avg > 0.4: eye_stress += (pupil_avg - 0.4) * 2.0
                eye_stress = min(1.0, max(0.0, eye_stress))
                
                # Combine FER probabilities with Eye heuristic
                c_stress_prob = (0.7 * f_stress_prob) + (0.3 * eye_stress)
                c_nonstress_prob = 1.0 - c_stress_prob
            else:
                c_stress_prob = f_stress_prob
                c_nonstress_prob = f_nonstress_prob
            
            c_pred_label = 1 if c_stress_prob >= 0.5 else 0
            # Confidence for a heuristic score
            c_confidence = max(c_stress_prob, c_nonstress_prob)
            
            print(f"CAMERA_PREDICTION | stress_probability={c_stress_prob:.4f} | nonstress_probability={c_nonstress_prob:.4f} | confidence={c_confidence:.4f}", flush=True)
            
            shared_state['camera_stress_prob'] = round(c_stress_prob, 4)
            shared_state['camera_nonstress_prob'] = round(c_nonstress_prob, 4)
            shared_state['camera_confidence'] = round(c_confidence, 4)
            shared_state['camera_pred'] = c_pred_label
            shared_state['camera_predictions_available'] = True
            camera_predicted_now = True

        if predicted_now or camera_predicted_now or speech_predicted_now:
            now_ts = datetime.datetime.now().strftime("%H:%M:%S")
            hist_entry = {'timestamp': now_ts}
            if shared_state.get('camera_predictions_available'):
                hist_entry['camera_prediction'] = 'STRESS' if shared_state.get('camera_pred') == 1 else 'NON-STRESS'
                hist_entry['camera_stress_prob_pct'] = round(shared_state.get('camera_stress_prob', 0) * 100, 1)
                hist_entry['camera_confidence'] = shared_state.get('camera_confidence')
                hist_entry['prediction'] = hist_entry['camera_prediction']
                hist_entry['stress_prob_pct'] = hist_entry['camera_stress_prob_pct']
                
            if shared_state.get('keyboard_predictions_available'):
                hist_entry['keyboard_prediction'] = 'STRESS' if shared_state.get('keyboard_pred') == 1 else 'NON-STRESS'
                hist_entry['keyboard_stress_prob_pct'] = round(shared_state.get('keyboard_stress_prob', 0) * 100, 1)
                hist_entry['keyboard_confidence'] = shared_state.get('keyboard_confidence')
                
            if shared_state.get('speech_predictions_available'):
                hist_entry['speech_prediction'] = 'STRESS' if shared_state.get('speech_pred') == 1 else 'NON-STRESS'
                hist_entry['speech_stress_prob_pct'] = round(shared_state.get('speech_stress_prob', 0) * 100, 1)
                hist_entry['speech_confidence'] = shared_state.get('speech_confidence')
                
            prediction_history.append(hist_entry)
            prediction_history = prediction_history[-MAX_HISTORY:]
            shared_state['prediction_timestamp'] = now_ts
            shared_state['prediction_history'] = _json.dumps(prediction_history)
            
            shared_state['predictions_available'] = True
        elif not is_trained and not camera_predicted_now:
            shared_state['successful_cycles'] += 1

        shared_state['unique_samples'] = us
        shared_state['inference_counts'] = ic

        # Keyboard and Speech feed the stress model.
        fusion_mask = [
            1.0 if (speech_is_trained and buffer_fills['audio'] > 0) else 0.0,
            0.0, 
            1.0 if (is_trained and buffer_fills['keystroke'] >= T) else 0.0, 
            0.0, 
            0.0
        ]
        sum_f = sum(fusion_mask)
        if sum_f > 0:
            att_s = fusion_mask[0] / sum_f
            att_k = fusion_mask[2] / sum_f
        else:
            att_s = att_k = 0.0
            
        attention = {'keyboard': att_k, 'speech': att_s, 'facial': 0.0,
                     'eye_pupil': 0.0, 'handwriting': 0.0}
        shared_state['modality_weights'] = _json.dumps(attention)
        shared_state['reliabilities'] = _json.dumps(attention)
        shared_state['fusion_mask'] = _json.dumps(fusion_mask)
        shared_state['runtime_mask'] = _json.dumps(runtime_mask.tolist())
        shared_state['buffer_fills'] = _json.dumps(buffer_fills)
        shared_state['unavailable_modalities'] = _json.dumps(
            [n for n, v in zip(['speech', 'facial', 'keyboard', 'handwriting', 'eye_pupil'], fusion_mask) if v < 0.5])

        # ---- latency --------------------------------------------------------
        latencies = shared_state.get('latency', {})
        active_lats = [latencies[m] for i, m in enumerate(mod_order) if runtime_mask[i] > 0.5 and m in latencies]
        max_ext_ms = max(active_lats) if active_lats else 0.0
        shared_state['fusion_latency_ms'] = round(last_pred_latency, 2)
        shared_state['max_extraction_ms'] = round(max_ext_ms, 1)
        shared_state['pipeline_latency_ms'] = round(last_pred_latency + max_ext_ms, 1)
        feat_lat_samples.append(max_ext_ms)
        inf_lat_samples.append(last_pred_latency)
        total_lat_samples.append(last_pred_latency + max_ext_ms)
        cycle_timestamps.append(now)
        if len(feat_lat_samples) > LATENCY_WINDOW:
            for lst in (feat_lat_samples, inf_lat_samples, total_lat_samples, cycle_timestamps):
                lst.pop(0)
        scheduler_hz = 0.0
        if len(cycle_timestamps) >= 2 and cycle_timestamps[-1] > cycle_timestamps[0]:
            scheduler_hz = round((len(cycle_timestamps) - 1) / (cycle_timestamps[-1] - cycle_timestamps[0]), 2)
        shared_state['latency_percentiles'] = _json.dumps({
            'feature_extraction_ms': _percentiles(feat_lat_samples),
            'model_inference_ms': _percentiles(inf_lat_samples),
            'total_pipeline_ms': _percentiles(total_lat_samples),
            'scheduler_frequency_hz': scheduler_hz,
        })
        latencies['fusion'] = last_pred_latency
        latencies['pipeline'] = last_pred_latency + max_ext_ms

        # ---- runtime modality status ---------------------------------------
        runtime_status = {}
        mod_names_to_buf = {'speech': 'audio', 'facial': 'face', 'keyboard': 'keystroke',
                            'handwriting': 'handwriting', 'eye_pupil': 'eye'}
        mod_shapes = {'speech': 169, 'facial': 12, 'keyboard': 7, 'handwriting': 9, 'eye_pupil': 5}
        runtime_mask_map = {'speech': 0, 'facial': 1, 'keyboard': 2, 'handwriting': 3, 'eye_pupil': 4}
        hw_status_keys = {'speech': 'mic_status', 'facial': 'face_status', 'keyboard': 'keyboard_status',
                          'handwriting': 'handwriting_status', 'eye_pupil': 'eye_status'}
        sample_time_keys = {'keyboard': 'keyboard_last_sample_time', 'speech': 'audio_last_sample_time',
                            'facial': 'face_last_sample_time', 'eye_pupil': 'eye_last_sample_time',
                            'handwriting': 'handwriting_last_sample_time'}
        for mod_name in ['keyboard', 'speech', 'facial', 'eye_pupil', 'handwriting']:
            fill = buffer_fills.get(mod_names_to_buf[mod_name], 0)
            hw_st = str(shared_state.get(hw_status_keys[mod_name], 'UNKNOWN'))
            rm_active = runtime_mask[runtime_mask_map[mod_name]] > 0
            if 'HARDWARE_UNAVAILABLE' in hw_st or 'ERROR' in hw_st:
                status_str, input_str, feat_ready = 'UNAVAILABLE', 'NO HARDWARE', False
            elif mod_name == 'handwriting':
                if shared_state.get('handwriting_submission_count', 0) > 0 and fill > 0:
                    status_str, input_str, feat_ready = 'ACTIVE', 'RECEIVING', True
                else:
                    status_str, input_str, feat_ready = 'WAITING', 'WAITING FOR HANDWRITING', False
            elif rm_active or (mod_name in ('facial', 'eye_pupil') and shared_state.get('camera_connected', False)):
                status_str, feat_ready = 'ACTIVE', fill > 0
                if mod_name == 'keyboard':
                    input_str = 'RECEIVING' if fill > 0 else 'WAITING FOR TYPING'
                elif mod_name == 'speech':
                    input_str = 'RECEIVING AUDIO' if fill > 0 else 'WAITING FOR MICROPHONE'
                elif mod_name == 'facial':
                    input_str = 'FACE DETECTED' if hw_st == 'DETECTED' else 'NO FACE DETECTED'
                else:
                    input_str = 'EYES DETECTED' if hw_st == 'DETECTED' else 'WAITING FOR EYE INPUT'
            elif mod_name == 'speech':
                status_str = 'ACTIVE' if rm_active else 'FEATURE EXTRACTION ONLY'
                if status_str == 'FEATURE EXTRACTION ONLY' and fill == 0:
                    status_str = 'NO INPUT'
                feat_ready = fill > 0
                input_str = 'RECEIVING AUDIO' if fill > 0 else 'WAITING FOR MICROPHONE'
            elif mod_name == 'keyboard' and shared_state.get('keyboard_status') == 'RECEIVING':
                status_str, input_str, feat_ready = 'WAITING', 'KEEP TYPING', fill > 0
            else:
                status_str, input_str, feat_ready = 'NO INPUT', 'NO DATA', False
            runtime_status[mod_name] = {
                'status': status_str,
                'input': input_str,
                'features_ready': feat_ready,
                'feature_shape': f"({mod_shapes[mod_name]},)" if feat_ready else None,
                'buffer_fill': fill,
                'samples': fill,
                'feature_dimension': mod_shapes[mod_name],
                'last_sample_time': shared_state.get(sample_time_keys[mod_name], 0.0),
                'extraction_status': 'READY' if feat_ready else 'WAITING',
                'role': 'STRESS MODEL INPUT' if (mod_name == 'keyboard' or (mod_name == 'speech' and rm_active)) else 'FEATURE EXTRACTION ONLY',
            }
        shared_state['runtime_modality_status'] = _json.dumps(runtime_status)

        if predicted_now:
            try:
                with open(log_file_path, 'a') as lf:
                    lf.write(f"{shared_state['prediction_timestamp']} | keyboard_window=10x7 | "
                             f"P(stress)={shared_state['raw_stress_prob']:.4f} | thr={predictor.threshold:.3f} | "
                             f"pred={'STRESS' if shared_state['fusion_pred'] == 1 else 'NON-STRESS'} | "
                             f"inf={last_pred_latency:.2f}ms | runtime_mask={runtime_mask.tolist()}\n")
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
CAMERA_STALE_SECONDS = 3.0
MIC_STALE_SECONDS = 5.0
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
    if raw is None:
        return default
    if isinstance(raw, (dict, list)):
        return raw
    if isinstance(raw, str):
        try:
            return _json.loads(raw)
        except Exception:
            return default
    return default


def build_status_payload(state):
    """Build the canonical /status JSON contract for frontend and tests."""
    lat = dict(state.get('latency', {}) or {})
    model_status = state.get('model_status', 'LOADING')
    now = time.time()
    # Stale-input guard: if the browser stops sending frames / audio, do not keep
    # reporting the last FACE DETECTED / RECEIVING AUDIO state.
    last_frame_ts = float(state.get('last_frame_timestamp', 0.0) or 0.0)
    camera_fresh = last_frame_ts > 0 and (now - last_frame_ts) < CAMERA_STALE_SECONDS
    camera_connected = bool(state.get('camera_connected', False)) and camera_fresh
    face_status = state.get('face_status', 'UNKNOWN')
    eye_status = state.get('eye_status', 'UNKNOWN')
    if not camera_fresh and face_status not in ('HARDWARE_UNAVAILABLE',):
        face_status = 'WAITING_FOR_CAMERA'
    if not camera_fresh and eye_status not in ('HARDWARE_UNAVAILABLE',):
        eye_status = 'WAITING_FOR_CAMERA'
    face_detected = bool(state.get('face_detected', False)) and camera_fresh
    mic_status = state.get('mic_status', 'UNKNOWN')
    last_audio_ts = max(float(state.get('audio_last_upload_time', 0.0) or 0.0),
                        float(state.get('audio_last_sample_time', 0.0) or 0.0))
    if mic_status == 'RECEIVING_AUDIO' and (now - last_audio_ts) >= MIC_STALE_SECONDS:
        mic_status = 'WAITING_FOR_MIC'
    is_trained = model_status == 'TRAINED'
    is_live = bool(state.get('is_live_monitoring', False))

    reliabilities = _parse_json_field(state.get('reliabilities'), {})
    modality_attention = _parse_json_field(state.get('modality_weights'), {})
    fusion_mask = _parse_json_field(state.get('fusion_mask'), [0, 0, 0, 0, 0])
    runtime_mask = _parse_json_field(state.get('runtime_mask'), [0, 0, 0, 0, 0])
    unavailable = _parse_json_field(state.get('unavailable_modalities'), [])
    runtime_modality_status = _parse_json_field(state.get('runtime_modality_status'), {})

    # available_modalities: all modalities with active hardware (based on runtime_mask)
    runtime_mask_to_mod = {0: 'speech', 1: 'facial', 2: 'keyboard', 3: 'handwriting', 4: 'eye_pupil'}
    runtime_available = []
    for idx, val in enumerate(runtime_mask):
        if val > 0:
            runtime_available.append(runtime_mask_to_mod.get(idx, ''))
    # Also include modalities with hardware active but no data yet (e.g. handwriting waiting)
    hw_status_map = {
        'keyboard': state.get('keyboard_status', 'UNKNOWN'),
        'speech': mic_status,
        'facial': face_status,
        'eye_pupil': eye_status,
        'handwriting': state.get('handwriting_status', 'UNKNOWN'),
    }
    for mod_name, hw_st in hw_status_map.items():
        hw_str = str(hw_st)
        if 'HARDWARE_UNAVAILABLE' not in hw_str and 'ERROR' not in hw_str and mod_name not in runtime_available:
            if hw_str not in ('UNKNOWN', 'PENDING'):
                runtime_available.append(mod_name)
    available = list(dict.fromkeys(runtime_available))  # deduplicate, preserve order

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

    # Compute truly unavailable modalities (hardware problems only)
    truly_unavailable = []
    for mod_name, hw_st in hw_status_map.items():
        hw_str = str(hw_st)
        if 'HARDWARE_UNAVAILABLE' in hw_str or 'ERROR' in hw_str:
            truly_unavailable.append(mod_name)

    if is_trained and is_live:
        if not camera_connected:
            primary_prediction = 'WAITING FOR CAMERA'
        elif not face_detected:
            primary_prediction = 'WAITING FOR FACE'
        elif not state.get('camera_predictions_available', False):
            primary_prediction = 'WAITING FOR VALID CAMERA WINDOW'
        else:
            primary_prediction = 'STRESS' if state.get('camera_pred', 0) == 1 else 'NON-STRESS'
            
        primary_stress_prob = state.get('camera_stress_prob', 0.0)
        primary_nonstress_prob = state.get('camera_nonstress_prob', 0.0)
        primary_confidence = state.get('camera_confidence', 0.0)
        
        if not keyboard_window_ready or not state.get('keyboard_predictions_available', False):
            secondary_prediction = 'WAITING FOR KEYBOARD WINDOW'
        else:
            secondary_prediction = 'STRESS' if state.get('keyboard_pred', 0) == 1 else 'NON-STRESS'
            
        secondary_stress_prob = state.get('keyboard_stress_prob', 0.0)
        secondary_nonstress_prob = state.get('keyboard_nonstress_prob', 0.0)
        secondary_confidence = state.get('keyboard_confidence', 0.0)
        
        prediction = primary_prediction
        stress_probability = primary_stress_prob
        non_stress_probability = primary_nonstress_prob
        confidence = primary_confidence
    else:
        primary_prediction = 'NOT AVAILABLE'
        primary_stress_prob = None
        primary_nonstress_prob = None
        primary_confidence = None
        secondary_prediction = 'NOT AVAILABLE'
        secondary_stress_prob = None
        secondary_nonstress_prob = None
        secondary_confidence = None
        
        prediction = 'NOT AVAILABLE'
        stress_probability = None
        non_stress_probability = None
        confidence = None

    feature_extraction_ms = state.get('max_extraction_ms', 0.0)
    model_inference_ms = state.get('fusion_latency_ms', 0.0)
    total_pipeline_ms = state.get('pipeline_latency_ms', 0.0)

    trained_mods_raw = _parse_json_field(state.get('trained_modalities'), [])
    usable_mods = []
    for m in available:
        if m in trained_mods_raw:
            usable_mods.append(m)
        elif m == 'speech' and 'audio' in trained_mods_raw:
            usable_mods.append(m)
        elif m == 'facial' and 'face' in trained_mods_raw:
            usable_mods.append(m)
        elif m == 'keyboard' and 'keystroke' in trained_mods_raw:
            usable_mods.append(m)
        elif m == 'eye_pupil' and 'eye' in trained_mods_raw:
            usable_mods.append(m)

    fusion_pred = int(state.get('fusion_pred', -1))

    payload = {
        'status': status,
        'model_status': model_status,
        'predictions_available': is_trained and is_live,
        'prediction': prediction,
        'stress_probability': stress_probability,
        'non_stress_probability': non_stress_probability,
        'confidence': confidence,
        
        'primary_prediction': primary_prediction,
        'primary_stress_prob': primary_stress_prob,
        'primary_nonstress_prob': primary_nonstress_prob,
        'primary_confidence': primary_confidence,
        
        'secondary_prediction': secondary_prediction,
        'secondary_stress_prob': secondary_stress_prob,
        'secondary_nonstress_prob': secondary_nonstress_prob,
        'secondary_confidence': secondary_confidence,
        
        'keyboard_window_ready': keyboard_window_ready,
        'modality_reliability': reliabilities,
        'modality_attention': modality_attention,
        'available_modalities': available,
        'unavailable_modalities': truly_unavailable,
        'trained_modalities': trained_mods_raw,
        'usable_modalities': usable_mods,
        'runtime_modality_status': runtime_modality_status,
        'runtime_mask': runtime_mask,
        'prediction_source': 'trained_model' if is_trained else 'none',
        'stress_model': {
            'modality': 'keyboard',
            'model_name': state.get('model_name', ''),
            'decision_threshold': state.get('decision_threshold'),
            'test_metrics': _parse_json_field(state.get('model_test_metrics'), {}),
        },
        'feature_only_modalities': [m for m in ['speech', 'facial', 'eye_pupil', 'handwriting'] if m not in trained_mods_raw and not (m == 'speech' and 'audio' in trained_mods_raw)],
        'window_size': 10,
        'timestamp': _dt.datetime.now().isoformat(timespec='seconds'),
        'server_time': time.time(),
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
        'face_status': face_status,
        'eye_status': eye_status,
        'mic_status': mic_status,
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
        'face_fer_inference_count': state.get('face_fer_inference_count', 0),
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
        'camera_connected': camera_connected,
        'camera_index': state.get('camera_index', -1),
        'camera_width': state.get('camera_width', 0),
        'camera_height': state.get('camera_height', 0),
        'camera_fps': state.get('camera_fps', 0.0),
        'last_frame_timestamp': state.get('last_frame_timestamp', 0.0),
        'face_detected': face_detected,
        'face_count': state.get('face_count', 0) if face_detected else 0,
        'face_confidence': state.get('face_confidence', 'N/A'),
        'face_bbox': state.get('face_bbox', None) if face_detected else None,
        'face_emotion_label': state.get('face_emotion_label', None) if face_detected else None,
        'eye_detected': eye_status == 'DETECTED',
        'keyboard_pending_keystrokes': state.get('keyboard_pending_keystrokes', 0),
        'latest_features': {
            'keyboard_7d': state.get('keystroke_features_display'),
            'eye_5d': state.get('eye_features_display'),
            'face_12d': state.get('face_features_display'),
            'speech_169d_summary': state.get('audio_features_summary'),
            'handwriting_9d': state.get('handwriting_features_display'),
        },
        'face_emotion_probs': _parse_json_field(state.get('face_emotion_probs'), {}),
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



@app.after_request
def _no_store(resp):
    if request.path in ('/status', '/camera/status', '/api/model_status'):
        resp.headers['Cache-Control'] = 'no-store, max-age=0'
    return resp


@app.route('/status')
def status():
    # Rebuild on demand so the response is never older than the request.
    try:
        return jsonify(build_status_payload(manager_dict))
    except Exception:
        return jsonify(local_cache if local_cache else {})

@app.route('/api/chat', methods=['POST'])
def api_chat():
    data = request.get_json(silent=True) or {}
    question = str(data.get('message', '')).strip()
    if not question:
        return jsonify({'error': 'Please enter a question.'}), 400
    if len(question) > 1000:
        return jsonify({'error': 'Please keep questions under 1000 characters.'}), 400

    api_key = os.environ.get('GEMINI_API_KEY', '').strip()
    if not api_key:
        return jsonify({'error': 'Gemini is not configured on the backend.'}), 503

    project_context = """You explain this project only:
Scalable Non-Contact Stress Detection Using Hybrid Multimodal Intelligence.

Runtime modalities: Keyboard, Facial, Eye/Pupil, Speech, Handwriting.
Keyboard is currently the only actual stress-prediction modality. Real keyboard input becomes 7D typing features, 10 temporal samples, a trained Freihaut & Goeritz keyboard stress model, and STRESS/NON-STRESS probability/confidence.
Face, eye/pupil, speech, and handwriting receive real browser input and perform feature extraction only; they are not independently validated stress predictors.
FER emotion is never stress detection. RAVDESS emotion is never stress detection. Personality features are never stress detection.
The current keyboard evaluation is approximately 54.6% test accuracy and 0.554 ROC-AUC on the participant-independent Freihaut & Goeritz dataset, indicating weak predictive signal. Do not claim 94.3% as current accuracy.
The system is a research prototype and is not a medical diagnosis system.
Face uses the visitor webcam for real face detection and facial features. Eye uses the same camera for real landmarks and 5D eye features. Speech uses the visitor microphone for real 169D audio features. Handwriting uses the real canvas for 9D stroke features.
If a question is unrelated to this project, say that you are focused on explaining this project. Never invent metrics, datasets, models, features, capabilities, or results. Keep answers concise and accessible."""
    payload = {
        'contents': [{'role': 'user', 'parts': [{'text': question}]}],
        'systemInstruction': {'parts': [{'text': project_context}]},
        'generationConfig': {'temperature': 0.1, 'maxOutputTokens': 300},
    }
    endpoint = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent?key=' + api_key
    request_body = json.dumps(payload).encode('utf-8')
    gemini_request = urllib.request.Request(
        endpoint,
        data=request_body,
        headers={'Content-Type': 'application/json'},
        method='POST',
    )
    try:
        with urllib.request.urlopen(gemini_request, timeout=20) as response:
            result = json.loads(response.read().decode('utf-8'))
        answer = result['candidates'][0]['content']['parts'][0]['text'].strip()
        if not answer:
            raise ValueError('Gemini returned an empty response.')
        return jsonify({'answer': answer, 'provider': 'gemini'}), 200
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, KeyError, IndexError, ValueError, json.JSONDecodeError) as exc:
        if isinstance(exc, urllib.error.HTTPError):
            detail = f'Gemini request failed ({exc.code}).'
        else:
            detail = 'Gemini is temporarily unavailable.'
        return jsonify({'error': detail}), 502

@app.route('/camera/status')
def camera_status():
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

@app.route('/api/model_status')
def api_model_status():
    import json
    try:
        us = manager_dict.get('unique_samples', {})
    except Exception:
        us = {}
    try:
        ic = manager_dict.get('inference_counts', {})
    except Exception:
        ic = {}
        
    def _safe_get(d, key, default=0):
        try:
            return d.get(key, default)
        except Exception:
            return default

    return jsonify([
        {
          "modality": "keyboard",
          "samples_received": _safe_get(us, 'keystroke', 0),
          "unique_samples": _safe_get(us, 'keystroke', 0),
          "buffer_size": 10,
          "feature_dimension": 7,
          "model_loaded": manager_dict.get('model_status') == 'TRAINED',
          "model_inference_count": _safe_get(ic, 'keystroke', 0),
          "training_dataset": "Freihaut & Goeritz (2021)",
          "training_label_type": "stress (experimental condition)",
          "model_name": manager_dict.get('model_name', '')
        },
        {
          "modality": "speech",
          "samples_received": _safe_get(us, 'audio', 0),
          "unique_samples": _safe_get(us, 'audio', 0),
          "buffer_size": 10,
          "feature_dimension": 169,
          "model_loaded": False,
          "model_inference_count": 0,
          "training_dataset": "N/A",
          "training_label_type": "none (feature only)"
        },
        {
          "modality": "facial",
          "samples_received": _safe_get(us, 'face', 0),
          "unique_samples": _safe_get(us, 'face', 0),
          "buffer_size": 10,
          "feature_dimension": 12,
          "model_loaded": False,
          "model_inference_count": 0,
          "training_dataset": "N/A (FER-2013 emotion shown for display only, not used for stress)",
          "training_label_type": "none (feature only)"
        },
        {
          "modality": "eye/pupil",
          "samples_received": _safe_get(us, 'eye', 0),
          "unique_samples": _safe_get(us, 'eye', 0),
          "buffer_size": 10,
          "feature_dimension": 5,
          "model_loaded": False,
          "model_inference_count": 0,
          "training_dataset": "N/A",
          "training_label_type": "none (feature only)"
        },
        {
          "modality": "handwriting",
          "samples_received": _safe_get(us, 'handwriting', 0),
          "unique_samples": _safe_get(us, 'handwriting', 0),
          "buffer_size": 10,
          "feature_dimension": 9,
          "model_loaded": False,
          "model_inference_count": 0,
          "training_dataset": "N/A",
          "training_label_type": "none (feature only)"
        }
    ])

@app.route('/upload_handwriting', methods=['POST'])
def upload_handwriting():
    data = request.json
    if not manager_dict.get('is_live_monitoring'):
        return jsonify({'error': 'Monitoring is stopped.'}), 409
    if not data or 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400
    b64_str = data['image']
    if b64_str.startswith('data:image/png;base64,'):
        b64_str = b64_str.replace('data:image/png;base64,', '')
    import json as _json
    strokes = data.get('strokes')
    manager_dict['live_handwriting_strokes'] = _json.dumps(strokes) if isinstance(strokes, list) else None
    manager_dict['live_handwriting_b64'] = b64_str
    return jsonify({'status': 'success'})

def _new_session_marker():
    """Start a fresh keyboard window / prediction history (no stale values)."""
    manager_dict['keyboard_session_marker'] = time.time_ns()
    manager_dict['keyboard_window'] = []
    manager_dict['keystroke_features'] = None
    manager_dict['keyboard_pending_keystrokes'] = 0
    manager_dict['remote_keystroke_events'] = None
    manager_dict['remote_frame_b64'] = None
    manager_dict['remote_audio_b64'] = None
    manager_dict['live_handwriting_b64'] = None
    manager_dict['live_handwriting_strokes'] = None
    manager_dict['audio_features'] = None
    manager_dict['face_features'] = None
    manager_dict['eye_features'] = None
    manager_dict['handwriting_features'] = None
    manager_dict['buffer_fills'] = '{}'
    manager_dict['runtime_modality_status'] = '{}'
    manager_dict['runtime_mask'] = '[0,0,0,0,0]'
    manager_dict['fusion_mask'] = '[0,0,0,0,0]'
    manager_dict['prediction_timestamp'] = ''
    manager_dict['session_start_time'] = 0.0
    manager_dict['keyboard_status'] = 'WAITING_FOR_INPUT'
    manager_dict['mic_status'] = 'WAITING_FOR_MIC'
    manager_dict['face_status'] = 'WAITING_FOR_CAMERA'
    manager_dict['eye_status'] = 'WAITING_FOR_CAMERA'
    manager_dict['handwriting_status'] = 'WAITING_FOR_INPUT'
    for key in ('audio_sample_count', 'audio_sample_id', 'keyboard_sample_id',
                'camera_frame_count', 'face_detection_count', 'eye_detection_count',
                'face_sample_id', 'eye_sample_id', 'handwriting_submission_count',
                'handwriting_sample_id'):
        manager_dict[key] = 0
    manager_dict['predictions_available'] = False
    manager_dict['fusion_pred'] = -1
    manager_dict['prediction_history'] = '[]'
    manager_dict['raw_stress_prob'] = None
    manager_dict['raw_nonstress_prob'] = None
    manager_dict['smoothed_stress_prob'] = None
    manager_dict['smoothed_nonstress_prob'] = None
    manager_dict['confidence'] = None


def _stop_session_state():
    manager_dict['is_live_monitoring'] = False
    manager_dict['keyboard_window'] = []
    manager_dict['status'] = 'READY'
    manager_dict['remote_keystroke_events'] = None
    manager_dict['remote_frame_b64'] = None
    manager_dict['remote_audio_b64'] = None
    manager_dict['live_handwriting_b64'] = None
    manager_dict['live_handwriting_strokes'] = None
    manager_dict['audio_features'] = None
    manager_dict['face_features'] = None
    manager_dict['eye_features'] = None
    manager_dict['handwriting_features'] = None
    manager_dict['prediction_timestamp'] = ''
    manager_dict['predictions_available'] = False
    manager_dict['fusion_prob'] = None
    manager_dict['fusion_pred'] = -1
    manager_dict['raw_stress_prob'] = None
    manager_dict['raw_nonstress_prob'] = None
    manager_dict['smoothed_stress_prob'] = None
    manager_dict['smoothed_nonstress_prob'] = None
    manager_dict['confidence'] = None
    manager_dict['prediction_history'] = '[]'
    manager_dict['buffer_fills'] = '{}'
    manager_dict['runtime_modality_status'] = '{}'
    manager_dict['runtime_mask'] = '[0,0,0,0,0]'
    manager_dict['fusion_mask'] = '[0,0,0,0,0]'
    manager_dict['camera_connected'] = False
    manager_dict['face_detected'] = False
    manager_dict['face_bbox'] = None
    manager_dict['keyboard_status'] = 'WAITING_FOR_INPUT'
    manager_dict['mic_status'] = 'WAITING_FOR_MIC'
    manager_dict['face_status'] = 'WAITING_FOR_CAMERA'
    manager_dict['eye_status'] = 'WAITING_FOR_CAMERA'
    manager_dict['handwriting_status'] = 'WAITING_FOR_INPUT'
    manager_dict['session_start_time'] = 0.0


def _finalize_session_report():
    import json
    session_id = str(uuid.uuid4())[:8]
    report_filename = f"stress_session_{session_id}.pdf"
    os.makedirs(os.path.join(os.path.dirname(__file__), 'reports'), exist_ok=True)
    report_path = os.path.join(os.path.dirname(__file__), 'reports', report_filename)
    
    start_ts = manager_dict.get('session_start_time', 0.0)
    end_ts = time.time()
    duration_s = int(end_ts - start_ts) if start_ts > 0 else 0
    duration_str = f"{duration_s // 60}m {duration_s % 60}s"
    
    # Calculate averages from timeline
    timeline = json.loads(manager_dict.get('prediction_history', '[]'))
    stress_preds = [x for x in timeline if x.get('camera_prediction') == 'STRESS']
    nonstress_preds = [x for x in timeline if x.get('camera_prediction') == 'NON-STRESS']
    cam_probs = [x.get('camera_stress_prob_pct', 0) for x in timeline if 'camera_stress_prob_pct' in x]
    avg_stress_prob = sum(cam_probs) / len(cam_probs) if cam_probs else 0.0
    peak_stress_prob = max(cam_probs, default=0.0)
    
    final_pred = 'INSUFFICIENT DATA'
    if manager_dict.get('camera_predictions_available'):
        final_pred = 'STRESS' if manager_dict.get('camera_pred') == 1 else 'NON-STRESS'
    
    try:
        mod_status = json.loads(manager_dict.get('runtime_modality_status', '{}'))
    except:
        mod_status = {}
        
    def _ensure_historical(mod_name, count_key, max_fill, role):
        if mod_name not in mod_status:
            mod_status[mod_name] = {}
        c = manager_dict.get(count_key, 0)
        if c > 0:
            mod_status[mod_name]['buffer_fill'] = min(c, max_fill)
            if mod_status[mod_name].get('status', 'NO INPUT') in ('NO INPUT', 'WAITING'):
                mod_status[mod_name]['status'] = 'ACTIVE'
            mod_status[mod_name]['features_ready'] = True
            mod_status[mod_name]['role'] = role

    _ensure_historical('keyboard', 'keyboard_event_count', 10, 'STRESS MODEL INPUT')
    _ensure_historical('speech', 'audio_sample_count', 10, 'EMOTION-DERIVED PROXY INPUT')
    _ensure_historical('facial', 'face_detection_count', 10, 'FEATURE EXTRACTION ONLY')
    _ensure_historical('eye_pupil', 'eye_detection_count', 10, 'FEATURE EXTRACTION ONLY')
    _ensure_historical('handwriting', 'handwriting_submission_count', 10, 'FEATURE EXTRACTION ONLY')
        
    session_data = {
        'session_id': session_id,
        'participant': manager_dict.get('participant_details', {}),
        'date': datetime.datetime.now().strftime("%Y-%m-%d"),
        'start_time': datetime.datetime.fromtimestamp(start_ts).strftime("%H:%M:%S") if start_ts else "N/A",
        'end_time': datetime.datetime.fromtimestamp(end_ts).strftime("%H:%M:%S"),
        'duration': duration_str,
        'final_prediction': final_pred,
        'stress_probability': (manager_dict.get('camera_stress_prob') or 0.0) * 100,
        'nonstress_probability': (manager_dict.get('camera_nonstress_prob') or 0.0) * 100,
        'confidence': (manager_dict.get('camera_confidence') or 0.0) * 100,
        'prediction_source': 'CAMERA STRESS ESTIMATE (RESEARCH / NOT CLINICALLY VALIDATED)',

        
        'secondary_prediction': 'STRESS' if manager_dict.get('keyboard_pred') == 1 else ('NON-STRESS' if manager_dict.get('keyboard_predictions_available') else 'WAITING'),
        'secondary_stress_prob': (manager_dict.get('keyboard_stress_prob') or 0.0) * 100,
        'secondary_confidence': (manager_dict.get('keyboard_confidence') or 0.0) * 100,

        'valid_predictions_count': len([x for x in timeline if 'camera_prediction' in x]),
        'stress_count': len(stress_preds),
        'nonstress_count': len(nonstress_preds),
        'avg_stress_prob': avg_stress_prob,
        'peak_stress_prob': peak_stress_prob,
        'avg_confidence': max(avg_stress_prob, 100.0 - avg_stress_prob) if timeline else 0.0,
        'modalities': mod_status,
        'timeline': timeline,
        'data_quality': {
            'camera_connected': manager_dict.get('camera_connected', False),
            'failed_cycles': manager_dict.get('failed_cycles', 0)
        },
        'model_name': manager_dict.get('model_name', 'N/A')
    }
    
    # Augment modalities with specific counts
    if 'facial' not in session_data['modalities']:
        session_data['modalities']['facial'] = {}
    session_data['modalities']['facial']['face_detection_count'] = manager_dict.get('face_detection_count', 0)
    session_data['modalities']['facial']['latest_expression'] = manager_dict.get('face_emotion_label', 'N/A')
    session_data['modalities']['facial']['expression_confidence'] = "N/A"
    
    generate_session_pdf(session_data, report_path)
    return f"/reports/{report_filename}", session_data


@app.route('/api/control', methods=['POST'])
def api_control():
    data = request.json
    action = data.get('action')
    report_url = None
    session_info = None
    if action == 'start':
        if not manager_dict['is_live_monitoring']:
            _new_session_marker()
            manager_dict['session_start_time'] = time.time()
            manager_dict['is_live_monitoring'] = True
            manager_dict['status'] = 'LIVE'
            manager_dict['participant_details'] = data.get('participant', {})
    elif action == 'stop':
        if manager_dict['is_live_monitoring']:
            try:
                report_url, session_info = _finalize_session_report()
            except Exception as e:
                import traceback
                return jsonify({'status': 'error', 'error': str(e), 'traceback': traceback.format_exc()})
        _stop_session_state()
    elif action == 'reset':
        _stop_session_state()
        _new_session_marker()
        manager_dict['total_cycles'] = 0
        manager_dict['successful_cycles'] = 0
        manager_dict['failed_cycles'] = 0
        manager_dict['dropped_cycles'] = 0
        manager_dict['session_start_time'] = 0.0
        manager_dict['status'] = 'READY'
    elif action == 'clear_handwriting':
        manager_dict['handwriting_status'] = 'WAITING_FOR_INPUT'
        manager_dict['handwriting_clear_flag'] = True
    
    resp = {'status': 'success', 'is_live_monitoring': manager_dict['is_live_monitoring']}
    if report_url:
        resp['report_url'] = report_url
        resp['session_info'] = session_info
    return jsonify(resp)

@app.route('/reports/<filename>')
def serve_report(filename):
    return send_from_directory(os.path.join(os.path.dirname(__file__), 'reports'), filename)

@app.route('/api/upload_frame', methods=['POST'])
def api_upload_frame():
    data = request.json
    if manager_dict.get('is_live_monitoring') and data and 'image' in data:
        manager_dict['camera_connected'] = True
        manager_dict['remote_frame_b64'] = data['image']
        manager_dict['last_frame_timestamp'] = time.time()
    return jsonify({"status": "ok"})

@app.route('/api/upload_audio', methods=['POST'])
def api_upload_audio():
    data = request.json
    if manager_dict.get('is_live_monitoring') and data and 'audio' in data:
        manager_dict['mic_status'] = 'RECEIVING_AUDIO'
        manager_dict['audio_last_upload_time'] = time.time()
        manager_dict['remote_audio_b64'] = data['audio']
    return jsonify({"status": "ok"})

@app.route('/api/media_status', methods=['POST'])
def api_media_status():
    data = request.json or {}
    if not manager_dict.get('is_live_monitoring'):
        return jsonify({'status': 'ignored'}), 409
    if data.get('camera') == 'UNAVAILABLE':
        main_shared_state['camera_connected'] = False
        main_shared_state['face_status'] = 'HARDWARE_UNAVAILABLE'
        main_shared_state['eye_status'] = 'HARDWARE_UNAVAILABLE'
    if data.get('microphone') == 'UNAVAILABLE':
        main_shared_state['mic_status'] = 'HARDWARE_UNAVAILABLE'
    return jsonify({'status': 'ok'})

@app.route('/api/upload_keystrokes', methods=['POST'])
def api_upload_keystrokes():
    data = request.json
    if manager_dict.get('is_live_monitoring') and data and 'events' in data:
        print(f"KEYBOARD_EVENT | event_count={len(data['events'])}", flush=True)
        import json
        with KEY_EVENTS_LOCK:
            pending = main_shared_state.get('remote_keystroke_events')
            events = (json.loads(pending) if pending else []) + list(data['events'])
            main_shared_state['remote_keystroke_events'] = json.dumps(events)
    return jsonify({"status": "ok"})

@app.route('/api/debug/live')
def api_debug_live():
    return jsonify({"error": "Forbidden"}), 403


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

@app.route('/health')
def health_check():
    return jsonify({"status": "ok", "message": "Healthy"})

if __name__ == '__main__':
    
    
    
    main_shared_state = dict({
        'running': True,
        'audio_features': None,
        'face_features': None,
        'keystroke_features': None,
        'handwriting_features': None,
        'eye_features': None,
        'fusion_prob': None,
        'fusion_pred': -1,
        'modalities': dict(),
        'latency': dict(),
        'model_status': 'LOADING',
        'checkpoint_name': '',
        'face_status': 'PENDING',
        'eye_status': 'PENDING',
        'mic_status': 'PENDING',
        'keyboard_status': 'PENDING',
        'handwriting_status': 'WAITING_FOR_INPUT',
        'live_handwriting_b64': None,
        'live_handwriting_strokes': None,
        'latest_frame_jpg': None,
        'keyboard_window': [],
        'keyboard_session_marker': 0.0,
        'keyboard_pending_keystrokes': 0,
        'audio_last_upload_time': 0.0,
        'predictions_available': False,
        # Prediction output fields
        'raw_stress_prob': None,
        'raw_nonstress_prob': None,
        'smoothed_stress_prob': None,
        'smoothed_nonstress_prob': None,
        'confidence': None,
        'modality_weights': '{}',
        'reliabilities': '{}',
        'unavailable_modalities': '[]',
        'latency_percentiles': '{}',
        'prediction_timestamp': '',
        'prediction_history': '[]',
        'fusion_latency_ms': 0.0,
        'fusion_mask': '[0,0,0,0,0]',
        'runtime_mask': '[0,0,0,0,0]',
        'buffer_fills': '{}',
        'runtime_modality_status': '{}',
        # Diagnostic counters
        'keyboard_event_count': 0,
        'audio_sample_count': 0,
        'audio_sample_id': 0,
        'audio_last_sample_time': 0.0,
        'keyboard_sample_id': 0,
        'keyboard_last_sample_time': 0.0,
        'camera_frame_count': 0,
        'face_detection_count': 0,
        'eye_detection_count': 0,
        'face_sample_id': 0,
        'face_last_sample_time': 0.0,
        'eye_sample_id': 0,
        'eye_last_sample_time': 0.0,
        'face_fer_inference_count': 0,
        'handwriting_submission_count': 0,
        'handwriting_sample_id': 0,
        'handwriting_last_sample_time': 0.0,
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
        threading.Thread(target=keyboard_worker, args=(main_shared_state,), daemon=True),
        threading.Thread(target=audio_worker, args=(main_shared_state,), daemon=True),
        threading.Thread(target=webcam_worker, args=(main_shared_state,), daemon=True),
        threading.Thread(target=handwriting_worker, args=(main_shared_state,), daemon=True),
        threading.Thread(target=inference_worker, args=(main_shared_state,), daemon=True)
    ]
    
    for w in workers:
        w.start()
        
    print("========================================")
    print("RA-HMSD REAL-TIME MULTIMODAL SERVER")
    print("========================================")
    print(f"Dashboard URL: http://localhost:{os.environ.get('PORT', 5000)}")
    print(f"API Endpoint:  http://localhost:{os.environ.get('PORT', 5000)}/status")
    print("========================================\n")
    
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    
    try:
        cache_thread = threading.Thread(target=update_local_cache, daemon=True)
        cache_thread.start()
        app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False, use_reloader=False, threaded=True)
    except KeyboardInterrupt:
        pass
    finally:
        main_shared_state['running'] = False
        for w in workers:
            
            w.join()

