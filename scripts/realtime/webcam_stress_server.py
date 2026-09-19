"""
Real-Time Webcam Stress Detection Server
==========================================
Uses your webcam + MediaPipe Face Mesh to detect iris size in real-time,
feeds it into the trained CNN model, and streams predictions to a web dashboard.

Usage:
    python scripts/realtime/webcam_stress_server.py

Then open: http://localhost:5000
"""

import os
import sys
import json
import time
import threading
import numpy as np
import cv2
import base64
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from flask import Flask, render_template, Response, jsonify
from DL_models import cnn_model
from collections import deque
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

# ── Config ───────────────────────────────────────────────────────────────────
WEIGHTS_PATH  = "results/local_smoke/vr_goalkeeper/DL/local-smoke-test/CNN/smoke_vr_cnn_pd/test_ID_0/best_weights_foldnan.weights.h5"
TIMESTEPS     = 450       # samples the CNN expects
BUFFER_SECS   = 5.0       # collect 5s of data before first prediction
PREDICT_EVERY = 15        # re-predict every N new frames
WEBCAM_IDX    = 0

BEST_PARAMS = {
    'filters1': 16, 'filters2': 16, 'filters3': 16,
    'k_size': 2, 'p_size': 2,
    'drop1': 0.1, 'drop2': 0.1, 'drop3': 0.1,
    'dense': 16
}

# MediaPipe iris landmark indices (left iris: 468-471, right iris: 472-475)
LEFT_IRIS  = [468, 469, 470, 471]
RIGHT_IRIS = [472, 473, 474, 475]

app = Flask(__name__, template_folder='templates')

# ── Load CNN model ────────────────────────────────────────────────────────────
print("[INFO] Loading CNN model...")
model = cnn_model(data_dim=1, timesteps=TIMESTEPS, num_classes=2, params=BEST_PARAMS)
model.load_weights(WEIGHTS_PATH)
print("[INFO] Model ready.")

# ── Shared state ──────────────────────────────────────────────────────────────
state = {
    'frame_jpg':    None,        # latest JPEG frame bytes
    'iris_size':    0.0,         # latest iris diameter (pixels)
    'stress_prob':  0.0,         # latest stress probability
    'pred_label':   0,           # 0=calm, 1=stress
    'buffer':       deque(maxlen=int(30 * 15)),  # 15s at 30fps
    'frame_count':  0,
    'fps':          0.0,
    'face_found':   False,
    'running':      True,
    'history':      deque(maxlen=100),
    'lock':         threading.Lock()
}

# ── Iris size → pupil diameter (pixels) ──────────────────────────────────────
def get_iris_diameter(landmarks, iris_indices, img_w, img_h):
    """Return diameter of iris in pixels from 4 landmark points."""
    pts = []
    for idx in iris_indices:
        lm = landmarks[idx]
        pts.append((lm.x * img_w, lm.y * img_h))
    pts = np.array(pts)
    # diameter = max distance between any two points on the iris ring
    diam = np.linalg.norm(pts[0] - pts[2])  # top-bottom
    return float(diam)

# ── Preprocess buffer → model input ──────────────────────────────────────────
def buffer_to_model_input(buf):
    """Resample buffer to TIMESTEPS, z-score normalize, return (1,450,1) array."""
    arr = np.array(buf, dtype=np.float32)
    if len(arr) < 10:
        return None
    # Resample to TIMESTEPS using linear interpolation
    x_old = np.linspace(0, 1, len(arr))
    x_new = np.linspace(0, 1, TIMESTEPS)
    f     = interp1d(x_old, arr, kind='linear')
    resampled = f(x_new)
    # Smooth (like preprocessing pipeline)
    if len(resampled) > 11:
        resampled = savgol_filter(resampled, window_length=11, polyorder=2)
    # Z-score normalize
    mu, sigma = resampled.mean(), resampled.std()
    if sigma < 1e-6:
        return None
    normalized = (resampled - mu) / sigma
    return normalized.reshape(1, TIMESTEPS, 1).astype(np.float32)

# ── Webcam capture thread ─────────────────────────────────────────────────────
def webcam_thread():
    """
    Uses MediaPipe FaceMesh (0.10.9) if available, otherwise falls back to
    OpenCV Haar cascades for eye detection. Measures iris/eye diameter as
    a proxy for pupil dilation and feeds it into the CNN stress classifier.
    """
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh

    cap = cv2.VideoCapture(WEBCAM_IDX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    prev_time = time.time()
    frame_idx = 0

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,   # enables iris landmarks 468-477
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        while state['running']:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.033)
                continue

            h, w = frame.shape[:2]
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            iris_diam  = 0.0
            face_found = False

            if results.multi_face_landmarks:
                lms        = results.multi_face_landmarks[0].landmark
                face_found = True

                # Get iris diameter (average left + right)
                l_diam = get_iris_diameter(lms, LEFT_IRIS,  w, h)
                r_diam = get_iris_diameter(lms, RIGHT_IRIS, w, h)
                iris_diam = (l_diam + r_diam) / 2.0

                # ── Only add to buffer when face is present ──
                state['buffer'].append(iris_diam)
                frame_idx += 1

                # ── Run model only when face is present + buffer full ──
                buf    = list(state['buffer'])
                enough = len(buf) >= int(30 * BUFFER_SECS)
                if enough and frame_idx % PREDICT_EVERY == 0:
                    x = buffer_to_model_input(buf)
                    if x is not None:
                        probs       = model.predict(x, verbose=0)[0]
                        stress_prob = float(probs[1])
                        pred_label  = int(np.argmax(probs))
                        with state['lock']:
                            state['stress_prob'] = stress_prob
                            state['pred_label']  = pred_label
                            state['face_found']  = True
                            state['history'].append({
                                'sp':   round(stress_prob * 100, 1),
                                'pred': pred_label,
                                'ts':   round(time.time(), 2)
                            })
            else:
                # ── NO FACE: reset prediction state completely ──
                with state['lock']:
                    state['stress_prob'] = 0.0
                    state['pred_label']  = -1   # -1 = no prediction
                    state['face_found']  = False
                # Clear buffer so stale data doesn't carry over
                state['buffer'].clear()
                frame_idx = 0

            # Read current prediction for HUD (only valid if face found)
            stress_prob = state['stress_prob']
            pred_label  = state['pred_label']

            # Draw iris landmarks + eye box only when face found
            if face_found:
                for idx in LEFT_IRIS + RIGHT_IRIS:
                    lm = lms[idx]
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 2, (0, 255, 200), -1)

                eye_lms = [33, 133, 362, 263]
                ex = [int(lms[i].x * w) for i in eye_lms]
                ey = [int(lms[i].y * h) for i in eye_lms]
                x1, y1 = min(ex) - 20, min(ey) - 20
                x2, y2 = max(ex) + 20, max(ey) + 20
                box_color = (0, 70, 255) if pred_label == 1 else (0, 210, 90)
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

            # ── Overlay HUD ──
            cv2.rectangle(frame, (0, 0), (w, 55), (15, 15, 25), -1)
            cv2.rectangle(frame, (0, h - 55), (w, h), (15, 15, 25), -1)

            if face_found:
                sp_pct = stress_prob * 100
                buf_pct = min(100, int(len(state['buffer']) / (30 * BUFFER_SECS) * 100))
                ready   = buf_pct >= 100

                if ready and pred_label >= 0:
                    color = (0, 70, 255) if pred_label == 1 else (0, 210, 90)
                    label = "STRESSED" if pred_label == 1 else "CALM"
                    # Stress bar
                    bar_w = int(w * sp_pct / 100)
                    cv2.rectangle(frame, (0, h - 10), (bar_w, h), color, -1)
                    cv2.putText(frame, f"Stress: {sp_pct:.1f}%  [{label}]",
                                (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)
                else:
                    pct_text = f"Calibrating... {buf_pct}%"
                    cv2.putText(frame, pct_text,
                                (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 180, 255), 2)

                cv2.putText(frame, f"FACE DETECTED  Iris: {iris_diam:.1f}px  FPS:{state['fps']:.0f}",
                            (12, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 220, 100), 1)
            else:
                # No face — show scanning message, no stress display
                cv2.putText(frame, "NO FACE DETECTED — Position your face in frame",
                            (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 160, 255), 2)
                cv2.putText(frame, f"FPS: {state['fps']:.0f}",
                        (12, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 180), 1)

            if not face_found:
                cv2.putText(frame, "No face detected",
                            (w // 2 - 100, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 80, 255), 2)

            if len(state['buffer']) < int(30 * BUFFER_SECS):
                pct = int(len(state['buffer']) / (30 * BUFFER_SECS) * 100)
                cv2.putText(frame, f"Calibrating... {pct}%",
                            (w // 2 - 80, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)

            # FPS
            now = time.time()
            state['fps'] = 1.0 / max(now - prev_time, 0.001)
            prev_time = now

            # Encode JPEG
            _, jpg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            with state['lock']:
                state['frame_jpg']   = jpg.tobytes()
                state['iris_size']   = iris_diam
                state['face_found']  = face_found
                state['frame_count'] += 1

    cap.release()


# ── Start webcam thread ───────────────────────────────────────────────────────
t = threading.Thread(target=webcam_thread, daemon=True)
t.start()

# ── Flask routes ──────────────────────────────────────────────────────────────
def gen_frames():
    while True:
        time.sleep(0.033)
        with state['lock']:
            frame = state['frame_jpg']
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('webcam_dashboard.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/data')
def data():
    with state['lock']:
        return jsonify({
            'stress_prob':  round(state['stress_prob'] * 100, 1),
            'pred':         state['pred_label'],
            'iris_size':    round(state['iris_size'], 2),
            'face_found':   state['face_found'],
            'fps':          round(state['fps'], 1),
            'buf_pct':      min(100, int(len(state['buffer']) / (30 * BUFFER_SECS) * 100)),
            'history':      list(state['history'])[-50:]
        })

@app.route('/control/<action>')
def control(action):
    state['running'] = (action == 'start')
    return jsonify({'running': state['running']})

if __name__ == '__main__':
    print("\n" + "=" * 55)
    print("  Real-Time Webcam Stress Detection")
    print("  Open your browser at: http://localhost:5000")
    print("=" * 55 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
