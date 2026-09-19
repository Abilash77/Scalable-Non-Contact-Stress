"""
Real-Time Stress Detection Server
==================================
Loads the trained CNN model and streams live stress predictions
from the VR goalkeeper eye-tracking dataset via a web dashboard.

Usage:
    python scripts/realtime/realtime_stress_server.py

Then open: http://localhost:5000
"""

import os
import sys
import json
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from flask import Flask, render_template, Response, jsonify
import tensorflow as tf
from DL_models import cnn_model
from sklearn.preprocessing import StandardScaler

app = Flask(__name__, template_folder='templates')

# ── Config ──────────────────────────────────────────────────────────────────
WEIGHTS_PATH = "results/local_smoke/vr_goalkeeper/DL/local-smoke-test/CNN/smoke_vr_cnn_pd/test_ID_0/best_weights_foldnan.weights.h5"
DATA_PATH    = "data/vr_goalkeeper/dataframes/DL_out.pkl"
INPUT_COL    = "meanDia_corrected"
TIMESTEPS    = 450
N_CLASSES    = 2
STREAM_DELAY = 0.3

BEST_PARAMS = {
    'filters1': 16, 'filters2': 16, 'filters3': 16,
    'k_size': 2,    'p_size': 2,
    'drop1': 0.1,   'drop2': 0.1,   'drop3': 0.1,
    'dense': 16
}

# ── Load model & data ────────────────────────────────────────────────────────
print("[INFO] Loading CNN model...")
model = cnn_model(data_dim=1, timesteps=TIMESTEPS, num_classes=N_CLASSES, params=BEST_PARAMS)
model.load_weights(WEIGHTS_PATH)
print("[INFO] Model loaded successfully.")

print("[INFO] Loading dataset...")
with open(DATA_PATH, 'rb') as f:
    df_all = pickle.load(f)

subject_df = df_all[df_all['ID'] == 0].copy().reset_index(drop=True)

# Each row is a segment; meanDia_corrected is a pd.Series of 450 samples
segments = []
labels   = []
for _, row in subject_df.iterrows():
    seg = np.array(row[INPUT_COL].values, dtype=np.float32)
    lbl = int(row['lab_num'])
    segments.append(seg)
    labels.append(lbl)

segments = np.array(segments)   # shape: (n_shots, 450)
labels   = np.array(labels)
print(f"[INFO] Subject 0: {len(segments)} segments x {segments.shape[1]} samples")

# Standardize each segment (same as training)
scaler = StandardScaler()
flat   = segments.reshape(-1, 1)
scaler.fit(flat)
segments_scaled = scaler.transform(flat).reshape(segments.shape)

state = {'running': True}

# ── SSE Stream ───────────────────────────────────────────────────────────────
def generate_predictions():
    idx   = 0
    total = len(segments_scaled)

    while True:
        if not state['running']:
            time.sleep(0.2)
            yield f"data: {json.dumps({'paused': True})}\n\n"
            continue

        if idx >= total:
            idx = 0   # loop all segments

        seg        = segments_scaled[idx].reshape(1, TIMESTEPS, 1)
        probs      = model.predict(seg, verbose=0)[0]
        stress_prob = float(probs[1])
        pred_label  = int(np.argmax(probs))
        true_label  = int(labels[idx])
        raw_signal  = segments[idx].tolist()[::10]  # downsample 450→45 pts

        payload = {
            'pos':         idx,
            'total':       total,
            'stress_prob': round(stress_prob * 100, 1),
            'pred':        pred_label,
            'true':        true_label,
            'signal':      raw_signal,
            'ts':          round(time.time(), 2),
            'paused':      False
        }

        yield f"data: {json.dumps(payload)}\n\n"
        idx  += 1
        time.sleep(STREAM_DELAY)


# ── Routes ───────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/stream')
def stream():
    return Response(generate_predictions(),
                    mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

@app.route('/control/<action>')
def control(action):
    if action == 'start':
        state['running'] = True
    elif action == 'pause':
        state['running'] = False
    return jsonify({'running': state['running']})

@app.route('/info')
def info():
    return jsonify({
        'model':     'CNN (1D Convolution)',
        'input':     INPUT_COL,
        'timesteps': TIMESTEPS,
        'dataset':   'VR Goalkeeper — Subject 0',
        'weights':   WEIGHTS_PATH
    })

if __name__ == '__main__':
    print("\n" + "="*50)
    print("  Real-Time Stress Detection Dashboard")
    print("  Open: http://localhost:5000")
    print("="*50 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
