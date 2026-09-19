"""
Multimodal Real-Time Stress Detection Server
=============================================
Fuses inputs from Webcam (Eye + Face Emotion), Microphone (Audio), 
and Keyboard (Keystroke Dynamics) to predict stress in real-time.

Usage:
    python scripts/realtime/multimodal_stress_server.py
"""

import os
import sys
import time
import threading
import numpy as np
import cv2
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from DL_models import get_model
from audio_utils import extract_audio_features
from keystroke_utils import extract_keystroke_features

from flask import Flask, Response, jsonify, render_template

try:
    import pyaudio
except ImportError:
    pyaudio = None
    
try:
    from pynput import keyboard
except ImportError:
    keyboard = None

app = Flask(__name__, template_folder='templates')

# --- Shared State ---
state = {
    'stress_prob': 0.0,
    'pred_label': 0,
    'audio_features': np.zeros(128),
    'keystroke_events': [],
    'face_features': np.zeros((48, 48, 1)),
    'running': True,
    'lock': threading.Lock()
}

# --- Dummy Weights / Initialization ---
print("[INFO] Initializing Multimodal Fusion Model...")
input_shapes = {
    'audio': (166,),
    'physio': (12,),
    'face': (48, 48, 1),
    'keystroke': (5,),
    'handwriting': (6,)
}
# Load model
params = {}
model = get_model('multimodal_fusion', input_shapes, 0, 2, params)

try:
    model.load_weights("results/fusion/best_fusion_model.h5")
    print("[INFO] Loaded pre-trained fusion model weights.")
except Exception as e:
    print(f"[WARN] Could not load weights: {e}. Using random initialization.")

print("[INFO] Model ready.")

# --- Keyboard Listener Thread ---
def keyboard_listener():
    if keyboard is None:
        print("[WARN] pynput not installed. Keyboard tracking disabled.")
        return
        
    def on_press(key):
        with state['lock']:
            state['keystroke_events'].append({'key': str(key), 'action': 'press', 'time': time.time()})
            
    def on_release(key):
        with state['lock']:
            state['keystroke_events'].append({'key': str(key), 'action': 'release', 'time': time.time()})

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

# --- Audio Listener Thread ---
def audio_listener():
    if pyaudio is None:
        print("[WARN] pyaudio not installed. Audio tracking disabled.")
        return
        
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 22050
    RECORD_SECONDS = 3
    
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    
    while state['running']:
        frames = []
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)
            except Exception:
                pass
                
        # Convert to numpy array
        audio_data = np.frombuffer(b''.join(frames), dtype=np.int16).astype(np.float32)
        features = extract_audio_features(audio_data, sr=RATE)
        
        # Pad or truncate to match model input (128,)
        if len(features) > 128:
            features = features[:128]
        elif len(features) < 128:
            features = np.pad(features, (0, 128 - len(features)))
            
        with state['lock']:
            state['audio_features'] = features

# --- Webcam Listener Thread ---
def webcam_listener():
    cap = cv2.VideoCapture(0)
    
    # Use Haar Cascade as a simple face detector (fast for real-time)
    cascade_path = os.path.join(os.path.dirname(__file__), 'haarcascade_frontalface_default.xml')
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    while state['running']:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.033)
            continue
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # Take the largest face
        if len(faces) > 0:
            faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
            x, y, w, h = faces[0]
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            
            # Normalize for network
            face_img = roi_gray.astype(np.float32) / 255.0
            face_img = np.expand_dims(face_img, axis=-1) # (48, 48, 1)
            
            with state['lock']:
                state['face_features'] = face_img
        else:
            with state['lock']:
                state['face_features'] = np.zeros((48, 48, 1))

# --- Inference Loop ---
def inference_loop():
    """ Runs inference at 1Hz fusing all available modalities. """
    while state['running']:
        time.sleep(1.0)
        with state['lock']:
            # Extract keystroke features from the recent window
            recent_events = [e for e in state['keystroke_events'] if time.time() - e['time'] < 10]
            kf = extract_keystroke_features(recent_events)
            key_feats = np.array([kf['dwell_mean'], kf['dwell_std'], kf['flight_mean'], kf['flight_std'], kf['typing_speed']])
            
            aud_feats = state['audio_features']
            fac_feats = state['face_features']
            
            # Pad or trim audio to exact 166
            if len(aud_feats) < 166:
                aud_feats = np.pad(aud_feats, (0, 166 - len(aud_feats)))
            elif len(aud_feats) > 166:
                aud_feats = aud_feats[:166]
            
            # Predict
            try:
                preds = model.predict({
                    'audio_input': np.expand_dims(aud_feats, axis=0),
                    'physio_input': np.expand_dims(np.zeros(12), axis=0),
                    'face_input': np.expand_dims(fac_feats, axis=0),
                    'keystroke_input': np.expand_dims(key_feats, axis=0),
                    'handwriting_input': np.expand_dims(np.zeros(6), axis=0)
                }, verbose=0)[0]
                
                state['stress_prob'] = float(preds[1])
                state['pred_label'] = int(np.argmax(preds))
            except Exception as e:
                print(f"[ERROR] Inference failed: {e}")
            
            # Cleanup old keystrokes
            state['keystroke_events'] = recent_events

# --- Routes ---
@app.route('/data')
def data():
    with state['lock']:
        return jsonify({
            'stress_prob': round(state['stress_prob'] * 100, 1),
            'pred': state['pred_label'],
            'modalities_active': {
                'audio': pyaudio is not None,
                'keyboard': keyboard is not None,
                'webcam': True
            }
        })

if __name__ == '__main__':
    threading.Thread(target=keyboard_listener, daemon=True).start()
    threading.Thread(target=audio_listener, daemon=True).start()
    threading.Thread(target=webcam_listener, daemon=True).start()
    threading.Thread(target=inference_loop, daemon=True).start()
    
    print("\n" + "=" * 55)
    print("  Multimodal Real-Time Stress Detection Server")
    print("  Serving at http://localhost:5001")
    print("=" * 55 + "\n")
    app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)
