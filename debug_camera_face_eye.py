import cv2
import time
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.face_utils import extract_face_features
    from src.eye_utils import extract_eye_features
except ImportError as e:
    print(f"Error importing extractors: {e}")
    sys.exit(1)

def run_diagnostic():
    print("==========================================")
    print("CAMERA AND EXTRACTOR DIAGNOSTIC")
    print("==========================================")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[FAIL] CAMERA: Could not open cv2.VideoCapture(0)")
        return
    print("[PASS] CAMERA: Successfully opened cv2.VideoCapture(0)")
    
    # Read a few frames to let the camera warm up
    for _ in range(5):
        cap.read()
        time.sleep(0.1)
        
    ret, frame = cap.read()
    if not ret or frame is None or frame.size == 0:
        print("[FAIL] LIVE FRAME: Could not read a valid frame from the camera.")
        return
        
    h, w, c = frame.shape
    print(f"[PASS] LIVE FRAME: Read successfully. Shape: {w}x{h}x{c}")
    
    frame_small = cv2.resize(frame, (320, 240))
    print(f"       Resized for extraction to: 320x240")
    
    # 1. Test Facial Extractor
    print("\n--- FACIAL DETECTOR ---")
    try:
        t0 = time.perf_counter()
        face_feats = extract_face_features(frame_small)
        t_face = (time.perf_counter() - t0) * 1000
        
        if np.count_nonzero(face_feats) > 0:
            print(f"[PASS] FACIAL DETECTOR: Found face landmarks.")
            print(f"[PASS] FACIAL FEATURES: Shape {face_feats.shape}, Latency {t_face:.1f} ms")
            print(f"       Values: {face_feats}")
        else:
            print(f"[FAIL] FACIAL DETECTOR: No face detected in frame. (Is a human face visible?)")
            print(f"       Returned: {face_feats}")
    except Exception as e:
        print(f"[FAIL] FACIAL DETECTOR: Exception occurred: {e}")
        
    # 2. Test Eye/Pupil Extractor
    print("\n--- EYE DETECTOR ---")
    try:
        t0 = time.perf_counter()
        eye_feats = extract_eye_features(frame_small)
        t_eye = (time.perf_counter() - t0) * 1000
        
        if np.count_nonzero(eye_feats) > 0:
            print(f"[PASS] EYE DETECTOR: Found eye landmarks.")
            print(f"[PASS] EYE FEATURES: Shape {eye_feats.shape}, Latency {t_eye:.1f} ms")
            print(f"       Values: {eye_feats}")
        else:
            print(f"[FAIL] EYE DETECTOR: No eyes detected in frame. (Is a human face visible?)")
            print(f"       Returned: {eye_feats}")
    except Exception as e:
        print(f"[FAIL] EYE DETECTOR: Exception occurred: {e}")

    print("\n==========================================")
    print("If you are a human sitting in front of the camera, and FACIAL or EYE fails,")
    print("the issue is with MediaPipe detection (lighting, distance) or the camera feed itself.")
    print("==========================================")
    
    cap.release()

if __name__ == '__main__':
    run_diagnostic()
