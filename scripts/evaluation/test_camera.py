import cv2
import time
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.face_utils import extract_face_features

def test_camera():
    print("========================================")
    print("CAMERA HARDWARE & FACE DETECTION TEST")
    print("========================================\n")
    
    cap = None
    camera_idx = -1
    for i in range(4):
        print(f"Trying camera index {i}...")
        temp_cap = cv2.VideoCapture(i)
        if temp_cap.isOpened():
            ret, _ = temp_cap.read()
            if ret:
                cap = temp_cap
                camera_idx = i
                break
            temp_cap.release()
            
    if cap is None:
        print("camera opened: NO")
        print("PHYSICAL CAMERA TEST REQUIRED")
        return
        
    print("camera opened: YES")
    print(f"Using camera index: {camera_idx}")
    
    # Warmup camera
    for _ in range(10):
        cap.read()
        time.sleep(0.033)
        
    total_frames = 900
    frames_received = 0
    faces_detected = 0
    latencies = []
    
    start_time = time.time()
    
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            continue
            
        frames_received += 1
        frame_small = cv2.resize(frame, (320, 240))
        
        t0 = time.perf_counter()
        feats, status, meta = extract_face_features(frame_small)
        lat = (time.perf_counter() - t0) * 1000
        latencies.append(lat)
        
        if status == "DETECTED":
            faces_detected += 1
            
        # Optional: visual feedback
        # cv2.imshow('test', frame_small)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        time.sleep(0.01)
        
    cap.release()
    cv2.destroyAllWindows()
    
    elapsed = time.time() - start_time
    camera_fps = frames_received / elapsed if elapsed > 0 else 0
    
    detection_rate = (faces_detected / frames_received) * 100 if frames_received > 0 else 0
    avg_lat = sum(latencies) / len(latencies) if latencies else 0
    
    latencies.sort()
    p95_lat = latencies[int(len(latencies) * 0.95)] if latencies else 0
    max_lat = latencies[-1] if latencies else 0
    
    print(f"frames received: {frames_received}")
    print(f"face detection attempts: {frames_received}")
    print(f"faces detected: {faces_detected}")
    print(f"face detection rate: {detection_rate:.1f}%")
    print(f"camera FPS: {camera_fps:.1f}")
    print(f"average detection latency: {avg_lat:.2f} ms")
    print(f"P95 detection latency: {p95_lat:.2f} ms")
    print(f"maximum detection latency: {max_lat:.2f} ms")
    
    if frames_received == 0:
        print("\nPHYSICAL CAMERA TEST REQUIRED (NO FRAMES RECEIVED)")
    
if __name__ == "__main__":
    test_camera()
