import cv2
import numpy as np

try:
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
except ImportError:
    mp_face_mesh = None
    face_mesh = None

def get_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def extract_face_features(img_array, landmarks=None):
    """
    Extracts 12 facial features using MediaPipe Face Mesh.
    Features: EAR_left, EAR_right, Pitch, Yaw, Roll, MAR (Mouth Aspect Ratio),
              Eyebrow_raise_left, Eyebrow_raise_right, 
              Mouth_width, Lip_distance, Nose_wrinkle (proxy), Jaw_drop
    """
    if img_array is None or img_array.size == 0:
        return np.zeros(12), "CAMERA_UNAVAILABLE", {}
        
    frame = img_array

    if landmarks is None:
        if face_mesh is None or frame is None:
            return np.zeros(12), "CAMERA_UNAVAILABLE", {}
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        if not results.multi_face_landmarks:
            return np.zeros(12), "NO_FACE_DETECTED", {"count": 0}
        landmarks = results.multi_face_landmarks[0].landmark
        face_count = len(results.multi_face_landmarks)
    else:
        face_count = 1

    h, w, _ = frame.shape
    pts = np.array([(lm.x * w, lm.y * h) for lm in landmarks])
    
    # 1. Eye Aspect Ratio (EAR)
    def calc_ear(eye_pts):
        # MediaPipe Eye indices (simplified)
        # Left: 33, 160, 158, 133, 153, 144
        # Right: 362, 385, 387, 263, 373, 380
        if len(eye_pts) < 6: return 0.0
        v1 = get_distance(pts[eye_pts[1]], pts[eye_pts[5]])
        v2 = get_distance(pts[eye_pts[2]], pts[eye_pts[4]])
        hz = get_distance(pts[eye_pts[0]], pts[eye_pts[3]])
        return (v1 + v2) / (2.0 * hz + 1e-6)
        
    ear_left = calc_ear([33, 160, 158, 133, 153, 144])
    ear_right = calc_ear([362, 385, 387, 263, 373, 380])
    
    # 2. Head Pose Proxy
    # Nose tip (1), chin (152), left eye corner (33), right eye corner (263)
    nose = pts[1]
    chin = pts[152]
    eye_l = pts[33]
    eye_r = pts[263]
    
    # Simple proxies instead of solvePnP for speed
    yaw = (nose[0] - (eye_l[0] + eye_r[0])/2) / (get_distance(eye_l, eye_r) + 1e-6)
    pitch = (nose[1] - (eye_l[1] + eye_r[1])/2) / (get_distance(nose, chin) + 1e-6)
    roll = (eye_r[1] - eye_l[1]) / (get_distance(eye_l, eye_r) + 1e-6)
    
    # 3. Facial Action Units (Proxies)
    # MAR (Mouth Aspect Ratio) - inner lips: 78, 81, 13, 311, 308, 402, 14, 178
    mar_v = get_distance(pts[13], pts[14])
    mar_h = get_distance(pts[78], pts[308])
    mar = mar_v / (mar_h + 1e-6)
    
    # Eyebrow raise (distance from eye to eyebrow)
    brow_l = get_distance(pts[159], pts[52]) / (mar_h + 1e-6)
    brow_r = get_distance(pts[386], pts[282]) / (mar_h + 1e-6)
    
    # Mouth width
    mouth_width = mar_h / (get_distance(eye_l, eye_r) + 1e-6)
    
    # Lip distance (outer)
    lip_dist = get_distance(pts[0], pts[17]) / (mar_h + 1e-6)
    
    # Nose wrinkle (distance between brows)
    nose_wrinkle = get_distance(pts[55], pts[285]) / (get_distance(eye_l, eye_r) + 1e-6)
    
    # Jaw drop
    jaw_drop = get_distance(pts[1], pts[152]) / (get_distance(eye_l, eye_r) + 1e-6)
    
    features = np.array([
        ear_left, ear_right,
        pitch, yaw, roll,
        mar, brow_l, brow_r,
        mouth_width, lip_dist,
        nose_wrinkle, jaw_drop
    ], dtype=np.float32)
    
    xs = [pt[0] for pt in pts]
    ys = [pt[1] for pt in pts]
    x_min, x_max = int(min(xs)), int(max(xs))
    y_min, y_max = int(min(ys)), int(max(ys))
    
    metadata = {
        "count": face_count,
        "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
        "confidence": "N/A"
    }
    
    return features, "DETECTED", metadata
