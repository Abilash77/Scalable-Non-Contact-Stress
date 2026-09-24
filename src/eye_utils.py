import cv2
import numpy as np

try:
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
    # Enable iris landmarks by setting refine_landmarks=True
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

def extract_eye_features(frame, landmarks=None):
    """
    Extracts 5 eye-specific features using MediaPipe Iris Landmarks.
    Features: pupil_size_left, pupil_size_right, gaze_x, gaze_y, eye_closure
    """
    if landmarks is None:
        if face_mesh is None or frame is None or frame.size == 0:
            return np.zeros(5), "CAMERA_UNAVAILABLE"
            
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return np.zeros(5), "NO_FACE_DETECTED"
            
        landmarks = results.multi_face_landmarks[0].landmark
        
    h, w, _ = frame.shape
    pts = np.array([(lm.x * w, lm.y * h) for lm in landmarks])
    
    # 1. Pupil Size (Relative to Eye Width)
    # MediaPipe Iris indices (left eye: 468-472, right eye: 473-477)
    # 468 is center of left iris, 473 is center of right iris.
    # We use distance between 469 and 471 (iris horiz) as proxy for pupil/iris size
    if len(pts) > 477:
        iris_l_h = get_distance(pts[469], pts[471])
        iris_r_h = get_distance(pts[474], pts[476])
        
        eye_l_w = get_distance(pts[33], pts[133])
        eye_r_w = get_distance(pts[362], pts[263])
        
        pupil_size_left = iris_l_h / (eye_l_w + 1e-6)
        pupil_size_right = iris_r_h / (eye_r_w + 1e-6)
        
        # 2. Gaze Direction (Relative Iris Center vs Eye Corners)
        iris_c_l = pts[468]
        iris_c_r = pts[473]
        
        # Gaze X: (Iris X - Inner Corner X) / (Eye Width)
        gaze_x_l = (iris_c_l[0] - pts[133][0]) / (eye_l_w + 1e-6)
        gaze_x_r = (iris_c_r[0] - pts[362][0]) / (eye_r_w + 1e-6)
        gaze_x = (np.abs(gaze_x_l) + np.abs(gaze_x_r)) / 2.0
        
        # Gaze Y: (Iris Y - Top Eyelid Y) / (Eye Height)
        eye_l_h = get_distance(pts[159], pts[145])
        eye_r_h = get_distance(pts[386], pts[374])
        
        gaze_y_l = (iris_c_l[1] - pts[159][1]) / (eye_l_h + 1e-6)
        gaze_y_r = (iris_c_r[1] - pts[386][1]) / (eye_r_h + 1e-6)
        gaze_y = (np.abs(gaze_y_l) + np.abs(gaze_y_r)) / 2.0
    else:
        pupil_size_left = 0.0
        pupil_size_right = 0.0
        gaze_x = 0.0
        gaze_y = 0.0
        
    # 3. Eye Closure (overall)
    # Average EAR of both eyes (lower means closed)
    if eye_l_w > 0 and eye_r_w > 0:
        ear_l = eye_l_h / eye_l_w
        ear_r = eye_r_h / eye_r_w
        eye_closure = (ear_l + ear_r) / 2.0
    else:
        eye_closure = 0.0
        
    features = np.array([
        pupil_size_left,
        pupil_size_right,
        gaze_x,
        gaze_y,
        eye_closure
    ], dtype=np.float32)
    
    return features, "DETECTED"
