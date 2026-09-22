import cv2
import numpy as np

def extract_handwriting_features(img_path=None, img_array=None, strokes=None):
    """
    Extracts geometric, stylistic, and morphological features from handwriting.
    Features: Stroke Width, Curvature/Slant, Density, Aspect Ratio.
    Also extracts Pen Pressure (proxy via intensity). When the real canvas
    strokes are supplied (list of strokes, each a list of {x, y, t} with t in
    ms), stroke velocity (px/ms, pen down) and timing ratio (pen-down time /
    total writing time) are computed from them; otherwise they are 0.
    Total 9 features.
    """
    if img_array is None and img_path is not None:
        img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
    if img_array is None or img_array.size == 0:
        return np.zeros(9)
        
    if len(img_array.shape) > 2:
        if img_array.shape[2] == 4:
            # Blend with white background using alpha channel
            alpha = img_array[:, :, 3] / 255.0
            bg = np.ones_like(img_array[:, :, :3]) * 255
            fg = img_array[:, :, :3]
            blended = (fg * alpha[:, :, None] + bg * (1 - alpha[:, :, None])).astype(np.uint8)
            img_array = cv2.cvtColor(blended, cv2.COLOR_BGR2GRAY)
        else:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        
    # Binarize
    _, binary = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 1. Stroke Width Estimation
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    try:
        # NOTE: cv2.ximgproc.thinning is the preferred implementation when available.
        skeleton = cv2.ximgproc.thinning(binary)
    except (AttributeError, cv2.error):
        # NOTE: The current fallback is a degraded substitute.
        # It is NOT equivalent to the original ximgproc skeletonization implementation.
        # Any latency measurement using this fallback must be identified as fallback-path latency.
        skeleton = np.zeros(binary.shape, np.uint8)
        eroded = np.zeros(binary.shape, np.uint8)
        temp = np.zeros(binary.shape, np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        img_copy = binary.copy()
        while True:
            cv2.erode(img_copy, kernel, eroded)
            cv2.dilate(eroded, kernel, temp)
            cv2.subtract(img_copy, temp, temp)
            cv2.bitwise_or(skeleton, temp, skeleton)
            img_copy = eroded.copy()
            if cv2.countNonZero(img_copy) == 0:
                break
    
    stroke_widths = dist_transform[skeleton > 0]
    stroke_width_mean = np.mean(stroke_widths) if len(stroke_widths) > 0 else 0.0
    stroke_width_std = np.std(stroke_widths) if len(stroke_widths) > 0 else 0.0
    
    # 2. Slant Estimation using Hough Lines
    lines = cv2.HoughLines(skeleton, 1, np.pi / 180, 50)
    angles = []
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            angle = np.degrees(theta)
            if angle > 90:
                angle -= 180
            angles.append(angle)
    slant_mean = np.mean(angles) if len(angles) > 0 else 0.0
    slant_std = np.std(angles) if len(angles) > 0 else 0.0
    
    # 3. Density / Bounding Box ratio
    x, y, w, h = cv2.boundingRect(binary)
    density = np.sum(binary > 0) / (w * h) if (w * h) > 0 else 0.0
    aspect_ratio = w / h if h > 0 else 0.0
    
    # 4. Pen Pressure (proxy: intensity of original image at stroke locations)
    # Original image is grayscale, darker pixels = higher pressure.
    if len(img_array[binary > 0]) > 0:
        pressure_proxy = 255.0 - np.mean(img_array[binary > 0])
    else:
        pressure_proxy = 0.0
        
    # 5. Temporal features from real canvas strokes (0 for static images)
    stroke_velocity, timing_ratio = stroke_timing_features(strokes)
    
    features = np.array([
        stroke_width_mean,
        stroke_width_std,
        slant_mean,
        slant_std,
        density,
        aspect_ratio,
        pressure_proxy,
        stroke_velocity,
        timing_ratio
    ], dtype=np.float32)
    
    return features


def stroke_timing_features(strokes):
    """(mean pen-down velocity px/ms, pen-down time / total writing time)."""
    if not strokes:
        return 0.0, 0.0
    dist = down_ms = 0.0
    starts, ends = [], []
    for stroke in strokes:
        pts = [p for p in stroke if isinstance(p, dict) and all(k in p for k in ('x', 'y', 't'))]
        if len(pts) < 2:
            continue
        xy = np.array([[p['x'], p['y']] for p in pts], dtype=np.float64)
        t = np.array([p['t'] for p in pts], dtype=np.float64)
        dist += float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))
        down_ms += float(t[-1] - t[0])
        starts.append(t[0])
        ends.append(t[-1])
    if down_ms <= 0 or not starts:
        return 0.0, 0.0
    total_ms = max(ends) - min(starts)
    return dist / down_ms, (down_ms / total_ms) if total_ms > 0 else 1.0
