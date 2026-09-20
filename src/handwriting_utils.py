import cv2
import numpy as np

def extract_handwriting_features(img_path=None, img_array=None):
    """
    Extracts geometric, stylistic, and morphological features from handwriting.
    Features: Stroke Width, Curvature/Slant, Density, Aspect Ratio.
    Also extracts Pen Pressure (proxy via intensity) and returns placeholders
    for velocity and timing if temporal data is absent.
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
        
    # 5. Temporal Features (Placeholders for static images)
    stroke_velocity = 0.0
    timing_ratio = 0.0
    
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
