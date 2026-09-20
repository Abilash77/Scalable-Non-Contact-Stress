
import cv2
import time

def check_cameras():
    print("Testing cameras...")
    for index in range(5):
        print(f"\nChecking index {index}...")
        try:
            # Try MSMF or DSHOW
            for api_preference in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]:
                print(f"  Trying API {api_preference} on index {index}...")
                cap = cv2.VideoCapture(index, api_preference)
                if cap is None or not cap.isOpened():
                    if cap:
                        cap.release()
                    continue
                
                # wait for warm-up
                time.sleep(0.5)
                ret, frame = cap.read()
                
                if ret and frame is not None:
                    h, w, c = frame.shape
                    print(f"  [PASS] Index {index} works with API {api_preference}")
                    print(f"  Resolution: {w}x{h}")
                    cap.release()
                    return index, api_preference
                else:
                    print(f"  [FAIL] Opened but no frame")
                cap.release()
        except Exception as e:
            print(f"  [ERROR] {e}")

    print("\nNo working cameras found.")
    return -1, -1

check_cameras()

