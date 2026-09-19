import requests
import time
import json
import sys
from datetime import datetime

print("Starting 5-Minute Live Stability Stress-Test on /status endpoint...")

start_time = time.time()
duration = 5 * 60  # 5 minutes
success_count = 0
error_count = 0
latencies = []

while time.time() - start_time < duration:
    t0 = time.perf_counter()
    try:
        res = requests.get('http://127.0.0.1:5000/status', timeout=2)
        if res.status_code == 200:
            success_count += 1
            lat = (time.perf_counter() - t0) * 1000
            latencies.append(lat)
            data = res.json()
            sys.stdout.write(f"\r[{datetime.now().strftime('%H:%M:%S')}] OK | Latency: {lat:.1f}ms | Masks: {data.get('fusion_mask', [])} | FC: {data.get('face_status', 'UNK')}   ")
            sys.stdout.flush()
        else:
            error_count += 1
            print(f"\n[ERROR] HTTP {res.status_code}")
    except Exception as e:
        error_count += 1
        print(f"\n[ERROR] Request failed: {e}")
        
    time.sleep(1)

print("\n\n=== STABILITY TEST RESULTS ===")
print(f"Total Requests: {success_count + error_count}")
print(f"Success: {success_count}")
print(f"Errors: {error_count}")
if latencies:
    print(f"Mean Latency: {sum(latencies)/len(latencies):.2f} ms")
    print(f"Max Latency:  {max(latencies):.2f} ms")
