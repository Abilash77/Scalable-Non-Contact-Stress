
import requests
import time
import subprocess
import os
import json
from pynput.keyboard import Controller, Key

def test():
    # Kill existing servers
    os.system("powershell -Command \"Stop-Process -Name python -Force -ErrorAction SilentlyContinue\"")
    time.sleep(1)
    
    print("Starting server...")
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    server = subprocess.Popen(["python", "run.py"], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait for server to boot
    server_up = False
    for _ in range(20):
        try:
            r = requests.get("http://127.0.0.1:5000/status")
            if r.status_code == 200:
                print("Server is up!")
                server_up = True
                break
        except:
            time.sleep(2)
            
    if not server_up:
        print("Server failed to start")
        server.terminate()
        return

    # Check status (Initial)
    r = requests.get("http://127.0.0.1:5000/status")
    data = r.json()
    print("Initial Camera Connected:", data.get("camera_connected"))
    
    print("Starting LIVE monitoring...")
    requests.post("http://127.0.0.1:5000/api/control", json={"action": "start"})
    
    print("Waiting for camera initialization (10s)...")
    time.sleep(10)
    
    r = requests.get("http://127.0.0.1:5000/status")
    data = r.json()
    print("Live Camera Connected:", data.get("camera_connected"))
    print("Live Camera Error:", data.get("camera_error"))
    print("Live Face Detected:", data.get("face_detected"))
    
    print("Typing 25 spaces...")
    keyboard = Controller()
    for i in range(25):
        keyboard.press(Key.space)
        keyboard.release(Key.space)
        time.sleep(0.1)
        
    time.sleep(1)
    r = requests.get("http://127.0.0.1:5000/status")
    data = r.json()
    print("Prediction Available:", data.get("predictions_available"))
    print("Prediction:", data.get("prediction"))
    print("Face Detected during Typing:", data.get("face_detected"))
    
    print("Testing /video_feed stream...")
    try:
        r_stream = requests.get("http://127.0.0.1:5000/video_feed", stream=True, timeout=5)
        print("Video Stream Status:", r_stream.status_code)
    except Exception as e:
        print("Video Stream Failed:", e)
        
    print("Stopping LIVE monitoring...")
    requests.post("http://127.0.0.1:5000/api/control", json={"action": "stop"})
    
    time.sleep(3)
    r = requests.get("http://127.0.0.1:5000/status")
    data = r.json()
    print("Stopped Camera Connected:", data.get("camera_connected"))
    
    print("Terminating server...")
    server.terminate()

if __name__ == "__main__":
    test()

