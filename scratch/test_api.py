
import requests
import time
from pynput.keyboard import Controller, Key
import json

def test():
    keyboard = Controller()
    
    # Wait for server to boot
    for _ in range(10):
        try:
            r = requests.get("http://127.0.0.1:5000/status")
            if r.status_code == 200:
                print("Server is up!")
                break
        except:
            time.sleep(2)
            
    # start monitoring
    requests.post("http://127.0.0.1:5000/api/control", json={"action": "start"})
        
    print("Typing...")
    # type to generate keystrokes over 15 seconds
    for i in range(15):
        keyboard.press(Key.space)
        keyboard.release(Key.space)
        time.sleep(1.0)
        
    time.sleep(1) # wait for buffer to roll
    
    # query status
    r = requests.get("http://127.0.0.1:5000/status")
    data = r.json()
    print("API RESPONSE:")
    print(json.dumps(data, indent=2))
    
    # stop monitoring
    requests.post("http://127.0.0.1:5000/api/control", json={"action": "stop"})
    
if __name__ == "__main__":
    test()

