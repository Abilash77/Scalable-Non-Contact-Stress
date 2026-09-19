import numpy as np

def extract_keystroke_features(events):
    """
    Extracts keystroke dynamics and cognitive load proxies from event sequence.
    Features: dwell_mean, dwell_std, flight_mean, flight_std, typing_speed,
              pause_rate, correction_rate
    """
    if not events:
        return np.zeros(7)
        
    press_times = {}
    dwell_times = []
    flight_times = []
    
    last_press_time = None
    pauses = 0
    corrections = 0
    PAUSE_THRESHOLD = 1.0 # seconds
    
    for event in events:
        key = event['key']
        action = event['action']
        t = event['time']
        
        # Track corrections (Backspace, Delete)
        if action == 'press' and key in ['Key.backspace', 'Key.delete']:
            corrections += 1
            
        if action == 'press':
            if key not in press_times:
                press_times[key] = t
            if last_press_time is not None:
                flight = t - last_press_time
                flight_times.append(flight)
                if flight > PAUSE_THRESHOLD:
                    pauses += 1
            last_press_time = t
            
        elif action == 'release':
            if key in press_times:
                dwell_times.append(t - press_times[key])
                del press_times[key]
                
    total_events = len([e for e in events if e['action'] == 'press'])
    duration = events[-1]['time'] - events[0]['time'] if len(events) > 1 else 0
    
    dwell_mean = np.mean(dwell_times) if dwell_times else 0.0
    dwell_std = np.std(dwell_times) if dwell_times else 0.0
    flight_mean = np.mean(flight_times) if flight_times else 0.0
    flight_std = np.std(flight_times) if flight_times else 0.0
    typing_speed = total_events / duration if duration > 0 else 0.0
    
    pause_rate = pauses / total_events if total_events > 0 else 0.0
    correction_rate = corrections / total_events if total_events > 0 else 0.0
    
    features = np.array([
        dwell_mean,
        dwell_std,
        flight_mean,
        flight_std,
        typing_speed,
        pause_rate,
        correction_rate
    ], dtype=np.float32)
    
    return features
