import gzip
import json
import os

def analyze_dataset(file_path):
    print(f"\n======================================")
    print(f"ANALYZING {file_path}")
    print(f"======================================")
    
    if not os.path.exists(file_path):
        print("File not found.")
        return

    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        data = json.load(f)
        
    num_participants = len(data)
    print(f"Total participants in JSON: {num_participants}")
    
    first_p = list(data.keys())[0]
    print("\nKeys for first participant:")
    print(list(data[first_p].keys()))
    
    # Let's explore the first participant's data to find typing events
    print(f"\n--- Checking first participant {first_p} ---")
    trials_count = 0
    event_counts = 0
    
    for key, val in data[first_p].items():
        if isinstance(val, dict):
            # Check for events or keyboard stuff
            if 'typing' in key.lower() or 'task' in key.lower() or 'experiment' in key.lower() or 'condition' in key.lower():
                print(f"Sub-dict '{key}' keys: {list(val.keys())[:10]}")
                if 'events' in val or 'Keyboard' in val or 'keyboard_events' in val:
                    print(f"  -> Found events in {key}")
                    
        elif isinstance(val, list):
            print(f"List '{key}' with {len(val)} elements")
            if len(val) > 0 and isinstance(val[0], dict):
                print(f"  First element keys: {list(val[0].keys())}")
                if 'timestamp' in val[0] or 'keycode' in val[0] or 'key' in val[0] or 'event' in val[0]:
                    print(f"  -> This looks like an event list!")
                    
    # Let's dig deeper into where the actual events might be stored
    print("\n--- Digging into potential trial data ---")
    # Usually data might be structured as data[participant][condition][trial] or similar
    # Let's print out the full structure for participant 1 up to depth 3
    def print_structure(d, depth=0, max_depth=3):
        if depth > max_depth:
            return
        if isinstance(d, dict):
            for k, v in list(d.items())[:5]:
                print("  " * depth + str(k))
                print_structure(v, depth + 1, max_depth)
        elif isinstance(d, list):
            print("  " * depth + f"[List of {len(d)} items]")
            if len(d) > 0:
                print_structure(d[0], depth + 1, max_depth)
    
    print_structure(data[first_p])

if __name__ == '__main__':
    analyze_dataset('data/freihaut_git/Data-Analysis-Lab-Study/Data/LabStudy_RawData.json.gz')
