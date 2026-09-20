import os
import glob
import json

def inspect_dataset():
    data_dir = 'data/keystroke-stress/data'
    inventory = {
        'total_files': 0,
        'pam_records': 0,
        'keystroke_logs': [],
        'pam_files': [],
        'other_files': []
    }
    
    if not os.path.exists(data_dir):
        print(f"Directory {data_dir} does not exist.")
        return
        
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            inventory['total_files'] += 1
            path = os.path.join(root, file)
            size = os.path.getsize(path)
            if 'pam_log' in file:
                inventory['pam_files'].append({'file': path, 'size': size})
                with open(path, 'r') as f:
                    inventory['pam_records'] += len(f.readlines())
            elif 'keys_and_mouse' in root and file.endswith('.txt'):
                inventory['keystroke_logs'].append({'file': path, 'size': size})
            else:
                inventory['other_files'].append({'file': path, 'size': size})
                
    os.makedirs('results', exist_ok=True)
    with open('results/dataset_inventory.json', 'w') as f:
        json.dump(inventory, f, indent=4)
        
    print(f"Inventory saved. Found {inventory['total_files']} files.")
    print(f"PAM records: {inventory['pam_records']}")
    print(f"Keystroke logs: {len(inventory['keystroke_logs'])}")
    
if __name__ == '__main__':
    inspect_dataset()
