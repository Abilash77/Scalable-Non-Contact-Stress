import os

def scan_dir(path):
    print(f"\nScanning: {path}")
    if not os.path.exists(path):
        print(f"{path} does not exist.")
        return
    
    for root, _, files in os.walk(path):
        for file in files:
            full_path = os.path.join(root, file)
            size_mb = os.path.getsize(full_path) / (1024 * 1024)
            print(f"{full_path} - {size_mb:.2f} MB")

scan_dir('data')
scan_dir('datasets')
scan_dir('raw')
