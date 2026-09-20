import os
import glob
import json
import pandas as pd

def audit_freihaut():
    base_dir = 'data/freihaut_stress/Freihaut-Study_Files_KeyboardData_Analysis-90cd829'
    
    inventory = {
        'lab_participants': 0,
        'online_participants': 0,
        'total_participants': 0,
        'usable_participants': 0,
        'files': 0,
        'records': 0,
        'sessions_trials': 0,
        'conditions': ['high-stress', 'low-stress']
    }
    
    print("Finding data files...")
    all_files = []
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            all_files.append(os.path.join(root, f))
            inventory['files'] += 1
            
    # We look for the main processed data or raw data.
    # Usually in R or CSV format. Let's find CSVs.
    csv_files = [f for f in all_files if f.endswith('.csv')]
    print(f"Found {len(csv_files)} CSV files.")
    
    # Let's inspect the CSV files to find the raw keystrokes and the processed features.
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file, sep=None, engine='python', nrows=5)
            print(f"\n--- File: {os.path.basename(csv_file)} ---")
            print("Columns:", list(df.columns))
        except Exception as e:
            pass

    # Try to load the dataset
    with open('results/freihaut_dataset_inventory.json', 'w') as f:
        json.dump(inventory, f, indent=4)
        
if __name__ == '__main__':
    audit_freihaut()
