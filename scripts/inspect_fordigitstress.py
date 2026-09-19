import os
import sys

def inspect_dataset(data_dir="data/fordigitstress/"):
    """
    Inspect the raw ForDigitStress dataset without assuming any internal structure.
    If the directory does not exist, explicitly halt.
    """
    if not os.path.exists(data_dir):
        print("\n" + "="*50)
        print("FOR-DIGIT-STRESS DATASET NOT FOUND")
        print("="*50)
        print(f"Expected directory: {os.path.abspath(data_dir)}")
        print("\nThe ForDigitStress dataset requires manual data acquisition.")
        print("Please visit https://hcai.eu/fordigitstress to request access.")
        print("Once downloaded, extract the participant folders (VP10, VP11, etc.) into the expected directory.")
        print("="*50 + "\n")
        sys.exit(1)

    print("="*50)
    print("RAW DATASET INSPECTION")
    print("="*50)

    try:
        entries = os.listdir(data_dir)
    except Exception as e:
        print(f"Error reading directory {data_dir}: {e}")
        sys.exit(1)

    # Subject count and IDs
    subject_dirs = [d for d in entries if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('VP')]
    subject_ids = sorted([int(d.replace('VP', '')) for d in subject_dirs if d.replace('VP', '').isdigit()])

    print(f"- subject count: {len(subject_dirs)}")
    if subject_ids:
        print(f"- subject IDs: {subject_ids}")
    else:
        print("- subject IDs: None detected (Expected VP10, VP11...)")

    if len(subject_dirs) == 0:
        print("\nWarning: Data directory exists but contains no valid subject folders.")
        sys.exit(1)

    # Global inventory metrics
    total_files = 0
    file_extensions = set()
    modality_candidates = set()
    annotation_files = set()
    total_size_bytes = 0

    for root, dirs, files in os.walk(data_dir):
        for f in files:
            total_files += 1
            f_path = os.path.join(root, f)
            try:
                total_size_bytes += os.path.getsize(f_path)
            except OSError:
                pass
            
            ext = os.path.splitext(f)[1].lower()
            file_extensions.add(ext)

            f_lower = f.lower()
            if 'stress.csv' in f_lower or 'label' in f_lower or 'annotation' in f_lower:
                annotation_files.add(f)
            if 'audio' in f_lower or ext in ['.wav', '.mp3']:
                modality_candidates.add('Speech')
            if 'video' in f_lower or ext in ['.mp4', '.avi']:
                modality_candidates.add('Facial')
            if 'pupil' in f_lower or 'eye' in f_lower or 'gaze' in f_lower:
                modality_candidates.add('Eye/Pupil')
            if 'key' in f_lower or 'type' in f_lower:
                modality_candidates.add('Keyboard')
            if 'hand' in f_lower or 'pen' in f_lower or 'stroke' in f_lower:
                modality_candidates.add('Handwriting')

    print(f"- file count: {total_files}")
    print(f"- file types: {list(file_extensions)}")
    print(f"- modality candidates: {list(modality_candidates)}")
    print(f"- annotation files: {list(annotation_files)}")
    
    # Missing information that requires deeper parsing
    print("- duration where determinable: [BLOCKED] Requires data parser")
    print("- timestamps: [BLOCKED] Requires data parser")
    print("- missing files: [BLOCKED] Requires baseline schema")
    print("- corrupted/unreadable files: [BLOCKED] Requires data loader")
    
    gb_size = total_size_bytes / (1024**3)
    print(f"- dataset size: {gb_size:.2f} GB")
    
    print("\nDataset inspection complete. Awaiting deep parser implementation for specific modalities.")

if __name__ == "__main__":
    inspect_dataset()
