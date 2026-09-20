#!/usr/bin/env python3
"""
Freihaut & Göritz (2021) Keyboard Stress Preprocessing Pipeline
================================================================

Converts raw KeyDown/KeyUp events from the Freihaut & Göritz keyboard stress
dataset into temporal (N, 10, 7) sequences for RA-HMSD model training.

Dataset: "Does People's Keyboard Typing Reflect Their Stress Level" 
Source:  Zenodo 10.5281/zenodo.4445197

RAW KEY EVENTS
    ->
participant / session / condition separation
    ->
KeyDown / KeyUp pairing (cleaned, sorted)
    ->
temporal keyboard features (7D per sub-window)
    ->
valid stress-condition labels
    ->
participant-level split (70/15/15)
    ->
10-step temporal sequences
    ->
saved processed arrays + metadata

=====================================================================
FEATURE TABLE
=====================================================================

RAW EVENT FIELD          -> FEATURE          -> UNIT       -> DEFINITION
─────────────────────────────────────────────────────────────────────
time (KeyDown->KeyUp)     -> mean_dwell       -> ms         -> Mean dwell time (KeyDown to KeyUp of same key)
time (KeyDown->KeyUp)     -> std_dwell        -> ms         -> Std dev of dwell time
time (KeyUp->next KeyDown)-> mean_flight      -> ms         -> Mean flight time (inter-key latency)
time (KeyUp->next KeyDown)-> std_flight       -> ms         -> Std dev of flight time
time, eventType count    -> typing_speed     -> keys/sec   -> Keystrokes per second in sub-window
key == 'Backspace'       -> backspace_freq   -> ratio 0-1  -> Fraction of keystrokes that are Backspace
isCorrect field          -> error_rate       -> ratio 0-1  -> Fraction of keystrokes marked incorrect

=====================================================================
LABEL MAPPING
=====================================================================

HIGH-STRESS (HS_PatternTyping)  -> label 1  (experimental stress condition)
LOW-STRESS  (LS_PatternTyping)  -> label 0  (experimental control condition)

These are experimentally defined conditions, NOT medical stress diagnoses.
=====================================================================
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import operator
import os
import pickle
import sys
import warnings
from collections import defaultdict
from itertools import groupby
from pathlib import Path

import numpy as np


# ===========================================================================
# CONSTANTS
# ===========================================================================
RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Windowing: group raw events into sub-windows of N keystrokes each
# Then chain 10 consecutive sub-windows into one (10, 7) sample
KEYSTROKES_PER_SUBWINDOW = 4   # Number of key-presses per temporal step
NUM_TIMESTEPS = 10              # Temporal sequence length (model expects 10)
STRIDE_SUBWINDOWS = 1           # Stride between consecutive samples (in sub-windows)
MIN_KEYSTROKES_PER_TRIAL = 20  # Minimum keystrokes for a condition to be usable

FEATURE_NAMES = [
    "mean_dwell",     # Mean dwell time (ms)
    "std_dwell",      # Std of dwell time (ms)
    "mean_flight",    # Mean flight time (ms)
    "std_flight",     # Std of flight time (ms)
    "typing_speed",   # Keystrokes per second
    "backspace_freq", # Fraction of backspace presses
    "error_rate",     # Fraction of incorrect keystrokes
]
NUM_FEATURES = len(FEATURE_NAMES)  # 7


# ===========================================================================
# RAW EVENT CLEANING (adapted from Freihaut's original notebook)
# ===========================================================================

def clean_keyboard_events(raw_events: list[dict]) -> list[dict]:
    """
    Clean raw events from a PatternTyping trial.
    
    Filters to KeyDown/KeyUp only, pairs them, removes artifacts,
    and returns a sorted list of valid paired keyboard events.
    
    Adapted from Freihaut's get_typing_data() in LabStudy_Keyboard_Data_Processing.ipynb.
    """
    # Filter to keyboard events only (exclude mouse events)
    key_events = [e for e in raw_events 
                  if isinstance(e, dict) and e.get("eventType") in ("KeyDown", "KeyUp")]
    
    if len(key_events) < 4:  # Need at least 2 complete keystrokes
        return []
    
    keyboard_events = []
    key_down_events = {}  # code -> [list of pending KeyDown events]
    artifacts = 0
    
    for event in key_events:
        # Skip Shift keys (they are modifiers, not character keys)
        if event.get("key") == "Shift":
            continue
            
        if event["eventType"] == "KeyDown":
            code = event["code"]
            if code not in key_down_events:
                key_down_events[code] = [event]
            else:
                # If less than 1750ms since last KeyDown of same code, it's key repeat
                time_diff = event["time"] - key_down_events[code][-1]["time"]
                if time_diff < 1750:
                    key_down_events[code].append(event)
                else:
                    # Artifact: stale KeyDown with no matching KeyUp
                    artifacts += 1
                    key_down_events[code] = [event]
                    
        elif event["eventType"] == "KeyUp":
            code = event["code"]
            if code in key_down_events and len(key_down_events[code]) > 0:
                # Pair with the first (oldest) pending KeyDown
                keyboard_events.append(key_down_events[code][0])
                keyboard_events.append(event)
                del key_down_events[code]
            else:
                # KeyUp without KeyDown - artifact
                artifacts += 1
    
    # Sort by time
    keyboard_events = sorted(keyboard_events, key=operator.itemgetter("time"))
    return keyboard_events


def extract_keystroke_pairs(clean_events: list[dict]) -> list[dict]:
    """
    Extract paired keystrokes from cleaned events.
    
    Returns a list of dicts, each representing one complete keystroke:
    {
        'code': str,
        'key': str,
        'down_time': int (ms),
        'up_time': int (ms),
        'dwell_time': float (ms),
        'is_correct': bool,
        'is_backspace': bool,
    }
    """
    pairs = []
    i = 0
    while i < len(clean_events) - 1:
        ev_down = clean_events[i]
        ev_up = clean_events[i + 1]
        
        # Verify this is a proper KeyDown/KeyUp pair for the same key
        if (ev_down["eventType"] == "KeyDown" and 
            ev_up["eventType"] == "KeyUp" and 
            ev_down["code"] == ev_up["code"]):
            
            dwell = ev_up["time"] - ev_down["time"]
            # Sanity: dwell should be positive and < 5 seconds
            if 0 < dwell < 5000:
                pairs.append({
                    "code": ev_down["code"],
                    "key": ev_down.get("key", ""),
                    "down_time": ev_down["time"],
                    "up_time": ev_up["time"],
                    "dwell_time": float(dwell),
                    "is_correct": ev_down.get("isCorrect", True),
                    "is_backspace": "Backspace" in ev_down.get("code", ""),
                })
            i += 2
        else:
            # Misalignment - skip one event
            i += 1
    
    return pairs


def compute_flight_times(pairs: list[dict]) -> list[float]:
    """
    Compute flight times (inter-key latency) between consecutive keystrokes.
    
    Flight time = KeyDown[i+1] - KeyUp[i]
    Can be negative if next key is pressed before previous is released.
    """
    flights = []
    for j in range(len(pairs) - 1):
        flight = pairs[j + 1]["down_time"] - pairs[j]["up_time"]
        # Clip extreme outliers (e.g., > 10 seconds = likely a pause)
        if abs(flight) < 10000:
            flights.append(float(flight))
    return flights


# ===========================================================================
# FEATURE EXTRACTION PER SUB-WINDOW
# ===========================================================================

def compute_subwindow_features(pairs: list[dict]) -> np.ndarray | None:
    """
    Compute the 7-dimensional feature vector for a sub-window of keystrokes.
    
    Args:
        pairs: List of keystroke pair dicts for this sub-window.
        
    Returns:
        np.ndarray of shape (7,) or None if insufficient data.
    """
    if len(pairs) < 3:  # Need at least 3 keystrokes for meaningful statistics
        return None
    
    dwell_times = [p["dwell_time"] for p in pairs]
    flight_times = compute_flight_times(pairs)
    
    if len(flight_times) < 2:
        return None
    
    # Total time span of this sub-window in seconds
    time_span = (pairs[-1]["up_time"] - pairs[0]["down_time"]) / 1000.0
    if time_span <= 0:
        return None
    
    # Feature 1: mean_dwell (ms)
    mean_dwell = np.mean(dwell_times)
    
    # Feature 2: std_dwell (ms)
    std_dwell = np.std(dwell_times) if len(dwell_times) > 1 else 0.0
    
    # Feature 3: mean_flight (ms)
    mean_flight = np.mean(flight_times)
    
    # Feature 4: std_flight (ms)
    std_flight = np.std(flight_times) if len(flight_times) > 1 else 0.0
    
    # Feature 5: typing_speed (keystrokes/second)
    typing_speed = len(pairs) / time_span
    
    # Feature 6: backspace_freq (ratio 0-1)
    n_backspace = sum(1 for p in pairs if p["is_backspace"])
    backspace_freq = n_backspace / len(pairs)
    
    # Feature 7: error_rate (ratio 0-1)
    n_errors = sum(1 for p in pairs if not p["is_correct"])
    error_rate = n_errors / len(pairs)
    
    return np.array([
        mean_dwell, std_dwell, mean_flight, std_flight,
        typing_speed, backspace_freq, error_rate
    ], dtype=np.float32)


# ===========================================================================
# WINDOWING: create (10, 7) temporal sequences
# ===========================================================================

def create_temporal_sequences(pairs: list[dict]) -> list[np.ndarray]:
    """
    Create temporal (NUM_TIMESTEPS, NUM_FEATURES) sequences from keystroke pairs.
    
    Strategy:
    - Divide the keystroke stream into sub-windows of KEYSTROKES_PER_SUBWINDOW keys
    - Compute 7D features for each sub-window
    - Slide a window of NUM_TIMESTEPS consecutive sub-windows with STRIDE_SUBWINDOWS stride
    - Each output is a (10, 7) array
    
    This ensures every timestep corresponds to a REAL temporal segment, 
    NOT a repeated static vector.
    """
    if len(pairs) < KEYSTROKES_PER_SUBWINDOW * 3:
        # Not enough keystrokes for even a partial sequence
        return []
    
    # Step 1: Divide into sub-windows
    subwindow_features = []
    n_subwindows = len(pairs) // KEYSTROKES_PER_SUBWINDOW
    
    for sw_idx in range(n_subwindows):
        start = sw_idx * KEYSTROKES_PER_SUBWINDOW
        end = start + KEYSTROKES_PER_SUBWINDOW
        sw_pairs = pairs[start:end]
        
        feat = compute_subwindow_features(sw_pairs)
        if feat is not None:
            subwindow_features.append(feat)
    
    if len(subwindow_features) < NUM_TIMESTEPS:
        return []
    
    # Step 2: Create sliding windows of NUM_TIMESTEPS consecutive sub-windows
    sequences = []
    for start_idx in range(0, len(subwindow_features) - NUM_TIMESTEPS + 1, STRIDE_SUBWINDOWS):
        seq = np.stack(subwindow_features[start_idx:start_idx + NUM_TIMESTEPS], axis=0)
        assert seq.shape == (NUM_TIMESTEPS, NUM_FEATURES), f"Bad shape: {seq.shape}"
        sequences.append(seq)
    
    return sequences


# ===========================================================================
# DATA LOADING
# ===========================================================================

def load_lab_study(data_path: str) -> dict:
    """
    Load Lab Study raw data.
    
    Returns: {participant_id: {condition: [(keystroke_pairs, label)]}}
    """
    gz_path = os.path.join(data_path, "Data-Analysis-Lab-Study", "Data", "LabStudy_RawData.json.gz")
    print(f"  Loading Lab Study: {gz_path}")
    
    with gzip.open(gz_path, "rt") as f:
        data = json.load(f)
    
    participants = {}
    skipped = 0
    
    for pid in data:
        # Check participant completed the study
        if "phaseFinishedTimestamps" not in data[pid]:
            skipped += 1
            continue
        if "BfiNeuroticism" not in data[pid].get("phaseFinishedTimestamps", {}):
            skipped += 1
            continue
        
        conditions = {}
        
        for task_key, label in [("HS_PatternTyping", 1), ("LS_PatternTyping", 0)]:
            if task_key not in data[pid]:
                continue
                
            raw_events = data[pid][task_key]
            if not isinstance(raw_events, list):
                continue
            
            # Clean and pair keyboard events
            clean_events = clean_keyboard_events(raw_events)
            pairs = extract_keystroke_pairs(clean_events)
            
            if len(pairs) >= MIN_KEYSTROKES_PER_TRIAL:
                conditions[task_key] = (pairs, label)
        
        if len(conditions) > 0:
            participants[f"lab_{pid}"] = conditions
    
    print(f"  Lab Study: {len(participants)} usable participants ({skipped} skipped)")
    return participants


def load_online_study(data_path: str) -> dict:
    """
    Load Online Study raw data.
    
    The online study uses:
    - Con_PatternTyping: the experimental condition (stress or control)
    - Pr_PatternTyping: practice condition (always low-stress baseline)
    - ExperimentMetaData.condition: the actual assigned condition
    
    IMPORTANT: Online study TrackerData is a DICT (not a list like lab study).
    Each value is an event dict with slightly different field names:
    - 'keyCode' (int) instead of 'code' (string)
    - 'validKey' field for filtering
    - 'isPractice' field to exclude practice trials within condition
    
    Returns: {participant_id: {condition_key: (keystroke_pairs, label)}}
    """
    gz_path = os.path.join(data_path, "Data-Analysis-Online-Study", "Data", "OnlineStudy_RawData.json.gz")
    ids_path = os.path.join(data_path, "Data-Analysis-Online-Study", "Data", "filtered_online_study_ids_studyLevel")
    
    print(f"  Loading Online Study: {gz_path}")
    print(f"  This may take a few minutes for the 288MB compressed file...")
    
    # Load valid participant IDs
    with open(ids_path, "rb") as f:
        valid_ids = pickle.load(f)
    print(f"  Valid online participant IDs: {len(valid_ids)}")
    
    with gzip.open(gz_path, "rt") as f:
        data = json.load(f)
    
    print(f"  Online study loaded: {len(data)} total participants")
    
    participants = {}
    skipped = 0
    
    for pid in data:
        if pid not in valid_ids:
            skipped += 1
            continue
        
        # Get the experimental condition
        try:
            exp_condition = data[pid]["ExperimentMetaData"]["condition"]
        except (KeyError, TypeError):
            skipped += 1
            continue
        
        conditions = {}
        
        for task_key, is_practice_task in [("Pr_PatternTyping", True), ("Con_PatternTyping", False)]:
            try:
                tracker_data = data[pid][task_key]["data"]["TrackerData"]
            except (KeyError, TypeError):
                continue
            
            # Convert dict-based TrackerData to a sorted list of events
            if isinstance(tracker_data, dict):
                events_list = _convert_online_tracker_data(tracker_data)
            elif isinstance(tracker_data, list):
                events_list = tracker_data
            else:
                continue
            
            # Clean and pair keyboard events
            clean_events = clean_keyboard_events(events_list)
            pairs = extract_keystroke_pairs(clean_events)
            
            if len(pairs) >= MIN_KEYSTROKES_PER_TRIAL:
                if is_practice_task:
                    # Practice = always low stress baseline
                    label = 0
                else:
                    # Condition task: label depends on assigned condition
                    # ExperimentMetaData condition: 0 = high stress, 1 = low stress
                    label = 1 if exp_condition == 0 else 0
                conditions[task_key] = (pairs, label)
        
        if len(conditions) > 0:
            participants[f"online_{pid}"] = conditions
    
    print(f"  Online Study: {len(participants)} usable participants ({skipped} skipped)")
    return participants


def _convert_online_tracker_data(tracker_dict: dict) -> list[dict]:
    """
    Convert online study TrackerData (dict keyed by string indices) 
    to a sorted list of event dicts matching the lab study format.
    
    Online events have:
    - 'keyCode' (int) instead of 'code' (string) 
    - 'validKey' field
    - 'isPractice' field
    
    We normalize them to have 'code' and 'key' fields compatible with
    our clean_keyboard_events() function.
    """
    # Map common keyCodes to code strings
    KEYCODE_TO_CODE = {
        8: "Backspace", 16: "ShiftLeft", 20: "CapsLock",
    }
    for i in range(48, 58):  # 0-9
        KEYCODE_TO_CODE[i] = f"Digit{chr(i)}"
    for i in range(65, 91):  # A-Z
        KEYCODE_TO_CODE[i] = f"Key{chr(i)}"
    for i in range(96, 106):  # Numpad 0-9
        KEYCODE_TO_CODE[i] = f"Numpad{i - 96}"
    
    events = []
    for key, event in tracker_dict.items():
        if not isinstance(event, dict):
            continue
        
        # Skip practice trial events
        if event.get("isPractice", False):
            continue
        
        # Skip invalid keys
        if not event.get("validKey", True):
            continue
        
        # Must be a keyboard event
        event_type = event.get("eventType", "")
        if event_type not in ("KeyDown", "KeyUp"):
            continue
        
        # Skip invalid keyCodes
        key_code = event.get("keyCode", 0)
        if key_code in (0, 255):
            continue
        
        # Normalize to lab study format
        code = KEYCODE_TO_CODE.get(key_code, f"Unknown{key_code}")
        key_char = event.get("key", chr(key_code) if 32 <= key_code < 127 else "")
        
        normalized_event = {
            "eventType": event_type,
            "code": code,
            "key": key_char if key_code != 16 else "Shift",
            "time": event.get("time", 0),
            "isCorrect": event.get("isCorrect", True),
            "taskNumber": event.get("taskNumber", 0),
        }
        events.append(normalized_event)
    
    # Sort by time
    events.sort(key=lambda e: e["time"])
    return events


# ===========================================================================
# PARTICIPANT-LEVEL SPLITTING
# ===========================================================================

def split_participants(participant_ids: list[str], seed: int = RANDOM_SEED) -> tuple:
    """
    Split participant IDs into train/val/test sets.
    
    Uses fixed random seed for reproducibility.
    70% train, 15% validation, 15% test.
    """
    rng = np.random.RandomState(seed)
    ids = sorted(participant_ids)  # Sort for determinism
    rng.shuffle(ids)
    
    n = len(ids)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    
    train_ids = ids[:n_train]
    val_ids = ids[n_train:n_train + n_val]
    test_ids = ids[n_train + n_val:]
    
    return train_ids, val_ids, test_ids


# ===========================================================================
# LEAKAGE CHECKS
# ===========================================================================

def check_leakage(train_ids, val_ids, test_ids, X_train, X_val, X_test, 
                  y_train, y_val, y_test) -> dict:
    """
    Comprehensive data leakage audit.
    """
    audit = {
        "participant_leakage": False,
        "session_leakage": False,
        "duplicate_windows": False,
        "identical_feature_vectors": False,
        "label_distribution_valid": True,
        "details": {}
    }
    
    # 1. Participant leakage
    train_set = set(train_ids)
    val_set = set(val_ids)
    test_set = set(test_ids)
    
    tv_overlap = train_set & val_set
    tt_overlap = train_set & test_set
    vt_overlap = val_set & test_set
    
    if tv_overlap or tt_overlap or vt_overlap:
        audit["participant_leakage"] = True
        audit["details"]["train_val_overlap"] = list(tv_overlap)
        audit["details"]["train_test_overlap"] = list(tt_overlap)
        audit["details"]["val_test_overlap"] = list(vt_overlap)
    
    # 2. Duplicate window check (exact matches across splits)
    def array_hash(arr):
        return hashlib.md5(arr.tobytes()).hexdigest()
    
    train_hashes = set(array_hash(X_train[i]) for i in range(len(X_train)))
    val_hashes = set(array_hash(X_val[i]) for i in range(len(X_val)))
    test_hashes = set(array_hash(X_test[i]) for i in range(len(X_test)))
    
    cross_dup_tv = train_hashes & val_hashes
    cross_dup_tt = train_hashes & test_hashes
    cross_dup_vt = val_hashes & test_hashes
    
    if cross_dup_tv or cross_dup_tt or cross_dup_vt:
        audit["duplicate_windows"] = True
        audit["details"]["cross_split_duplicates"] = {
            "train_val": len(cross_dup_tv),
            "train_test": len(cross_dup_tt),
            "val_test": len(cross_dup_vt),
        }
    
    # 3. Within-split duplicates
    within_train = len(X_train) - len(train_hashes)
    within_val = len(X_val) - len(val_hashes)
    within_test = len(X_test) - len(test_hashes)
    
    if within_train > 0 or within_val > 0 or within_test > 0:
        audit["identical_feature_vectors"] = True
        audit["details"]["within_split_duplicates"] = {
            "train": within_train,
            "val": within_val,
            "test": within_test
        }
    
    # 4. Label distribution check
    for name, y in [("train", y_train), ("val", y_val), ("test", y_test)]:
        unique, counts = np.unique(y, return_counts=True)
        if len(unique) < 2:
            audit["label_distribution_valid"] = False
            audit["details"][f"{name}_missing_class"] = True
    
    # Overall pass/fail
    critical_issues = (
        audit["participant_leakage"] or 
        audit["session_leakage"] or
        not audit["label_distribution_valid"]
    )
    audit["PASS"] = not critical_issues
    
    return audit


# ===========================================================================
# NORMALIZATION
# ===========================================================================

def normalize_features(X_train, X_val, X_test):
    """
    Z-score normalization using train-set statistics only.
    Returns normalized arrays and the normalization parameters.
    """
    # Reshape to (N, 10*7) for statistics
    n_train = X_train.shape[0]
    n_val = X_val.shape[0]
    n_test = X_test.shape[0]
    
    flat_train = X_train.reshape(-1, NUM_FEATURES)  # (N*10, 7)
    
    mean = flat_train.mean(axis=0)
    std = flat_train.std(axis=0)
    std[std < 1e-8] = 1.0  # Prevent division by zero
    
    X_train_norm = (X_train - mean) / std
    X_val_norm = (X_val - mean) / std
    X_test_norm = (X_test - mean) / std
    
    return X_train_norm, X_val_norm, X_test_norm, mean, std


# ===========================================================================
# MAIN PIPELINE
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Freihaut & Göritz (2021) Keyboard Stress Preprocessing Pipeline"
    )
    parser.add_argument(
        "--raw-data-path", 
        default="data/freihaut_git",
        help="Path to the cloned Freihaut git repository"
    )
    parser.add_argument(
        "--output-path",
        default="data/processed/freihaut",
        help="Output directory for processed arrays"
    )
    parser.add_argument(
        "--lab-only",
        action="store_true",
        help="Process only the Lab Study (faster, 53 participants)"
    )
    parser.add_argument(
        "--skip-online",
        action="store_true",
        help="Skip the Online Study to avoid long loading time"
    )
    args = parser.parse_args()
    
    print("=" * 70)
    print("FREIHAUT & GÖRITZ (2021) KEYBOARD STRESS PREPROCESSING")
    print("=" * 70)
    print()
    
    # -----------------------------------------------------------------------
    # Step 1: Load raw data
    # -----------------------------------------------------------------------
    print("[STEP 1] Loading raw data...")
    all_participants = {}
    
    # Always load Lab Study
    lab_participants = load_lab_study(args.raw_data_path)
    all_participants.update(lab_participants)
    
    # Optionally load Online Study
    if not args.lab_only and not args.skip_online:
        online_participants = load_online_study(args.raw_data_path)
        all_participants.update(online_participants)
    else:
        print("  [SKIP] Online Study skipped (use without --lab-only/--skip-online to include)")
    
    print(f"\n  Total usable participants: {len(all_participants)}")
    
    if len(all_participants) < 10:
        print("[FATAL] Insufficient participants for meaningful split. Aborting.")
        sys.exit(1)
    
    # -----------------------------------------------------------------------
    # Step 2: Generate temporal sequences per participant/condition
    # -----------------------------------------------------------------------
    print("\n[STEP 2] Generating temporal sequences...")
    
    # participant_id -> [(sequence, label)]
    participant_sequences = {}
    total_sequences = 0
    total_stress = 0
    total_nonstress = 0
    
    for pid, conditions in all_participants.items():
        seqs = []
        for cond_key, (pairs, label) in conditions.items():
            sequences = create_temporal_sequences(pairs)
            for seq in sequences:
                seqs.append((seq, label))
                if label == 1:
                    total_stress += 1
                else:
                    total_nonstress += 1
        
        if len(seqs) > 0:
            participant_sequences[pid] = seqs
            total_sequences += len(seqs)
    
    print(f"  Participants with sequences: {len(participant_sequences)}")
    print(f"  Total sequences: {total_sequences}")
    print(f"  Stress (label=1): {total_stress}")
    print(f"  Non-stress (label=0): {total_nonstress}")
    
    if total_sequences < 20:
        print("[FATAL] Insufficient sequences for training. Aborting.")
        sys.exit(1)
    
    # -----------------------------------------------------------------------
    # Step 3: Participant-level split
    # -----------------------------------------------------------------------
    print("\n[STEP 3] Participant-level splitting (70/15/15)...")
    
    pids = list(participant_sequences.keys())
    train_ids, val_ids, test_ids = split_participants(pids)
    
    print(f"  Train participants: {len(train_ids)}")
    print(f"  Val participants:   {len(val_ids)}")
    print(f"  Test participants:  {len(test_ids)}")
    
    # Verify no overlap
    assert len(set(train_ids) & set(val_ids)) == 0, "TRAIN  INTERSECT  VAL != EMPTY"
    assert len(set(train_ids) & set(test_ids)) == 0, "TRAIN  INTERSECT  TEST != EMPTY"
    assert len(set(val_ids) & set(test_ids)) == 0, "VAL  INTERSECT  TEST != EMPTY"
    print("  [OK] No participant overlap between splits")
    
    # -----------------------------------------------------------------------
    # Step 4: Assemble arrays
    # -----------------------------------------------------------------------
    print("\n[STEP 4] Assembling arrays...")
    
    def collect_split(ids):
        X_list, y_list, pid_list = [], [], []
        for pid in ids:
            for seq, label in participant_sequences[pid]:
                X_list.append(seq)
                y_list.append(label)
                pid_list.append(pid)
        if len(X_list) == 0:
            return np.zeros((0, NUM_TIMESTEPS, NUM_FEATURES), dtype=np.float32), \
                   np.zeros((0,), dtype=np.int64), []
        return np.stack(X_list).astype(np.float32), np.array(y_list, dtype=np.int64), pid_list
    
    X_train, y_train, pids_train = collect_split(train_ids)
    X_val, y_val, pids_val = collect_split(val_ids)
    X_test, y_test, pids_test = collect_split(test_ids)
    
    print(f"  Train: {X_train.shape} | stress={np.sum(y_train==1)} | non-stress={np.sum(y_train==0)}")
    print(f"  Val:   {X_val.shape} | stress={np.sum(y_val==1)} | non-stress={np.sum(y_val==0)}")
    print(f"  Test:  {X_test.shape} | stress={np.sum(y_test==1)} | non-stress={np.sum(y_test==0)}")
    
    # Check we have both classes in each split
    for name, y in [("train", y_train), ("val", y_val), ("test", y_test)]:
        if len(np.unique(y)) < 2:
            warnings.warn(f"[WARNING] {name} split has only one class! Consider adjusting split.")
    
    # -----------------------------------------------------------------------
    # Step 5: Normalize
    # -----------------------------------------------------------------------
    print("\n[STEP 5] Normalizing features (z-score from train set)...")
    X_train_norm, X_val_norm, X_test_norm, feat_mean, feat_std = normalize_features(
        X_train, X_val, X_test
    )
    print(f"  Feature means: {np.round(feat_mean, 3)}")
    print(f"  Feature stds:  {np.round(feat_std, 3)}")
    
    # -----------------------------------------------------------------------
    # Step 6: Leakage audit
    # -----------------------------------------------------------------------
    print("\n[STEP 6] Running leakage audit...")
    leakage_audit = check_leakage(
        train_ids, val_ids, test_ids,
        X_train_norm, X_val_norm, X_test_norm,
        y_train, y_val, y_test
    )
    
    if leakage_audit["PASS"]:
        print("  [OK] LEAKAGE AUDIT PASSED")
    else:
        print("  [FAIL] LEAKAGE AUDIT FAILED")
        print(f"  Details: {json.dumps(leakage_audit['details'], indent=2)}")
        print("[FATAL] Leakage detected. Fix before training.")
        sys.exit(1)
    
    if leakage_audit.get("duplicate_windows"):
        print(f"  [NOTE] Cross-split duplicates found (likely similar typing patterns)")
        print(f"         {json.dumps(leakage_audit['details'].get('cross_split_duplicates', {}))}")
    
    if leakage_audit.get("identical_feature_vectors"):
        print(f"  [NOTE] Within-split duplicates found:")
        print(f"         {json.dumps(leakage_audit['details'].get('within_split_duplicates', {}))}")
    
    # -----------------------------------------------------------------------
    # Step 7: Save outputs
    # -----------------------------------------------------------------------
    print(f"\n[STEP 7] Saving to {args.output_path}...")
    os.makedirs(args.output_path, exist_ok=True)
    
    np.save(os.path.join(args.output_path, "X_train.npy"), X_train_norm)
    np.save(os.path.join(args.output_path, "y_train.npy"), y_train)
    np.save(os.path.join(args.output_path, "X_val.npy"), X_val_norm)
    np.save(os.path.join(args.output_path, "y_val.npy"), y_val)
    np.save(os.path.join(args.output_path, "X_test.npy"), X_test_norm)
    np.save(os.path.join(args.output_path, "y_test.npy"), y_test)
    
    # Save participant split info
    split_info = {
        "train": sorted(train_ids),
        "val": sorted(val_ids),
        "test": sorted(test_ids),
    }
    with open(os.path.join(args.output_path, "participant_split.json"), "w") as f:
        json.dump(split_info, f, indent=2)
    
    # Save normalization parameters
    norm_params = {
        "mean": feat_mean.tolist(),
        "std": feat_std.tolist(),
        "feature_names": FEATURE_NAMES,
    }
    with open(os.path.join(args.output_path, "normalization_params.json"), "w") as f:
        json.dump(norm_params, f, indent=2)
    
    # Save leakage audit
    audit_path = os.path.join("results", "freihaut_leakage_audit.json")
    os.makedirs("results", exist_ok=True)
    with open(audit_path, "w") as f:
        json.dump(leakage_audit, f, indent=2)
    print(f"  Saved leakage audit: {audit_path}")
    
    # Save preprocessing metadata
    metadata = {
        "dataset": "Freihaut & Göritz (2021) Keyboard Stress",
        "source": "Zenodo 10.5281/zenodo.4445197",
        "participants_total": len(participant_sequences),
        "participants_train": len(train_ids),
        "participants_val": len(val_ids),
        "participants_test": len(test_ids),
        "train_windows": int(len(y_train)),
        "val_windows": int(len(y_val)),
        "test_windows": int(len(y_test)),
        "stress_windows": int(total_stress),
        "nonstress_windows": int(total_nonstress),
        "feature_count": NUM_FEATURES,
        "feature_names": FEATURE_NAMES,
        "window_length": NUM_TIMESTEPS,
        "keystrokes_per_subwindow": KEYSTROKES_PER_SUBWINDOW,
        "stride_subwindows": STRIDE_SUBWINDOWS,
        "min_keystrokes_per_trial": MIN_KEYSTROKES_PER_TRIAL,
        "random_seed": RANDOM_SEED,
        "split_method": "participant-independent (70/15/15)",
        "normalization": "z-score (train-set statistics)",
        "label_mapping": {"0": "NON-STRESS (Low-Stress condition)", "1": "STRESS (High-Stress condition)"},
        "tensor_shape": f"(N, {NUM_TIMESTEPS}, {NUM_FEATURES})",
        "feature_definitions": {
            "mean_dwell": "Mean dwell time in ms (KeyDown to KeyUp)",
            "std_dwell": "Std dev of dwell time in ms",
            "mean_flight": "Mean flight time in ms (KeyUp to next KeyDown)",
            "std_flight": "Std dev of flight time in ms",
            "typing_speed": "Keystrokes per second in sub-window",
            "backspace_freq": "Fraction of keystrokes that are Backspace (0-1)",
            "error_rate": "Fraction of keystrokes marked incorrect (0-1)",
        },
        "leakage_audit_pass": leakage_audit["PASS"],
    }
    
    meta_path = os.path.join("results", "freihaut_preprocessing_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved metadata: {meta_path}")
    
    # -----------------------------------------------------------------------
    # Step 8: Sanity check - print real examples
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SANITY CHECK - REAL EXAMPLES")
    print("=" * 70)
    
    print(f"\nTRAIN:")
    print(f"  Shape: {X_train_norm.shape}")
    print(f"  Class counts: stress={np.sum(y_train==1)}, non-stress={np.sum(y_train==0)}")
    print(f"  Unique participants: {len(set(pids_train))}")
    
    print(f"\nVALIDATION:")
    print(f"  Shape: {X_val_norm.shape}")
    print(f"  Class counts: stress={np.sum(y_val==1)}, non-stress={np.sum(y_val==0)}")
    print(f"  Unique participants: {len(set(pids_val))}")
    
    print(f"\nTEST:")
    print(f"  Shape: {X_test_norm.shape}")
    print(f"  Class counts: stress={np.sum(y_test==1)}, non-stress={np.sum(y_test==0)}")
    print(f"  Unique participants: {len(set(pids_test))}")
    
    # Print a few real examples (unnormalized for interpretability)
    print(f"\nSAMPLE EXAMPLES (unnormalized feature values):")
    for split_name, X_raw, y, pids in [
        ("TRAIN", X_train, y_train, pids_train),
        ("VAL", X_val, y_val, pids_val),
        ("TEST", X_test, y_test, pids_test),
    ]:
        if len(X_raw) > 0:
            idx = 0
            label_str = "STRESS" if y[idx] == 1 else "NON-STRESS"
            print(f"\n  [{split_name}] Participant: {pids[idx]}, Label: {label_str}")
            print(f"  Timestep features ({FEATURE_NAMES}):")
            for t in range(min(3, X_raw.shape[1])):
                vals = X_raw[idx, t, :]
                print(f"    t={t}: {np.round(vals, 2)}")
            print(f"    ... ({X_raw.shape[1]} timesteps total)")
    
    # -----------------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE")
    print("=" * 70)
    print(f"  Dataset:          {metadata['dataset']}")
    print(f"  Participants:     {metadata['participants_total']}")
    print(f"  Train/Val/Test:   {metadata['participants_train']}/{metadata['participants_val']}/{metadata['participants_test']} participants")
    print(f"  Windows:          {metadata['train_windows']}/{metadata['val_windows']}/{metadata['test_windows']}")
    print(f"  Stress/NonStress: {metadata['stress_windows']}/{metadata['nonstress_windows']}")
    print(f"  Tensor shape:     {metadata['tensor_shape']}")
    print(f"  Features:         {metadata['feature_names']}")
    print(f"  Leakage audit:    {'PASS' if metadata['leakage_audit_pass'] else 'FAIL'}")
    print(f"  Output dir:       {args.output_path}")
    print()
    
    return metadata


if __name__ == "__main__":
    main()
