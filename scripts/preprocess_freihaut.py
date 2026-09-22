#!/usr/bin/env python3
"""
Freihaut & Goeritz (2021) Keyboard Stress Preprocessing Pipeline
================================================================

Dataset: "Does People's Keyboard Typing Reflect Their Stress Level"
Source:  Zenodo 10.5281/zenodo.4445197 (data/freihaut_git)

RAW KEY EVENTS (PatternTyping trials, cached by cache_freihaut_trials.py)
    -> KeyDown/KeyUp pairing          (src/keyboard_stress_features.py)
    -> 7D features per 4-keystroke sub-window
    -> 10 consecutive sub-windows = one (10, 7) sample
    -> participant-level stratified split 70/15/15
    -> duplicate removal + leakage audit
    -> RAW (unnormalised) arrays + participant/trial ids

Normalisation is NOT done here: train.py fits the scaler on the training
split only and saves it next to the checkpoint.

LABELS (experimental condition only)
    Lab study (within-subject):   HS_PatternTyping -> 1, LS_PatternTyping -> 0
    Online study (between-subj.): Con_PatternTyping, condition 0 (HS) -> 1,
                                  condition 1 (LS) -> 0
    Practice trials (lab PR_, online Pr_) are pre-manipulation and are NOT
    labelled. (Earlier versions labelled every online practice trial as
    NON-STRESS, which created a 3:1 imbalance and a practice-vs-task confound.)
    Only participants who passed the authors' study-level filter are used.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import pickle
import subprocess
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from keyboard_stress_features import (  # noqa: E402
    FEATURE_NAMES, KEYSTROKES_PER_SUBWINDOW, NUM_FEATURES, NUM_TIMESTEPS,
    events_to_pairs, pairs_to_subwindow_features, subwindows_to_sequences,
)

RANDOM_SEED = 42
TRAIN_RATIO, VAL_RATIO = 0.70, 0.15
CACHE = os.path.join(ROOT, "data", "processed", "freihaut", "pattern_typing_trials.pkl.gz")


def trial_label(trial):
    if trial["study"] == "lab":
        return {"HS_PatternTyping": 1, "LS_PatternTyping": 0}.get(trial["task"])
    if trial["task"] == "Con_PatternTyping" and trial["condition"] in (0, 1):
        return 1 if trial["condition"] == 0 else 0
    return None


def load_trials():
    if not os.path.exists(CACHE):
        print("  Trial cache missing -> running scripts/cache_freihaut_trials.py (one-time, ~3 min)")
        subprocess.check_call([sys.executable, os.path.join(ROOT, "scripts", "cache_freihaut_trials.py")], cwd=ROOT)
    with gzip.open(CACHE, "rb") as f:
        return pickle.load(f)


def split_participants(groups, seed=RANDOM_SEED):
    """Participant-level split, stratified by (study, participant label pattern)."""
    rng = np.random.RandomState(seed)
    train, val, test = [], [], []
    for key in sorted(groups):
        ids = sorted(groups[key])
        rng.shuffle(ids)
        n = len(ids)
        n_tr, n_va = int(round(n * TRAIN_RATIO)), int(round(n * VAL_RATIO))
        train += ids[:n_tr]
        val += ids[n_tr:n_tr + n_va]
        test += ids[n_tr + n_va:]
    return train, val, test


def _hash(a):
    return hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-path", default=os.path.join(ROOT, "data", "processed", "freihaut"))
    args = ap.parse_args()

    print("=" * 70)
    print("FREIHAUT & GOERITZ (2021) KEYBOARD STRESS PREPROCESSING")
    print("=" * 70)
    trials = load_trials()

    # ---- sequences per labelled trial --------------------------------------
    per_pid = defaultdict(list)  # pid -> [(seq, label, trial_id)]
    dropped_short = 0
    for t in trials:
        label = trial_label(t)
        if label is None or not t["completed"]:
            continue
        pairs = events_to_pairs(t["events"])
        seqs = subwindows_to_sequences(pairs_to_subwindow_features(pairs))
        if not seqs:
            dropped_short += 1
            continue
        tid = f"{t['pid']}|{t['task']}"
        per_pid[t["pid"]] += [(s, label, tid) for s in seqs]
    print(f"  Participants with usable labelled trials: {len(per_pid)}  (trials too short: {dropped_short})")

    # ---- stratified participant split --------------------------------------
    groups = defaultdict(list)
    for pid, items in per_pid.items():
        labels = tuple(sorted({l for _, l, _ in items}))
        groups[(pid.split("_")[0], labels)].append(pid)
    train_ids, val_ids, test_ids = split_participants(groups)
    assert not (set(train_ids) & set(val_ids)) and not (set(train_ids) & set(test_ids)) \
        and not (set(val_ids) & set(test_ids)), "participant leakage"

    # ---- assemble + dedupe -------------------------------------------------
    seen = {}
    audit = {"within_split_duplicates_removed": {}, "cross_split_duplicates_removed": {}}
    out = {}
    for name, ids in (("train", train_ids), ("val", val_ids), ("test", test_ids)):
        X, y, P, T = [], [], [], []
        within = cross = 0
        for pid in ids:
            for seq, label, tid in per_pid[pid]:
                h = _hash(seq)
                if h in seen:
                    if seen[h] == name:
                        within += 1
                    else:
                        cross += 1
                    continue
                seen[h] = name
                X.append(seq); y.append(label); P.append(pid); T.append(tid)
        audit["within_split_duplicates_removed"][name] = within
        audit["cross_split_duplicates_removed"][name] = cross
        out[name] = (np.stack(X).astype(np.float32), np.array(y, dtype=np.int64), np.array(P), np.array(T))

    for name, (X, y, P, T) in out.items():
        assert X.shape[1:] == (NUM_TIMESTEPS, NUM_FEATURES)
        assert np.isfinite(X).all(), f"non-finite values in {name}"
        assert len(np.unique(y)) == 2, f"{name} split has a single class"
        print(f"  {name:5s}: X={X.shape} participants={len(set(P))} trials={len(set(T))} "
              f"stress={int((y == 1).sum())} non-stress={int((y == 0).sum())}")

    audit.update({
        "participant_overlap": {
            "train_val": len(set(train_ids) & set(val_ids)),
            "train_test": len(set(train_ids) & set(test_ids)),
            "val_test": len(set(val_ids) & set(test_ids)),
        },
        "trial_overlap": int(len(set(out["train"][3]) & (set(out["val"][3]) | set(out["test"][3])))),
        "synthetic_samples": 0,
        "normalization": "none in preprocessing; scaler fitted on train split only in train.py",
    })
    audit["PASS"] = (sum(audit["participant_overlap"].values()) == 0 and audit["trial_overlap"] == 0)
    print(f"  Leakage audit: {'PASS' if audit['PASS'] else 'FAIL'}  {json.dumps(audit['within_split_duplicates_removed'])}")
    if not audit["PASS"]:
        sys.exit(1)

    # ---- save --------------------------------------------------------------
    os.makedirs(args.output_path, exist_ok=True)
    for stale in ("normalization_params.json",):
        p = os.path.join(args.output_path, stale)
        if os.path.exists(p):
            os.remove(p)  # normalisation now lives with the checkpoint (train-only fit)
    for name, (X, y, P, T) in out.items():
        np.save(os.path.join(args.output_path, f"X_{name}.npy"), X)
        np.save(os.path.join(args.output_path, f"y_{name}.npy"), y)
        np.save(os.path.join(args.output_path, f"pid_{name}.npy"), P)
        np.save(os.path.join(args.output_path, f"trial_{name}.npy"), T)
    with open(os.path.join(args.output_path, "participant_split.json"), "w") as f:
        json.dump({"train": sorted(train_ids), "val": sorted(val_ids), "test": sorted(test_ids)}, f, indent=2)
    with open(os.path.join(ROOT, "results", "freihaut_leakage_audit.json"), "w") as f:
        json.dump(audit, f, indent=2)

    n_stress = sum(int((o[1] == 1).sum()) for o in out.values())
    n_total = sum(len(o[1]) for o in out.values())
    meta = {
        "dataset": "Freihaut & Goeritz (2021) Keyboard Stress",
        "source": "Zenodo 10.5281/zenodo.4445197",
        "participants_total": len(per_pid),
        "participants_train": len(train_ids),
        "participants_val": len(val_ids),
        "participants_test": len(test_ids),
        "train_windows": int(len(out["train"][1])),
        "val_windows": int(len(out["val"][1])),
        "test_windows": int(len(out["test"][1])),
        "stress_windows": n_stress,
        "nonstress_windows": n_total - n_stress,
        "feature_count": NUM_FEATURES,
        "feature_names": FEATURE_NAMES,
        "window_length": NUM_TIMESTEPS,
        "keystrokes_per_subwindow": KEYSTROKES_PER_SUBWINDOW,
        "stride_subwindows": 1,
        "random_seed": RANDOM_SEED,
        "split_method": "participant-independent, stratified by study and label pattern (70/15/15)",
        "normalization": "none (raw features); train-only scaler saved by train.py",
        "label_mapping": {"0": "NON-STRESS (low-stress condition)", "1": "STRESS (high-stress condition)"},
        "label_source": "lab HS/LS PatternTyping; online Con_PatternTyping by assigned condition; practice trials excluded",
        "tensor_shape": f"(N, {NUM_TIMESTEPS}, {NUM_FEATURES})",
        "leakage_audit_pass": audit["PASS"],
    }
    with open(os.path.join(ROOT, "results", "freihaut_preprocessing_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved arrays to {args.output_path}")
    return meta


if __name__ == "__main__":
    main()
