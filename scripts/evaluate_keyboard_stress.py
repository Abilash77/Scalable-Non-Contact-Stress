#!/usr/bin/env python3
"""
Re-evaluate the SAVED keyboard stress checkpoint on the untouched test split
through the runtime loader (KeyboardStressPredictor), and add participant-level
bootstrap 95% confidence intervals. No training or tuning happens here.
"""
import json
import os
import sys

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.chdir(ROOT)
from keyboard_stress_features import KeyboardStressPredictor  # noqa: E402

D = os.path.join("data", "processed", "freihaut")
X, y, pid = (np.load(os.path.join(D, f"{k}_test.npy")) for k in ("X", "y", "pid"))
pred = KeyboardStressPredictor()
prob = pred.predict_proba(X)
thr = pred.threshold
saved = pred.meta["test_metrics"]

auc = roc_auc_score(y, prob)
bal = balanced_accuracy_score(y, (prob >= thr).astype(int))
acc = float(((prob >= thr).astype(int) == y).mean())
assert abs(auc - saved["roc_auc"]) < 1e-6 and abs(bal - saved["balanced_accuracy"]) < 1e-6, \
    "runtime loader does not reproduce training-time test metrics"

rng = np.random.RandomState(0)
ids = np.unique(pid)
idx_by = {p: np.where(pid == p)[0] for p in ids}
boot = {"roc_auc": [], "balanced_accuracy": [], "accuracy": []}
for _ in range(2000):
    idx = np.concatenate([idx_by[p] for p in rng.choice(ids, len(ids), replace=True)])
    if len(np.unique(y[idx])) < 2:
        continue
    pb = (prob[idx] >= thr).astype(int)
    boot["roc_auc"].append(roc_auc_score(y[idx], prob[idx]))
    boot["balanced_accuracy"].append(balanced_accuracy_score(y[idx], pb))
    boot["accuracy"].append(float((pb == y[idx]).mean()))
ci = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in boot.items()}

out = {"model": pred.name, "threshold": thr, "test_accuracy": acc, "test_balanced_accuracy": bal,
       "test_roc_auc": auc, "participant_bootstrap_95ci": ci, "n_windows": int(len(y)),
       "n_participants": int(len(ids)), "reproduced_through_runtime_loader": True}
with open(os.path.join("results", "keyboard_stress_test_ci.json"), "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
