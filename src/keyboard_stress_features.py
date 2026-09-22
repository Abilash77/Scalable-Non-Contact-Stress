"""
Shared keyboard-stress feature pipeline (training AND runtime).

Both the Freihaut & Goeritz preprocessing and the live browser keyboard worker
use the functions in this module, so the model always sees the same features:

    KeyDown/KeyUp events (ms)
        -> cleaned + paired keystrokes
        -> non-overlapping sub-windows of KEYSTROKES_PER_SUBWINDOW keystrokes
        -> 7D feature vector per sub-window
        -> 10 consecutive sub-windows = one (10, 7) sample
        -> KeyboardFeatureTransform (log timing features, train-fitted z-score)
        -> trained keyboard stress model

Feature order (7D):
    mean_dwell, std_dwell, mean_flight, std_flight, typing_speed,
    backspace_freq, error_rate

error_rate in the dataset is "keystroke did not match the target pattern".
Free typing in the browser has no target pattern, so this value cannot be
computed at runtime. It is therefore excluded from the model (the transform
sets it to a constant 0 for both training and runtime) and is kept in the
7D vector only for shape compatibility.
"""
from __future__ import annotations

import json
import operator
import os

import numpy as np

KEYSTROKES_PER_SUBWINDOW = 4
NUM_TIMESTEPS = 10
FEATURE_NAMES = [
    "mean_dwell",
    "std_dwell",
    "mean_flight",
    "std_flight",
    "typing_speed",
    "backspace_freq",
    "error_rate",
]
NUM_FEATURES = len(FEATURE_NAMES)
# Heavy-tailed millisecond / rate features are log-transformed before z-scoring.
LOG_FEATURES = [0, 1, 3, 4]
# mean_flight can be negative (key overlap), so it uses a signed log.
SIGNED_LOG_FEATURES = [2]
UNAVAILABLE_AT_RUNTIME = [6]  # error_rate


# ---------------------------------------------------------------------------
# Event cleaning / pairing (adapted from Freihaut's get_typing_data())
# ---------------------------------------------------------------------------

def clean_keyboard_events(raw_events):
    """Filter to KeyDown/KeyUp, drop Shift, pair downs with ups, sort by time."""
    key_events = [e for e in raw_events
                  if isinstance(e, dict) and e.get("eventType") in ("KeyDown", "KeyUp")]
    if len(key_events) < 4:
        return []

    keyboard_events = []
    key_down_events = {}
    for event in key_events:
        if event.get("key") == "Shift":
            continue
        code = event["code"]
        if event["eventType"] == "KeyDown":
            if code not in key_down_events:
                key_down_events[code] = [event]
            elif event["time"] - key_down_events[code][-1]["time"] < 1750:
                key_down_events[code].append(event)  # key repeat
            else:
                key_down_events[code] = [event]  # stale KeyDown without KeyUp
        else:
            if key_down_events.get(code):
                keyboard_events.append(key_down_events[code][0])
                keyboard_events.append(event)
                del key_down_events[code]
    return sorted(keyboard_events, key=operator.itemgetter("time"))


def extract_keystroke_pairs(clean_events):
    """Turn sorted KeyDown/KeyUp events into keystroke dicts."""
    pairs = []
    i = 0
    while i < len(clean_events) - 1:
        down, up = clean_events[i], clean_events[i + 1]
        if (down["eventType"] == "KeyDown" and up["eventType"] == "KeyUp"
                and down["code"] == up["code"]):
            dwell = up["time"] - down["time"]
            if 0 < dwell < 5000:
                pairs.append({
                    "code": down["code"],
                    "key": down.get("key", ""),
                    "down_time": down["time"],
                    "up_time": up["time"],
                    "dwell_time": float(dwell),
                    "is_correct": down.get("isCorrect", True),
                    "is_backspace": "Backspace" in down.get("code", ""),
                })
            i += 2
        else:
            i += 1
    return pairs


def events_to_pairs(raw_events):
    return extract_keystroke_pairs(clean_keyboard_events(raw_events))


# ---------------------------------------------------------------------------
# 7D sub-window features
# ---------------------------------------------------------------------------

def compute_flight_times(pairs):
    """Flight = next KeyDown - previous KeyUp (ms); pauses >= 10 s are dropped."""
    flights = []
    for j in range(len(pairs) - 1):
        flight = pairs[j + 1]["down_time"] - pairs[j]["up_time"]
        if abs(flight) < 10000:
            flights.append(float(flight))
    return flights


def compute_subwindow_features(pairs):
    """7D feature vector for one sub-window of keystrokes, or None if unusable."""
    if len(pairs) < 3:
        return None
    dwell = [p["dwell_time"] for p in pairs]
    flight = compute_flight_times(pairs)
    if len(flight) < 2:
        return None
    span_s = (pairs[-1]["up_time"] - pairs[0]["down_time"]) / 1000.0
    if span_s <= 0:
        return None
    n = len(pairs)
    return np.array([
        np.mean(dwell),
        np.std(dwell),
        np.mean(flight),
        np.std(flight),
        n / span_s,
        sum(p["is_backspace"] for p in pairs) / n,
        sum(not p["is_correct"] for p in pairs) / n,
    ], dtype=np.float32)


def pairs_to_subwindow_features(pairs, k=KEYSTROKES_PER_SUBWINDOW):
    feats = []
    for start in range(0, len(pairs) - k + 1, k):
        f = compute_subwindow_features(pairs[start:start + k])
        if f is not None:
            feats.append(f)
    return feats


def subwindows_to_sequences(subwindow_feats, stride=1):
    if len(subwindow_feats) < NUM_TIMESTEPS:
        return []
    return [np.stack(subwindow_feats[s:s + NUM_TIMESTEPS])
            for s in range(0, len(subwindow_feats) - NUM_TIMESTEPS + 1, stride)]


# ---------------------------------------------------------------------------
# Browser events -> dataset event format
# ---------------------------------------------------------------------------

def browser_events_to_raw(events):
    """
    Convert dashboard key events {key, code, action: press|release, time (s)}
    into the dataset's {eventType, code, key, time (ms)} format.
    Auto-repeat presses (repeat=True) are dropped.
    """
    raw = []
    for e in events:
        action = e.get("action")
        if action not in ("press", "release") or e.get("repeat"):
            continue
        key = e.get("key", "")
        code = e.get("code") or {"Key.backspace": "Backspace", "Key.delete": "Delete"}.get(key, key)
        if key in ("Shift", "Key.shift") or code.startswith("Shift"):
            key = "Shift"
        raw.append({
            "eventType": "KeyDown" if action == "press" else "KeyUp",
            "code": code,
            "key": key,
            "time": float(e["time"]) * 1000.0,
            "isCorrect": True,  # no target pattern in free typing (feature excluded)
        })
    raw.sort(key=operator.itemgetter("time"))
    return raw


# ---------------------------------------------------------------------------
# Train-fitted transform (the "scaler")
# ---------------------------------------------------------------------------

class KeyboardFeatureTransform:
    """log/signed-log on timing features, drop error_rate, z-score (train stats)."""

    def __init__(self, mean=None, std=None):
        self.mean = None if mean is None else np.asarray(mean, dtype=np.float32)
        self.std = None if std is None else np.asarray(std, dtype=np.float32)

    @staticmethod
    def _pre(X):
        X = np.array(X, dtype=np.float32, copy=True)
        X[..., LOG_FEATURES] = np.log1p(np.clip(X[..., LOG_FEATURES], 0, None))
        X[..., SIGNED_LOG_FEATURES] = np.sign(X[..., SIGNED_LOG_FEATURES]) * np.log1p(
            np.abs(X[..., SIGNED_LOG_FEATURES]))
        X[..., UNAVAILABLE_AT_RUNTIME] = 0.0
        return X

    def fit(self, X_train):
        flat = self._pre(X_train).reshape(-1, NUM_FEATURES)
        self.mean = flat.mean(axis=0)
        self.std = flat.std(axis=0)
        self.std[self.std < 1e-6] = 1.0
        return self

    def transform(self, X):
        if self.mean is None:
            raise RuntimeError("KeyboardFeatureTransform used before fit()")
        return ((self._pre(X) - self.mean) / self.std).astype(np.float32)

    def to_dict(self):
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "log_features": LOG_FEATURES,
            "signed_log_features": SIGNED_LOG_FEATURES,
            "zeroed_features": UNAVAILABLE_AT_RUNTIME,
            "feature_names": FEATURE_NAMES,
        }

    @classmethod
    def from_dict(cls, d):
        if (d.get("log_features") != LOG_FEATURES
                or d.get("signed_log_features") != SIGNED_LOG_FEATURES
                or d.get("zeroed_features") != UNAVAILABLE_AT_RUNTIME):
            raise ValueError("Scaler was fitted with a different feature transform")
        return cls(d["mean"], d["std"])


def summary_features(Xn):
    """(N, 10, 7) normalised sequences -> (N, 30) per-feature mean/std/min/max/slope,
    excluding the zeroed error_rate column."""
    keep = [i for i in range(NUM_FEATURES) if i not in UNAVAILABLE_AT_RUNTIME]
    X = Xn[..., keep]
    t = np.arange(NUM_TIMESTEPS, dtype=np.float32)
    t = (t - t.mean()) / ((t - t.mean()) ** 2).sum()
    slope = np.einsum("t,ntf->nf", t, X)
    return np.concatenate([X.mean(1), X.std(1), X.min(1), X.max(1), slope], axis=1)


# ---------------------------------------------------------------------------
# Runtime predictor
# ---------------------------------------------------------------------------

DEFAULT_BUNDLE = os.path.join("results", "checkpoints", "keyboard_stress_model_metadata.json")


class KeyboardStressPredictor:
    """Loads the validated keyboard stress model + scaler written by train.py."""

    def __init__(self, metadata_path=DEFAULT_BUNDLE):
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)
        if not self.meta.get("is_trained"):
            raise ValueError("Keyboard stress metadata does not describe a trained model")
        base = os.path.dirname(metadata_path)
        with open(os.path.join(base, self.meta["scaler_file"]), "r", encoding="utf-8") as f:
            self.transform = KeyboardFeatureTransform.from_dict(json.load(f))
        if self.meta.get("keystrokes_per_subwindow") != KEYSTROKES_PER_SUBWINDOW:
            raise ValueError("Checkpoint was trained with a different sub-window size")
        self.threshold = float(self.meta["decision_threshold"])
        self.input_kind = self.meta["input_kind"]  # "summary" or "sequence"
        model_path = os.path.join(base, self.meta["model_file"])
        fmt = self.meta["model_format"]
        self._fusion = False
        if fmt == "joblib":
            import joblib
            self.model = joblib.load(model_path)
            if hasattr(self.model, "n_jobs"):
                self.model.n_jobs = 1  # single-window runtime calls: avoid per-call thread-pool start-up
            self._keras = False
        elif fmt == "keras":
            import tensorflow as tf
            self.model = tf.keras.models.load_model(model_path, compile=False)
            self._keras = True
        elif fmt == "fusion_weights":
            from DL_models import get_model
            shapes = {"audio": (10, 169), "face": (10, 12), "keystroke": (10, 7),
                      "handwriting": (10, 9), "eye": (10, 5)}
            self.model = get_model("fusion", input_shapes=shapes, num_classes=2)
            self.model.load_weights(model_path)
            self._keras = self._fusion = True
        else:
            raise ValueError(f"Unknown keyboard model format: {fmt}")
        self.name = self.meta.get("model_name", "keyboard_stress")

    def predict_proba(self, X_raw):
        """X_raw: (N, 10, 7) or (10, 7) raw 7D features -> P(stress) per sample."""
        X_raw = np.asarray(X_raw, dtype=np.float32)
        if X_raw.ndim == 2:
            X_raw = X_raw[None]
        if X_raw.shape[1:] != (NUM_TIMESTEPS, NUM_FEATURES):
            raise ValueError(f"Expected (N, 10, 7) keyboard input, got {X_raw.shape}")
        Xn = self.transform.transform(X_raw)
        X = summary_features(Xn) if self.input_kind == "summary" else Xn
        if self._fusion:
            n = len(X)
            mask = np.zeros((n, 5), np.float32)
            mask[:, 2] = 1.0
            out = self.model({"input_audio": np.zeros((n, 10, 169), np.float32),
                              "input_face": np.zeros((n, 10, 12), np.float32),
                              "input_keystroke": X,
                              "input_handwriting": np.zeros((n, 10, 9), np.float32),
                              "input_eye": np.zeros((n, 10, 5), np.float32),
                              "input_mask": mask}, training=False)
            return np.asarray(out["fusion_output"])[:, 1].astype(float)
        if self._keras:
            p = np.asarray(self.model(X, training=False)).reshape(len(X), -1)
            return p[:, -1].astype(float)
        return self.model.predict_proba(X)[:, 1].astype(float)
