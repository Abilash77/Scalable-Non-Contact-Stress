#!/usr/bin/env python3
"""
End-to-end runtime check against a RUNNING server (python run.py).

    PORT=5055 python run.py &
    python scripts/verify_runtime_e2e.py --base http://127.0.0.1:5055

1. KEYBOARD: replays REAL key events of held-out TEST participants through
   /api/upload_keystrokes in 0.5 s batches (as the browser does) and checks that
   the live /status features and probabilities equal the offline training
   pipeline + saved model for the same keystrokes.
2. FACE / EYE: real photo (test_face.jpg) -> /api/upload_frame -> FACE DETECTED
   + bbox, EYES DETECTED; blank frame -> NO FACE; stale camera detection.
3. SPEECH: two different real-valued waveforms -> RECEIVING AUDIO, 169D
   features that differ.
4. HANDWRITING: two different canvas images + stroke timings -> 9D features
   that differ, incl. real velocity/timing features.
"""
import argparse
import base64
import gzip
import json
import os
import pickle
import sys
import time
import urllib.request

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.chdir(ROOT)
from keyboard_stress_features import (  # noqa: E402
    KeyboardStressPredictor, events_to_pairs, pairs_to_subwindow_features, subwindows_to_sequences,
)

FAILS = []


def check(cond, msg):
    print(("  [PASS] " if cond else "  [FAIL] ") + msg)
    if not cond:
        FAILS.append(msg)


def post(base, path, payload):
    req = urllib.request.Request(base + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=10).read())


def status(base):
    return json.loads(urllib.request.urlopen(base + "/status", timeout=10).read())


def wait_for(base, pred, timeout=8.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        s = status(base)
        if pred(s):
            return s
        time.sleep(0.1)
    return status(base)


def to_browser_events(raw_events, start_id):
    out = []
    for i, e in enumerate(raw_events):
        if e.get("eventType") not in ("KeyDown", "KeyUp"):
            continue
        out.append({"event_id": start_id + i, "key": e.get("key", ""), "code": e.get("code", ""),
                    "repeat": False, "action": "press" if e["eventType"] == "KeyDown" else "release",
                    "time": e["time"] / 1000.0})
    return out


def keyboard_check(base, n_trials):
    print("\n[KEYBOARD] real held-out test-participant typing -> live server")
    predictor = KeyboardStressPredictor()
    test_pids = set(json.load(open("data/processed/freihaut/participant_split.json"))["test"])
    trials = pickle.load(gzip.open("data/processed/freihaut/pattern_typing_trials.pkl.gz"))
    chosen = []
    for t in trials:
        if t["pid"] in test_pids and t["task"] in ("HS_PatternTyping", "LS_PatternTyping", "Con_PatternTyping"):
            seqs = subwindows_to_sequences(pairs_to_subwindow_features(events_to_pairs(t["events"])))
            if len(seqs) >= 2:
                chosen.append((t, seqs))
        if len(chosen) >= n_trials:
            break
    all_live = []
    for k, (t, seqs) in enumerate(chosen):
        post(base, "/api/control", {"action": "reset"})
        time.sleep(0.3)
        offline = predictor.predict_proba(np.stack(seqs))
        events = to_browser_events(t["events"], start_id=10_000_000 * (k + 1))
        # 0.5 s batches in event time, like the dashboard's upload interval
        batches, cur, t_end = [], [], None
        for e in events:
            if t_end is None:
                t_end = e["time"] + 0.5
            if e["time"] > t_end:
                batches.append(cur)
                cur, t_end = [], e["time"] + 0.5
            cur.append(e)
        if cur:
            batches.append(cur)
        live_probs = []
        seen = 0
        for b in batches:
            post(base, "/api/upload_keystrokes", {"events": b})
            time.sleep(0.25)
            s = status(base)
            hist = s.get("prediction_history") or []
            for h in hist[seen:]:
                live_probs.append(h["stress_prob_pct"] / 100.0)
            seen = len(hist)
        s = wait_for(base, lambda s: (s.get("buffer_fills") or {}).get("keystroke", 0) >= 10)
        live_feat = (s.get("latest_features") or {}).get("keyboard_7d")
        off_last = seqs[-1][-1]
        # error_rate (index 6) needs the dataset's target pattern; it is zeroed for the model on both sides.
        feat_ok = live_feat is not None and np.allclose(live_feat[:6], np.round(off_last[:6], 3), atol=2e-3)
        check(feat_ok, f"{t['pid']} {t['task']}: live 7D features == training features (model-used 6) "
                       f"{np.round(off_last[:6], 1).tolist()}")
        match = all(np.min(np.abs(offline - p)) < 1.5e-3 for p in live_probs) and len(live_probs) > 0
        check(match, f"{t['pid']} {t['task']}: {len(live_probs)} live probabilities all equal offline model outputs "
                     f"(live={np.round(live_probs, 3).tolist()}, offline={np.round(offline, 3).tolist()})")
        check(abs(live_probs[-1] - offline[-1]) < 1.5e-3 if live_probs else False,
              f"{t['pid']} {t['task']}: final live P(stress)={live_probs[-1] if live_probs else None} "
              f"== offline {offline[-1]:.4f}; label={'STRESS' if s['prediction'] == 'STRESS' else 'NON-STRESS'} "
              f"@thr {predictor.threshold:.3f}")
        check(s["prediction"] in ("STRESS", "NON-STRESS") and s["stress_probability"] is not None,
              f"/status prediction={s['prediction']} stress_probability={s['stress_probability']}")
        all_live += live_probs
    check(len(set(np.round(all_live, 4))) > 1, f"different typing -> different probabilities ({len(set(np.round(all_live, 4)))} distinct)")

    # Reset clears stale prediction
    post(base, "/api/control", {"action": "reset"})
    s = wait_for(base, lambda s: s["prediction"] == "WAITING FOR KEYBOARD WINDOW")
    check(s["prediction"] == "WAITING FOR KEYBOARD WINDOW" and s["stress_probability"] is None,
          "reset -> WAITING FOR KEYBOARD WINDOW, no stale probability")
    # Too little typing never produces a prediction (no fallback)
    few = [{"event_id": 99_000_000 + i, "key": "a", "code": "KeyA", "repeat": False,
            "action": "press" if i % 2 == 0 else "release", "time": time.time() + i * 0.1} for i in range(10)]
    post(base, "/api/upload_keystrokes", {"events": few})
    time.sleep(0.6)
    s = status(base)
    check(s["prediction"] == "WAITING FOR KEYBOARD WINDOW" and s["stress_probability"] is None,
          "5 keystrokes only -> still WAITING (no random / hard-coded fallback)")


def jpeg_b64(img):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def camera_check(base):
    print("\n[FACE / EYE] real photo through /api/upload_frame")
    img = cv2.resize(cv2.imread("test_face.jpg"), (320, 240))
    shifted = np.roll(img, 40, axis=1)
    feats = []
    for frame in (img, shifted):
        for _ in range(3):
            post(base, "/api/upload_frame", {"image": jpeg_b64(frame)})
            time.sleep(0.35)
        s = wait_for(base, lambda s: s["face_detected"])
        feats.append((s["latest_features"]["face_12d"], s["latest_features"]["eye_5d"], s["face_bbox"]))
        check(s["face_detected"] and s["face_bbox"] and s["face_status"] == "DETECTED",
              f"FACE DETECTED, bbox={s['face_bbox']}")
        check(s["eye_status"] == "DETECTED" and s["eye_detected"], "EYES DETECTED")
        rms = s["runtime_modality_status"]
        check(rms.get("facial", {}).get("input") == "FACE DETECTED" and rms.get("eye_pupil", {}).get("input") == "EYES DETECTED",
              f"runtime status facial={rms.get('facial', {}).get('input')} eye={rms.get('eye_pupil', {}).get('input')}")
    check(feats[0][2] != feats[1][2], f"bbox follows the face ({feats[0][2]} -> {feats[1][2]})")
    check(feats[0][0] is not None and len(feats[0][0]) == 12 and len(feats[0][1]) == 5, "12D face + 5D eye features")
    blank = np.full((240, 320, 3), 40, np.uint8)
    for _ in range(2):
        post(base, "/api/upload_frame", {"image": jpeg_b64(blank)})
        time.sleep(0.35)
    s = wait_for(base, lambda s: not s["face_detected"])
    check(not s["face_detected"] and s["face_bbox"] is None, f"blank frame -> NO FACE DETECTED ({s['face_status']})")
    time.sleep(3.5)
    s = status(base)
    check(not s["camera_connected"] and s["face_status"] == "WAITING_FOR_CAMERA",
          "no frames for >3 s -> camera reported stale (no frozen status)")


def audio_check(base):
    print("\n[SPEECH] waveform through /api/upload_audio")
    sr = 16000
    t = np.arange(sr) / sr
    rng = np.random.RandomState(1)
    voiced = (0.2 * np.sin(2 * np.pi * 180 * t) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t))
              + 0.02 * rng.randn(sr)).astype(np.float32)
    noise = (0.05 * rng.randn(sr)).astype(np.float32)
    summaries = []
    for wav in (voiced, noise):
        prev = status(base).get("audio_sample_count", 0)
        post(base, "/api/upload_audio", {"audio": base64.b64encode(wav.tobytes()).decode()})
        s = wait_for(base, lambda s: s.get("audio_sample_count", 0) > prev)
        summaries.append(s["latest_features"]["speech_169d_summary"])
        check(s["mic_status"] == "RECEIVING_AUDIO", f"mic_status={s['mic_status']} summary={summaries[-1]}")
    check(summaries[0] != summaries[1], "different audio -> different 169D features")
    s = wait_for(base, lambda s: (s["runtime_modality_status"].get("speech") or {}).get("input") == "RECEIVING AUDIO")
    check((s["runtime_modality_status"].get("speech") or {}).get("input") == "RECEIVING AUDIO", "speech runtime status RECEIVING AUDIO")


def handwriting_check(base):
    print("\n[HANDWRITING] canvas image + strokes through /upload_handwriting")
    out = []
    for text, dt in (("hello", 8.0), ("stress test", 20.0)):
        c = np.full((160, 640, 3), 255, np.uint8)
        cv2.putText(c, text, (20, 100), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2, (0, 0, 0), 2)
        ok, buf = cv2.imencode(".png", c)
        strokes = [[{"x": 20 + 5 * i + 60 * s, "y": 100 + (i % 7), "t": 1000 * s + dt * i} for i in range(30)]
                   for s in range(4)]
        prev = status(base).get("handwriting_submission_count", 0)
        post(base, "/upload_handwriting", {"image": "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode(),
                                           "strokes": strokes})
        s = wait_for(base, lambda s: s.get("handwriting_submission_count", 0) > prev)
        f = s["latest_features"]["handwriting_9d"]
        out.append(f)
        check(f is not None and len(f) == 9 and f[7] > 0 and f[8] > 0,
              f"9D handwriting features incl. stroke velocity/timing: {f}")
    check(out[0] != out[1], "different handwriting -> different features")
    s = wait_for(base, lambda s: (s["runtime_modality_status"].get("handwriting") or {}).get("status") == "ACTIVE")
    check((s["runtime_modality_status"].get("handwriting") or {}).get("status") == "ACTIVE",
          "handwriting stays ACTIVE after submit (not wiped by the canvas clear)")


def model_status_check(base):
    print("\n[MODEL STATUS]")
    s = status(base)
    check(s["model_status"] == "TRAINED" and s["trained_modalities"] == ["keyboard"], "only keyboard is stress-trained")
    check(s["modality_attention"].get("speech", 0) == 0 and s["modality_attention"].get("facial", 0) == 0,
          "no emotion/personality model contributes to stress")
    ms = json.loads(urllib.request.urlopen(base + "/api/model_status", timeout=10).read())
    roles = {m["modality"]: m["training_label_type"] for m in ms}
    check(all(v.startswith("none") for k, v in roles.items() if k != "keyboard"), f"/api/model_status roles: {roles}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:5000")
    ap.add_argument("--trials", type=int, default=4)
    ap.add_argument("--only", nargs="*", default=["model", "keyboard", "camera", "audio", "handwriting"])
    args = ap.parse_args()
    post(args.base, "/api/control", {"action": "start"})
    try:
        if "model" in args.only:
            model_status_check(args.base)
        if "keyboard" in args.only:
            keyboard_check(args.base, args.trials)
        if "camera" in args.only:
            camera_check(args.base)
        if "audio" in args.only:
            audio_check(args.base)
        if "handwriting" in args.only:
            handwriting_check(args.base)
    finally:
        post(args.base, "/api/control", {"action": "stop"})
    print(f"\n{'ALL RUNTIME CHECKS PASSED' if not FAILS else f'{len(FAILS)} CHECK(S) FAILED'}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
