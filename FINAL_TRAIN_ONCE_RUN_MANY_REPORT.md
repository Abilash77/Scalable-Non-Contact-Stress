# TRAIN-ONCE / RUN-MANY — Verification Report

> Generated: 2026-09-19T02:37 IST  
> Project: Scalable Non-Contact Stress Detection Using Hybrid Multimodal Intelligence

---

## Architecture Summary

```
TRAIN ONCE                      RUN MANY
─────────────                   ──────────────
python train.py                 python run.py
      │                               │
      ▼                               ▼
ForDigitStress dataset          Load checkpoint
      │                               │
      ▼                               ▼
5-modality training             Initialize sensors
      │                               │
      ▼                               ▼
Save checkpoint                 Live inference loop
results/checkpoints/                  │
  stress_model.keras                  ▼
                                Dashboard + API
```

---

## Commands

| Purpose | Command | Status |
|---------|---------|--------|
| **Training** (offline, explicit) | `python train.py` | Blocked — dataset required |
| **Runtime** (online, inference-only) | `python run.py` | Operational |
| **Dataset validation** | `python train.py --validate-data` | Blocked — dataset required |
| **Regression tests** | `python run_tests.py` | 12/12 PASS |
| **Checkpoint verification** | `python verify_checkpoint.py` | NOT FOUND (correct) |

---

## Checkpoint Paths

| Checkpoint | Path | Purpose | Current State |
|------------|------|---------|---------------|
| **Runtime** | `results/checkpoints/stress_model.keras` | Live 5-modality inference | **NOT FOUND** |
| **Research** | `results/checkpoints/research_pd_model.keras` | PD unimodal research | **NOT FOUND** |

`run.py` only accepts `stress_model.keras`. It will never load `research_pd_model.keras`.

---

## Proof: run.py Has ZERO Training Paths

Static analysis — searched run.py for all training-related patterns:

| Search Term | Occurrences in run.py |
|-------------|----------------------|
| `model.fit` | **0** |
| `fit` | **0** |
| `fit_generator` | **0** |
| `train_step` | **0** |
| `optimizer` | **0** |
| `compile` | **0** |
| `save_weights` | **0** |
| `save` | **0** |
| `train.py` | **0** |
| `fordigitstress` | **0** |
| `research_pd` | **0** |

**Conclusion: run.py contains absolutely no code path that can train, save, compile, or process training data.**

---

## Proof: run.py Does Not Create Checkpoints During Runtime

### Pre-runtime state
```
results/checkpoints/stress_model.keras       → NOT FOUND
results/checkpoints/research_pd_model.keras  → NOT FOUND
```

### After running python run.py for ~4 minutes
```
results/checkpoints/stress_model.keras       → NOT FOUND  PASS
results/checkpoints/research_pd_model.keras  → NOT FOUND  PASS
```

**Conclusion: Runtime execution creates zero files in the checkpoint directory.**

---

## Live /status API Response (while UNTRAINED)

Captured from http://localhost:5000/status during live runtime:

```json
{
  "model_status": "UNTRAINED",
  "checkpoint_name": "",
  "prediction": "NON-STRESS",
  "stress_probability": 0.1988,
  "non_stress_probability": 0.8012,
  "confidence": 0.8012,
  "unavailable_modalities": ["facial", "keyboard", "handwriting", "eye_pupil"],
  "prediction_timestamp": "02:33:41",
  "processing_latency_ms": 603.3
}
```

The `model_status` is `"UNTRAINED"`. The dashboard displays "Awaiting trained checkpoint" and "PREDICTION: NOT SCIENTIFICALLY VALID". The raw probability values come from the randomly-initialized (untrained) model architecture — they are NOT presented as valid predictions on the dashboard.

---

## Checkpoint Compatibility

The inference worker in run.py (line 174-204):

1. Checks if `results/checkpoints/stress_model.keras` exists
2. If found, attempts `model.load_weights(cp_file)` inside a `try/except`
3. If loading succeeds → sets `model_status = "TRAINED"`
4. If loading fails (wrong architecture, corrupt file) → falls back to `model_status = "UNTRAINED"`

The model architecture is hardcoded to the 5-modality fusion:
- audio: (10, 169)
- face: (10, 12)
- keystroke: (10, 7)
- handwriting: (10, 9)
- eye: (10, 5)

An incompatible checkpoint would be **rejected** by `load_weights()` and the system would remain UNTRAINED.

---

## train.py — Training Blocker Verification

Running `python train.py` produces:

```
============================================================
RA-HMSD SCIENTIFIC TRAINING PIPELINE
============================================================

[INFO] Checking hardware capabilities...
[INFO] No GPU detected. Training on CPU.

[INFO] Validating raw dataset directory: ...\data\fordigitstress

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
TRAINING BLOCKED — FORDIGITSTRESS DATASET REQUIRED
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
```

Exit code: **1** (clean failure, no crash, no fake data generated).

---

## Test Suite Results

```
==================================================
FINAL TEST RESULTS
==================================================
WESAD Exclusion from Non-Contact Runtime      : PASS
Architecture Validation                       : PASS
Test Extractor: Keyboard                      : PASS
Test Extractor: Speech                        : PASS
Test Extractor: Facial                        : PASS
Test Extractor: Eye/Pupil                     : PASS
Test Extractor: Handwriting                   : PASS
Missing Modality                              : PASS
Invalid Input                                 : PASS
Dashboard Startup                             : PASS
API Response                                  : PASS
Latency Benchmark                             : PASS

Total: 12
Passed: 12
Failed: 0
```

---

## What Was Actually Verified

| Assertion | Method | Result |
|-----------|--------|--------|
| run.py starts without training | Live execution | VERIFIED |
| run.py contains no training code | Static grep (11 patterns) | VERIFIED |
| run.py does not create checkpoints | Pre/post file system comparison | VERIFIED |
| run.py does not process the dataset | Static grep + runtime observation | VERIFIED |
| Dashboard shows UNTRAINED when no checkpoint | Live /status API capture | VERIFIED |
| Dashboard shows "Awaiting trained checkpoint" | Live browser observation | VERIFIED |
| No fake stress predictions displayed | Live dashboard observation | VERIFIED |
| train.py blocks cleanly without dataset | Live execution, exit code 1 | VERIFIED |
| Checkpoint isolation (runtime != research) | Code review of hardcoded path | VERIFIED |
| Incompatible checkpoint rejected | Code review of try/except guard | VERIFIED |
| 12/12 regression tests pass | python run_tests.py | VERIFIED |

---

## What Was NOT Verified (and why)

| Assertion | Reason |
|-----------|--------|
| Model produces scientifically valid predictions | No trained checkpoint exists |
| python train.py produces a valid checkpoint | Training dataset not available |
| Trained checkpoint loads correctly in run.py | No trained checkpoint exists |
| Dashboard transitions to TRAINED/ACTIVE state | No trained checkpoint exists |

The model is NOT trained. The project is NOT scientifically validated.

---

## Reboot Guarantee

Once stress_model.keras is legitimately trained and saved:

```
Day 1:  python train.py  →  saves stress_model.keras
Day 2:  Reboot computer
Day 2:  python run.py     →  loads stress_model.keras  →  live inference
Day 3:  Reboot computer
Day 3:  python run.py     →  loads stress_model.keras  →  live inference
...
```

No retraining. No dataset processing. Just load and run.
