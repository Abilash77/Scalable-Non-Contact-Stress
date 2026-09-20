# Current Project Status

**Last verified:** 2026-09-20 (local runtime + `python run_tests.py`)

## Target vs Current

| Area | Target (Reference Spec) | Current Status |
|------|-------------------------|----------------|
| Five-modality fusion | RA-HMSD with reliability-aware attention | **Implemented** (214,166 params) |
| Real-time inference | ~10 Hz live dashboard | **Working** (~8.4 Hz measured) |
| Stress prediction | Trained STRESS / NON-STRESS | **Blocked** — no legitimate checkpoint |
| Training | ForDigitStress end-to-end | **Blocked** — dataset not present locally |
| Camera / face / eye | Physical webcam + detection | **Working** (MediaPipe Face Mesh) |

## What Works

### Runtime (`run.py`)
- Multiprocessing workers: keyboard, speech, webcam (face+eye), handwriting, inference
- Flask dashboard at `http://localhost:5000` with MJPEG camera stream
- `/status` JSON contract (see `docs/REAL_TIME_OUTPUT.md`)
- Dynamic modality masking with `-1e9` attention penalty for unavailable inputs
- **Untrained mode:** `prediction = NOT AVAILABLE`, probabilities are `null` (no fake stress claims)
- Rolling latency percentiles (P50/P95/MAX) and scheduler frequency in `latency_percentiles`

### Model (`src/DL_models.py`)
- Modality encoders: Dense(128) → Dropout → GRU(64)
- Reliability: `r_m = sigmoid(W_r · H_m)`
- Attention: `α_m = softmax(W_a · (H_m ⊙ r_m) + mask_penalty)`
- Fusion: `F = Σ α_m H_m` → Softmax classifier

### Feature Extractors
| Modality | Dim | Module | Status |
|----------|-----|--------|--------|
| Speech | 169 | `src/audio_utils.py` | Working |
| Facial | 12 | `src/face_utils.py` (MediaPipe) | Working |
| Keyboard | 7 | `src/keystroke_utils.py` | Working |
| Handwriting | 9 | `src/handwriting_utils.py` | Working |
| Eye/Pupil | 5 | `src/eye_utils.py` (MediaPipe iris) | Working |

### Tests
- `python run_tests.py`: **17/17 PASS** (executed 2026-09-20)

## What Is Blocked

1. **ForDigitStress dataset** — `data/fordigitstress/` not present locally
2. **Legitimate stress checkpoint** — `results/checkpoints/stress_model.keras` not present
3. **End-to-end stress validation** — requires trained weights on real stress labels

Training exits with actionable messages; no fake checkpoints or metrics are generated.

## Dataset Status

| Path | Status | Notes |
|------|--------|-------|
| `data/fordigitstress/` | **Missing** | Primary target; request at hcai.eu/fordigitstress |
| `data/swell_kw/` | **Missing** | Optional SWELL-KW adapter path |
| Local keystroke/handwriting/fer | May exist | **Not valid stress ground truth** — do not repurpose |

Validate structure when acquired:
```bash
python train.py --validate-data
```

## Model / Checkpoint Status

- **Runtime model:** Random initialization (UNTRAINED)
- **Checkpoint policy:** `run.py` loads weights only when `training_metadata.json` has `is_trained: true` and matching `.keras` file exists
- **Verify utility:** `python verify_checkpoint.py` — reports NOT FOUND honestly when no checkpoint

## Real-Time Dashboard

- **UI:** `templates/index.html` — polls `/status` every 100 ms
- **Untrained display:** Shows `ARCHITECTURE TEST ONLY`, hides stress probabilities
- **Sections:** System status, current state, modality monitor, camera, reliability/attention, performance, temporal history, handwriting canvas

## Camera

- Opens physical webcam via `cv2.VideoCapture(0..3)` fallback scan
- MediaPipe Face Mesh for 12D facial + 5D eye features
- Bounding box overlay on MJPEG stream
- Camera released cleanly on worker shutdown

## Measured Performance (Local, 2026-09-20)

From live `python run.py` session (UNTRAINED, monitoring active):

| Metric | P50 | P95 | Max |
|--------|-----|-----|-----|
| Feature extraction | 50.9 ms | 53.7 ms | 53.7 ms |
| Model inference | 6.9 ms | 8.0 ms | 8.2 ms |
| Total pipeline | 58.6 ms | 61.3 ms | 61.9 ms |

| Other | Value |
|-------|-------|
| Scheduler frequency | **8.4 Hz** (measured; target 10 Hz) |
| Camera FPS | ~29.5 |
| Model parameters | 214,166 |
| HTTP `/status` latency | ~0.6 ms mean |

From `run_tests.py` benchmark (synchronous, no I/O workers):

| Metric | Mean |
|--------|------|
| Direct E2E (extract + infer) | 66.6 ms |
| Fusion inference (steady) | 16.8 ms |

## Internal Tensor Contract

Model input order (`src/DL_models.py`):

| Index | Input | Shape |
|-------|-------|-------|
| 0 | `input_audio` (speech) | (B, 10, 169) |
| 1 | `input_face` (facial) | (B, 10, 12) |
| 2 | `input_keystroke` (keyboard) | (B, 10, 7) |
| 3 | `input_handwriting` | (B, 10, 9) |
| 4 | `input_eye` (eye/pupil) | (B, 10, 5) |
| — | `input_mask` | (B, 5) |

## Remaining Work (Non-Code)

1. Acquire ForDigitStress and place under `data/fordigitstress/`
2. Complete preprocessing mappings in `src/build_fordigitstress_dataset.py`
3. Train: `python train.py`
4. Verify: `python verify_checkpoint.py`
5. Run live: `python run.py` → dashboard transitions to TRAINED mode automatically

**TRAINING BLOCKED ONLY BY DATASET AVAILABILITY.** All other pipeline components are production-ready.
