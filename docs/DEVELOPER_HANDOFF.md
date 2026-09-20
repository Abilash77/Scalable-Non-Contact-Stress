# Developer Handoff

## Project Reality

This repository is a **fully wired architecture demonstrator** for RA-HMSD multimodal stress detection. The real-time pipeline, API, dashboard, feature extractors, fusion model, and masking logic are implemented and tested. **Stress predictions are not scientifically valid until a legitimate checkpoint is trained on real stress labels.**

## Quick Start

```bash
pip install -r requirements.txt
pip install -e .

# Run automated validation (17 tests)
python run_tests.py

# Start live server + dashboard
python run.py
# Open http://localhost:5000
# Click START MONITORING or POST /api/control {"action":"start"}
```

## What Works

| Component | Location | Notes |
|-----------|----------|-------|
| Real-time server | `run.py` | Inference never trains; loads checkpoint if valid |
| Fusion model | `src/DL_models.py` | 214K params, 5 modalities, reliability attention |
| Speech features | `src/audio_utils.py` | 169D via librosa |
| Facial features | `src/face_utils.py` | 12D via MediaPipe Face Mesh |
| Eye features | `src/eye_utils.py` | 5D via MediaPipe iris landmarks |
| Keyboard features | `src/keystroke_utils.py` | 7D via pynput events |
| Handwriting | `src/handwriting_utils.py` + `/upload_handwriting` | 9D from canvas upload |
| Dashboard | `templates/index.html` | Full `/status` contract consumer |
| Dataset validator | `src/build_fordigitstress_dataset.py` | Structural checks only |
| Checkpoint verify | `verify_checkpoint.py` | Requires metadata + weights |

## What Does NOT Work Yet

- **Stress prediction:** No trained checkpoint → `model_status: UNTRAINED`, `prediction: NOT AVAILABLE`
- **Training:** `python train.py` exits until ForDigitStress is present and preprocessing is complete

## Untrained Mode Contract

When no valid checkpoint exists, `/status` returns:
- `status`: `ARCHITECTURE_TEST_ONLY` (when monitoring active)
- `predictions_available`: `false`
- `stress_probability`: `null`
- `architecture_debug`: raw forward-pass values labeled as **NOT valid stress detection**

The dashboard **never** displays "STRESS DETECTED" from random weights.

## Mask / Modality Mapping

Internal model order → API names:

| Mask index | Model input | API name |
|------------|-------------|----------|
| 0 | audio | speech |
| 1 | face | facial |
| 2 | keystroke | keyboard |
| 3 | handwriting | handwriting |
| 4 | eye | eye_pupil |

## Training Path (When Dataset Available)

```bash
# 1. Place ForDigitStress under data/fordigitstress/VPxx/
python train.py --validate-data

# 2. Complete preprocessing (build_fordigitstress_dataset.py)
python train.py --epochs 100 --batch_size 32

# 3. Verify
python verify_checkpoint.py

# 4. Runtime picks up checkpoint automatically
python run.py
```

Metadata must have `"is_trained": true`. The helper `save_training_metadata()` in `train.py` is called only after a real `model.fit()` — never from scaffolding.

## Security

- Do not commit datasets, checkpoints, or `.env` files
- `.gitignore` blocks `data/`, weights, and archives

## Test Results (2026-09-20)

```
python run_tests.py → 17/17 PASS
```

## Performance Reference

Live runtime (measured):
- Scheduler: ~8.4 Hz
- Pipeline P50: ~58.6 ms
- Camera: ~30 FPS

See `docs/CURRENT_PROJECT_STATUS.md` for full measurements.

## Next Developer Actions

1. Obtain ForDigitStress from [hcai.eu/fordigitstress](https://hcai.eu/fordigitstress/)
2. Resolve remaining modality file mappings in `FordigitstressModalityAdapter`
3. Implement temporal alignment (T=10) with real timestamps
4. Train and verify checkpoint
5. Re-run `python run_tests.py` and live validation
