# Real-Time Output Contract

This document defines the JSON schema served by `GET /status` when running `python run.py`.

## Modes

| `model_status` | `status` (live monitoring) | Predictions |
|----------------|---------------------------|-------------|
| `LOADING` | `LOADING` | Not available |
| `UNTRAINED` | `ARCHITECTURE_TEST_ONLY` | **NOT AVAILABLE** — debug values in `architecture_debug` only |
| `TRAINED` | `LIVE` | Valid stress/non-stress probabilities |

## Canonical Schema

```json
{
  "status": "LIVE | ARCHITECTURE_TEST_ONLY | READY | LOADING",
  "model_status": "TRAINED | UNTRAINED | LOADING",
  "predictions_available": true,
  "prediction": "STRESS | NON-STRESS | NOT AVAILABLE",
  "stress_probability": 0.924,
  "non_stress_probability": 0.076,
  "confidence": 0.924,
  "modality_reliability": {
    "keyboard": 0.82,
    "speech": 0.91,
    "facial": 0.76,
    "eye_pupil": 0.88,
    "handwriting": 0.64
  },
  "modality_attention": {
    "keyboard": 0.15,
    "speech": 0.28,
    "facial": 0.22,
    "eye_pupil": 0.25,
    "handwriting": 0.10
  },
  "available_modalities": ["keyboard", "speech", "facial"],
  "unavailable_modalities": ["eye_pupil", "handwriting"],
  "window_size": 10,
  "timestamp": "2026-09-20T14:30:01",
  "latency": {
    "feature_extraction_ms": 12.4,
    "model_inference_ms": 7.5,
    "total_pipeline_ms": 19.9
  },
  "latency_percentiles": {
    "feature_extraction_ms": { "p50": 50.9, "p95": 53.7, "max": 53.7 },
    "model_inference_ms": { "p50": 6.9, "p95": 8.0, "max": 8.2 },
    "total_pipeline_ms": { "p50": 58.6, "p95": 61.3, "max": 61.9 },
    "scheduler_frequency_hz": 8.4
  }
}
```

**Important:** Example numeric values are structural only. All runtime values must come from the live system.

## Internal Mask Order vs API Names

The fusion model uses this internal mask index order (defined in `src/DL_models.py`):

| Index | Model input | API name |
|-------|-------------|----------|
| 0 | `input_audio` | `speech` |
| 1 | `input_face` | `facial` |
| 2 | `input_keystroke` | `keyboard` |
| 3 | `input_handwriting` | `handwriting` |
| 4 | `input_eye` | `eye_pupil` |

## Untrained Mode

When no valid checkpoint exists:

```json
{
  "status": "ARCHITECTURE_TEST_ONLY",
  "model_status": "UNTRAINED",
  "predictions_available": false,
  "prediction": "NOT AVAILABLE",
  "stress_probability": null,
  "non_stress_probability": null,
  "confidence": null,
  "architecture_debug": {
    "note": "Synthetic/untrained forward-pass values — NOT valid stress detection",
    "raw_stress_probability": 0.512,
    "smoothed_stress_probability": 0.498
  }
}
```

The dashboard must **not** display untrained outputs as genuine stress detection.

## Additional Endpoints

| Endpoint | Purpose |
|----------|---------|
| `GET /camera/status` | Camera FPS, face bbox, frame age |
| `POST /upload_handwriting` | Submit base64 handwriting canvas image |
| `POST /api/control` | `start`, `stop`, `reset` monitoring session |
| `GET /video_feed` | MJPEG webcam stream |

## Logging

Inference cycles are appended to `results/realtime_session.log` when monitoring is active.
