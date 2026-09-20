# Training Guide

## Overview

Training is **separate** from runtime inference.

| Command | Purpose |
|---------|---------|
| `python train.py` | ForDigitStress five-modality fusion training |
| `python train_swell.py` | SWELL-KW adapter training (optional) |
| `python run.py` | **Inference only** — never trains |

## Prerequisites

1. Python 3.10+
2. `pip install -r requirements.txt`
3. Legitimate stress dataset placed under `data/fordigitstress/` or `data/swell_kw/`

## Step 1: Validate Dataset Structure

```bash
python train.py --validate-data
```

Expected when dataset is missing:

```
[FAIL] Directory not found: .../data/fordigitstress/
TRAINING READINESS: BLOCKED
```

Expected when dataset exists but preprocessing is incomplete:

```
DATASET PREPARATION INCOMPLETE
[BLOCKED] UNRESOLVED MAPPING: ...
```

## Step 2: Train (When Pipeline Is Complete)

```bash
python train.py --epochs 100 --batch_size 32
```

Training requirements (scaffold in place; blocked until dataset tensors exist):

- Subject-level split (no participant leakage)
- Train / validation / test separation
- Checkpoint saved to `results/checkpoints/stress_model.keras`
- Metadata saved via `save_training_metadata()` **only after** `model.fit()` completes

If preprocessing completes but tensors are not ready, training raises `NotImplementedError` rather than writing a fake checkpoint.

### Required Metadata Format

```json
{
  "dataset_name": "ForDigitStress",
  "model_architecture": "fusion",
  "is_trained": true,
  "expected_input_shapes": {
    "audio": [10, 169],
    "face": [10, 12],
    "keystroke": [10, 7],
    "handwriting": [10, 9],
    "eye": [10, 5]
  }
}
```

`run.py` loads checkpoints only when `is_trained: true` and the matching `.keras` file exists.

## Step 3: Verify Checkpoint

```bash
python verify_checkpoint.py
```

## Step 4: Runtime Inference

```bash
python run.py
```

Open `http://localhost:5000`, click **START MONITORING**. With a valid checkpoint, `model_status` becomes `TRAINED` and predictions become available.

## SWELL-KW Alternative

```bash
python train_swell.py
```

Produces `results/checkpoints/swell_kw_best.keras` with `model_architecture: "swell_fusion"`.

## What Will NOT Happen

- Random-init weights are never saved as a stress detector
- Emotion/personality datasets are not silently converted to stress
- Training does not run inside `run.py`
