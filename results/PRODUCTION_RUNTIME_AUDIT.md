# Production Runtime Audit

## 1. Old DriveDB Selection Path
Previously, `run.py` was hardcoded to intercept models with `"model_architecture": "drivedb_fusion"` and load `drivedb_best.keras`. It would also manually inject physiological dummy inputs `(10, 4)` and a single mask dimension.

## 2. Files Responsible
- `run.py`: Handled the model fallback and dummy data generation.
- `train.py`: Handled the DriveDB fallback training on physiological data.
- `src/DL_models.py`: Built the 1-modality physiological adapter architecture.

## 3. Changes Made
- **run.py**: Removed all `drivedb_fusion` fallback loading mechanisms. Enforced strict Freihaut checkpoint criteria. Bypassed live stress inference completely when no verified checkpoint exists, enforcing a "READY FOR TRAINING" status.
- **train.py**: Entirely rewritten to target the Freihaut & Göritz dataset. It expects `freihaut_X_train.npy` (which does not exist yet) and fails fast, preventing accidental training on legacy subsets.
- **Checkpoints**: `results/checkpoints/training_metadata.json` for the SWELL prototype was explicitly updated to mark it as a `PROTOTYPE`. This ensures `run.py` correctly rejects it for production selection.

## 4. Current Production Checkpoint Policy
**Priority:**
1. Verified Freihaut & Göritz keyboard-stress checkpoint.
2. Otherwise: **NO PRODUCTION MODEL**.

**Never Allow:**
- DriveDB fallback.
- Random models.
- Prototype SWELL subsets (rejected via `"is_prototype": true`).

## 5. Current Dataset Status
**Dataset:** Freihaut & Göritz Keyboard Stress
**Status:** Preprocessing required.

## 6. Prototype Checkpoint Status
**File:** `results/checkpoints/keystroke_stress.keras`
**Status:** Maintained for historical auditing, but explicitly disabled from live inference via `"is_prototype": true` metadata tag.

## 7. run.py Behavior
When executed, `run.py` will log:
```
MODEL STATUS: READY FOR TRAINING
DATA STATUS: FREIHAUT & GÖRITZ KEYBOARD STRESS
MODE: ARCHITECTURE TEST ONLY
MESSAGE: "No valid trained stress model found. Run `python train.py` after preprocessing Freihaut data."
```
The dashboard will reflect these states natively.

## 8. train.py Behavior
When executed, `train.py` will attempt to load `data/processed/freihaut_X_train.npy` and fail explicitly if preprocessing has not been completed, preventing arbitrary dataset model training.
