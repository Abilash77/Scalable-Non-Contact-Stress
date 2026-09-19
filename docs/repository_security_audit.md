# Repository Security & Data Audit

## Scope
This audit ensures that the repository only tracks source code, documentation, and safe configuration files. All private datasets, trained checkpoints, and sensitive credentials have been verified to be excluded from version control.

## Private Dataset Audit
- **Path:** `data/fer2013/dataframes/fer2013_processed.pkl`, `data/handwriting/dataframes/handwriting_processed.pkl`, `data/keystroke/dataframes/keystroke_processed.pkl`, `data/ravdess/dataframes/ravdess_processed.pkl`, `data/vr_goalkeeper/dataframes/DL_out.pkl`, `data/vr_goalkeeper/dataframes/features_out.pkl`, `results/source_data/vr_model_comparison_metrics.csv`
- **Type:** `.pkl`, `.csv`
- **Purpose:** Processed ML features and labels.
- **Source:** Local generation / derived from raw datasets.
- **Status:** Private/Restricted
- **Push decision:** NO
- **Reason:** Derived dataset feature files containing potential participant behavior; explicitly ignored by `.gitignore`.

## Restricted Dataset Audit
- **Path:** (None found inside the project index; datasets like ForDigitStress or SWELL-KW are only referenced in documentation, not stored as files).
- **Type:** N/A
- **Purpose:** N/A
- **Source:** N/A
- **Status:** N/A
- **Push decision:** NO (None to push)
- **Reason:** N/A

## Downloaded Archive Audit
- **Path:** `neurokit2.zip`, `nk.zip`, `data/ravdess/raw/Audio_Speech_Actors_01-24.zip`
- **Type:** `.zip`
- **Purpose:** Raw dataset archives and downloaded library sources.
- **Source:** External
- **Status:** Unsafe for repository
- **Push decision:** NO
- **Reason:** Redundant large binary archives; ignored by `.gitignore`.

## Model Checkpoint Audit
- **Path:** `results/checkpoints/handwriting_encoder_REJECTED_neuroticism_proxy.weights.h5`, `results/checkpoints/keystroke_encoder.weights.h5`
- **Type:** `.h5`
- **Purpose:** Trained encoder weights for specific modalities.
- **Source:** Local training output.
- **Status:** Local binary artifact
- **Push decision:** NO
- **Reason:** Unnecessary for source repository and potentially derived from restricted data; ignored by `.gitignore`. (Note: these are rejected proxy checkpoints, not legitimate final stress checkpoints).

## Credential Audit
- **Path:** (None found)
- **Type:** N/A
- **Purpose:** N/A
- **Source:** N/A
- **Status:** N/A
- **Push decision:** NO
- **Reason:** No `.env` or local credential files found.

## API Key Audit
- **Found:** NO
- **Push decision:** NO
- **Reason:** No hardcoded API keys found in source code.

## Database Credential Audit
- **Found:** NO
- **Push decision:** NO
- **Reason:** No database connection strings found in source code.

## Git History Audit
- **Old history present:** NO (Only `Abilash Aruva <abilasharuva@gmail.com>` exists in the 3 commits of this new repository).
- **Secret history detected:** NO (No secrets present in the fresh history).

## Git Tracked Files Audit
- **Safe source files:** Yes (`src/*.py`, `scripts/**/*.py`)
- **Documentation:** Yes (`*.md`, `configs/**/*.json`)
- **Unsafe files excluded:** Yes (Checked `git ls-files`; no `.csv`, `.pkl`, `.zip`, `.h5` tracked).

## .gitignore Audit
- **Status:** Strong (covers `data/`, `results/checkpoints/`, `.env`, `*.zip`, `*.h5`, `*.pkl`, etc.).

## Final Repository Classification
The repository is fully clean. It safely contains only source code, documentation, and configuration templates. All external data, trained models, and secrets remain securely local.
