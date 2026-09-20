# KAGGLE DATASET VERIFICATION

## 1. EXACT DATASET
- **Dataset Title:** `Stress Detection by Keystroke,App & Mouse Changes`
- **Kaggle Owner:** Chamindu Weerasinghe (`chaminduweerasinghe`)
- **Official URL:** `https://www.kaggle.com/datasets/chaminduweerasinghe/stress-detection-by-keystrokeapp-mouse-changes`
- **Publication/Update:** Updated approximately 5 years ago (2021).
- **License:** CC BY-SA 4.0 (Creative Commons Attribution-ShareAlike 4.0 International)
- **Downloadable Files:** 12 TSV files split across two user folders (`Data/user 1/` and `Data/user 2/`).
- **File Sizes:** Total uncompressed size is 270.94 MB (compressed zip is ~42 MB).

## 2. INSPECT THE ACTUAL DATA
*Access Restrictions:* Direct download is not possible without Kaggle authentication. The browser agent verified that attempting to download redirects to `/account/login`. Consequently, the dataset metadata and structure were inspected via the Kaggle Data Explorer.
- **Participants/Users:** Exactly 2 users (`user 1`, `user 2`).
- **Directory Structure:**
  - `activewindows.tsv`
  - `inactivity.tsv`
  - `keystrokes.tsv`
  - `mouse_mov_speeds.tsv`
  - `mousedata.tsv`
  - `usercondition.tsv` (Labels)
- **Missing/Duplicate Records:** Cannot be exhaustively verified without full unauthenticated download, but the Data Explorer confirms minimal sample sizes overall.

## 3. VERIFY STRESS LABELS
> [!TIP]
> **Stress labels are self-reported and categorically distinct.**

The ground truth labels are found in `usercondition.tsv`.
- **Label Source:** Self-reported by the users at intervals of 5 to 30 minutes.
- **Is it an experimental condition?** No, it appears to be passive monitoring paired with ecological momentary assessment (self-report).
- **Values:** The `Stress_Val` column contains explicit categories: `Neutral`, `S_Stressed` (Slightly Stressed), `V_Stressed` (Very Stressed), and `F_Good` (Feeling Good).

## 4. VERIFY KEYBOARD DATA
> [!TIP]
> **Keyboard data exists and contains valid temporal features.**

The keystroke data is located in `keystrokes.tsv`.
- **Keyboard Features:** The available columns are `Key`, `Press_Time`, `Relase_Time`, and `Daylight`.
- **Temporal Windows:** Because exact press and release times are recorded in standard chronological formats, flight times and dwell times (`Relase_Time - Press_Time`) can be successfully derived and partitioned into sequential windows.

## 5. VERIFY PARTICIPANTS
> [!WARNING]
> **Extreme data sparsity: Only 2 unique participants.**

- **Unique Participants:** 2
- **Records per Participant:** Approximately 33 self-report records (labels) per participant.
- **Subject-Independent Splitting:** **IMPOSSIBLE**. You cannot perform meaningful subject-independent evaluation (train on $N$ subjects, test on $M$ unseen subjects) when $N+M = 2$.

## 6. LEAKAGE CHECK
With only 2 users and multiple temporally sequential windows per user, any random test/train split would result in extreme data leakage. Training data would invariably contain windows from the exact same participant appearing in the test set.

## 7. DATA SUFFICIENCY
- **Total Labeled Windows:** ~66 total self-report samples across both users.
- **Stress / Non-Stress Windows:** Unknown exact split, but constrained within the 66 total samples.
- **Project Criteria Check:**
  - $\ge 15$ Participants: **FAIL** (Only 2 participants)
  - $\ge 500$ Labeled Windows: **FAIL** (Only ~66 total samples)
  - Subject-Independent Split: **FAIL**

## 8. LICENSE / ACADEMIC USE
- **License:** CC BY-SA 4.0.
- This permits academic research, modification, preprocessing, and redistribution of the derived model/data, provided appropriate attribution is given and derived works are shared under the same license.

## 9. COMPATIBILITY WITH CURRENT MODEL
The current 7D Keystroke Feature Vector expects:
`[mean_dwell, std_dwell, mean_flight, std_flight, typing_speed, backspace_freq, error_rate]`

The Kaggle dataset only provides raw `Press_Time` and `Relase_Time` for specific `Key` values.
- **Dwell Time:** Can be mapped (`Relase_Time - Press_Time`).
- **Flight Time:** Can be mapped (`Press_Time[i] - Relase_Time[i-1]`).
- **Typing Speed / Error Rate:** Can be approximated using the `Key` column (e.g., counting BACKSPACE events).
Therefore, **preprocessing pipeline modification would be required**, but it is technically legitimate and compatible without inventing fake data.

## 10. FINAL DECISION
**REJECT — INSUFFICIENT PARTICIPANT DATA**

*Reasoning:* Despite having valid keystroke timestamps and self-reported stress labels, the dataset contains only 2 individuals and approximately 66 labeled windows. It completely fails the established project acceptance criteria ($\ge 15$ participants, $\ge 500$ windows), making it impossible to validate a subject-independent stress model. No further integration of this dataset should occur.
