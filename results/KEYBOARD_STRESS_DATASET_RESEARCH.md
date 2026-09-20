# KEYBOARD STRESS DATASET RESEARCH

## SEARCH SCOPE
Repositories searched: UCI, PhysioNet, Zenodo, Figshare, Kaggle, Harvard Dataverse, GitHub.
Search terms utilized: "keystroke dynamics psychological stress", "typing behavior stress detection dataset open access", "keyboard interaction stress", etc.

## CANDIDATE EVALUATION

| Dataset | Official source | Paper | Participants | Keyboard data | Stress labels | Label source | Temporal data | Sessions | Labeled samples | Participant IDs | License | Access method | Subject split possible? | 7D feature comp | Reason accepted/rejected |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **SWELL-KW** | DANS / TNO | Koldijk et al. (2014) | 25 | Yes (Raw logs) | Yes | Experimental condition + Self-report | Yes | 3 per user | >500 | Yes | Restricted (DANS) | Manual request/approval | Yes | Yes (needs extraction) | **ACCEPT — SUFFICIENT** (But requires manual institutional access) |
| **Keyboard Stress Analysis** | Zenodo (10.5281/zenodo.4445197) | Freihaut & Göritz (2021) | 977 (53 Lab + 924 Online) | Yes (11 timing features) | Yes | Experimental condition (High/Low stress tasks) | Yes | 2 per user | >1,000 | Yes | Open Access | Direct ZIP download | Yes | Yes (maps to dwell/flight/error) | **ACCEPT — SUFFICIENT** (Best Open Dataset) |
| **Kaggle Stress Detection** | Kaggle | N/A | 2 | Yes (Timestamps) | Yes | Self-report (PAM) | Yes | Variable | ~66 | Yes | CC BY-SA 4.0 | Kaggle API | No | Yes (needs extraction) | **REJECT — TOO FEW PARTICIPANTS** |
| **WESAD** | UCI Machine Learning | Schmidt et al. (2018) | 15 | No (Physiological only) | Yes | Experimental | Yes | Variable | High | Yes | Public | Direct | Yes | N/A | **REJECT — NO KEYBOARD DATA** |
| **Tappy Keystroke Data** | PhysioNet | Giancardo et al. | 200+ | Yes | No (Parkinson's Disease) | Clinical Diagnosis | Yes | Variable | High | Yes | Public | Direct | Yes | N/A | **REJECT — NO REAL STRESS LABEL** |
| **CMU Keystroke Benchmark** | CMU Repository | Killourhy & Maxion (2009) | 51 | Yes | No (Authentication/Identity) | Identity ID | Yes | 8 per user | High | Yes | Public | Direct | Yes | N/A | **REJECT — NO REAL STRESS LABEL** |
| **BiAffect** | biaffect.com | Leow et al. | >1,000 | Yes (Mobile) | No (Mood / Bipolar disorder) | Clinical Diagnosis | Yes | Variable | High | Yes | Restricted | Application | Yes | N/A | **REJECT — NO REAL STRESS LABEL** |

## BEST VERIFIED CANDIDATE

**1. Best verified candidate:** 
"Study Material for the paper: Does People's Keyboard Typing Reflect their Stress Level – An Explorative Study" (Freihaut & Göritz, 2021). Hosted on Zenodo.

**2. Exact acquisition procedure:**
Directly downloadable via Zenodo's open API:
`wget https://zenodo.org/api/records/4445197/files/Freihaut/Study_Files_KeyboardData_Analysis-1.0.0.zip/content`

**3. Expected participant count:**
977 total distinct participants (53 from a controlled laboratory study, 924 from a remote online study).

**4. Expected labeled sample/window count:**
>1,000. Each participant completed typing tasks under both high-stress and low-stress conditions, yielding thousands of labeled typing trials.

**5. Label definition:**
Experimental Condition. Participants typed standardized text sequences during an explicitly high-stress condition and a low-stress condition. Manipulation checks confirmed consistent differences in participant stress levels between conditions.

**6. Whether subject-independent evaluation is possible:**
**YES**. With 977 unique participant identifiers and distinct tasks per participant, a robust Leave-N-Subjects-Out cross-validation split is natively supported.
