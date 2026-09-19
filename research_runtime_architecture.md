# Dual Architecture Design: Research vs. Live Runtime

This project enforces a strict architectural separation between the **offline scientific research experiments** and the **live engineering demonstration**. 

## 1. The Research Architecture (`train_pd.py`)

The scientific goal of the research pipeline is to reproduce the ForDigitStress deep-learning experiments detailed in the project's foundational paper. 

**Scientific Constraints:**
- The paper's ForDigitStress neural network experiments (`exp201` - `exp205`) were strictly **unimodal**, relying solely on Pupil Diameter (PD) as the comparable eye-tracking signal.
- The true ForDigitStress dataset provides labels, pupil features, Action Units, and GeMAPS features, but lacks any Keyboard or Handwriting dynamics.

To prevent fabricating cross-modal alignment or inventing features to satisfy geometric extractors, the research architecture is mathematically constrained to **PD-only inputs** (`T=10, 5`). 

**What Claims It Supports:**
- We can legitimately claim this architecture reproduces the paper's unimodal PD stress classification accuracy using true ForDigitStress annotations and temporal windows.

**What Claims It DOES NOT Support:**
- We **cannot** present the PD-only model as a 5-modality fusion model. 
- It **does not** validate multimodal attention weights on ForDigitStress.

## 2. The Live Runtime Architecture (`run.py`)

The live runtime is an engineering framework built *around* the core research, acting as a scalable demonstrator for real-time multimodal inference. 

**Engineering Capabilities:**
- Implements a Reliability-Aware Attention Fusion model spanning **5 independent modalities**: Keyboard, Speech, Facial, Eye/Pupil, and Handwriting.
- Actively consumes inputs from webcams, microphones, keyboard listeners, and HTML canvases. 

**Why Unrelated Datasets Are Not Concatenated:**
- Combining disjoint datasets (e.g., Keystrokes for Modality 1, RAVDESS for Modality 2, ForDigitStress for Modality 4) would synthetically fake temporal correlation. Since the fusion model's attention mechanism calculates real-time interplay between simultaneous physical reactions, training it on randomly stitched disjoint modalities destroys the scientific validity of the attention mechanics.

## 3. Interaction Between Architectures

The Live Runtime is designed to load checkpoint weights flexibly using an `input_mask`. 
When the `research_pd_model.keras` checkpoint is eventually trained, the runtime can (if desired) load those weights *strictly* into the Eye/Pupil branch, allowing the dashboard to function gracefully while the other 4 branches are masked or degraded. 

However, a Research checkpoint will **never** cause the Live Runtime to display `TRAINED` for the full 5-modality fusion, ensuring the system remains honest about its calibrated capabilities.
