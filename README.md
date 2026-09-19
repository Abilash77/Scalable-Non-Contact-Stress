# Multimodal Stress Classification

"A multimodal architecture supporting five non-contact modalities was developed, with modality-specific training data sourced from datasets appropriate to each modality. ForDigitStress provides synchronized speech, facial, and eye/pupil data, while keyboard and handwriting are obtained from separate datasets. Cross-dataset training is therefore treated as a hybrid multimodal framework rather than a direct five-modality reproduction of a single synchronized dataset."

This repository contains the code for a comprehensive Multimodal Stress Classification framework. The project leverages deep learning approaches to classify mental stress using diverse, multi-source datasets covering physiological signals, facial expressions, speech, keyboard dynamics, and handwriting behavior.

## Datasets

Our framework fuses information from the following five core behavioral/non-contact modalities:

*   **(A) Keyboard dynamics**  
    [Keystroke Dynamics - Benchmark Data Set](https://www.kaggle.com/datasets/carnegiecylab/keystroke-dynamics-benchmark-data-set)
    *Note: The keyboard encoder is pretrained via biometric subject identification rather than a stress-labeled task; its stress-relevance is a plausible but unvalidated transfer hypothesis.*
*   **(B) Speech / audio emotion**  
    [RAVDESS Dataset](https://zenodo.org/record/1188976)
*   **(C) Facial expression data**  
    [FER2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013)
*   **(D) Eye / Pupil**  
    Eye tracking and pupillometry features (using MediaPipe)
*   **(E) Handwriting behavior**  
    [Handwriting & Personality Traits Dataset](https://www.kaggle.com/datasets/khushikyad001/handwriting-and-personality-traits-dataset)
    *Note: The available handwriting dataset was evaluated through a Neuroticism/personality proxy task rather than a stress-labeled task. The resulting encoder did not demonstrate useful predictive performance and is therefore not transferred into the stress fusion model. The Handwriting branch remains architecturally present but untrained pending a genuinely stress-relevant handwriting dataset.*

### Supplemental Datasets
*   **Physiological Reference/Base Dataset**  
    [WESAD (Wearable Stress and Affect Detection Dataset)](https://uni-siegen.sciebo.de/public.php/dav/files/HGdUkoNlW1Ub0Gx/?accept=zip)
*   **Primary Research/Reproduction Dataset**  
    ForDigitStress Dataset

## Project Structure and Usage

### 1. Preprocessing

Before training, each dataset must be preprocessed. Ensure you have downloaded the raw datasets to their respective `data/<dataset_name>/raw/` folders. Run the following preprocessing scripts:

### OpenCV ximgproc Dependency
This project uses OpenCV. For handwriting feature extraction, `cv2.ximgproc` is the preferred implementation when available (via `opencv-contrib-python`). If it is missing from your environment, the code uses a fallback morphological substitute. 
**Important Note:** The fallback is a degraded substitute and is NOT equivalent to the original `ximgproc` skeletonization implementation. Any latency measurement gathered while using the fallback must be strictly identified as fallback-path latency.

```bash
python scripts/preprocessing/preprocess_wesad.py --raw_data_path data/wesad/raw/ --output_path data/wesad/dataframes/
python scripts/preprocessing/preprocess_fer.py --raw_data_path data/fer2013/raw/ --output_path data/fer2013/dataframes/
python scripts/preprocessing/preprocess_ravdess.py --raw_data_path data/ravdess/raw/ --output_path data/ravdess/dataframes/
python scripts/preprocessing/preprocess_keystrokes.py --raw_data_path data/keystroke/raw/ --output_path data/keystroke/dataframes/
python scripts/preprocessing/preprocess_handwriting.py --raw_data_path data/handwriting/raw/ --output_path data/handwriting/dataframes/
```

### 2. Training the Fusion Network

Once all datasets are preprocessed into DataFrames, you can train the multimodal fusion network:

```bash
python scripts/training/train_fusion_network.py \
  --wesad_path data/wesad/dataframes/wesad_processed.pkl \
  --ravdess_path data/ravdess/dataframes/ravdess_processed.pkl \
  --fer_path data/fer2013/dataframes/fer2013_processed.pkl \
  --keystroke_path data/keystroke/dataframes/keystroke_processed.pkl \
  --hw_path data/handwriting/dataframes/handwriting_processed.pkl \
  --model_out results/fusion/
```

### 3. Real-Time Server

The framework also includes a real-time server module for multimodal stress detection:

```bash
python -m scripts.realtime.multimodal_stress_server
```

You can also test the standalone face-stress detector:

```bash
python scripts/realtime/face_stress_detector.py
```

### End-to-End Testing

To validate the entire pipeline, run the end-to-end test script:

```bash
python run_tests.py
```
