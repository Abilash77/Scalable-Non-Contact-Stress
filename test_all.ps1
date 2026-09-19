$ErrorActionPreference = "Continue"

echo "=== WESAD Preprocessing ===" > validation2.log
python scripts/preprocessing/preprocess_wesad.py --raw_data_path data/wesad/raw/ --output_path data/wesad/dataframes/ 2>&1 >> validation2.log

echo "=== FER2013 Preprocessing ===" >> validation2.log
python scripts/preprocessing/preprocess_fer.py --raw_data_path data/fer2013/raw/ --output_path data/fer2013/dataframes/ 2>&1 >> validation2.log

echo "=== RAVDESS Preprocessing ===" >> validation2.log
python scripts/preprocessing/preprocess_ravdess.py --raw_data_path data/ravdess/raw/ --output_path data/ravdess/dataframes/ 2>&1 >> validation2.log

echo "=== Keystroke Preprocessing ===" >> validation2.log
python scripts/preprocessing/preprocess_keystrokes.py --raw_data_path data/keystroke/raw/ --output_path data/keystroke/dataframes/ 2>&1 >> validation2.log

echo "=== Handwriting Preprocessing ===" >> validation2.log
python scripts/preprocessing/preprocess_handwriting.py --raw_data_path data/handwriting/raw/ --output_path data/handwriting/dataframes/ 2>&1 >> validation2.log

echo "=== Training Script (Smoke Test) ===" >> validation2.log
# Limit to 1 epoch by sed or just run it and see if it compiles (it's 50 epochs by default, but it'll fail or run fast if datasets are small/empty)
python scripts/training/train_fusion_network.py --wesad_path data/wesad/dataframes/wesad_processed.pkl --ravdess_path data/ravdess/dataframes/ravdess_processed.pkl --fer_path data/fer2013/dataframes/fer2013_processed.pkl --keystroke_path data/keystroke/dataframes/keystroke_processed.pkl --hw_path data/handwriting/dataframes/handwriting_processed.pkl --model_out results/fusion/ 2>&1 >> validation2.log

echo "=== Server Smoke Test ===" >> validation2.log
# We can't keep the server running forever, so we just check if it parses and imports correctly without syntax errors.
python -c "import scripts.realtime.multimodal_stress_server" 2>&1 >> validation2.log
