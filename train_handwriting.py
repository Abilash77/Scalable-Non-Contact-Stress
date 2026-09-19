import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Import ModalityBranch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from DL_models import ModalityBranch

def load_and_prepare_handwriting_data(csv_path, T=10):
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} records. Preparing sequences of length {T}...")
    
    # We will predict Neuroticism (binary: High vs Low) as the proxy task
    median_neur = df['Neuroticism'].median()
    df['label'] = (df['Neuroticism'] > median_neur).astype(int)
    
    # Extract the first 9 features to match the runtime input shape of (10, 9)
    feature_cols = [f'Feature_{i}' for i in range(1, 10)]
    features = df[feature_cols].values.astype(np.float32)
    labels = df['label'].values
    
    X = []
    y = []
    
    # Create sequences of length T with step=1
    # Since Handwriting_Sample is essentially unique per row, we simulate temporal 
    # sequences by repeating the feature vector for T timesteps, representing 
    # a static handwriting stroke analyzed temporally.
    for i in range(len(features)):
        # Replicate the static 9D feature to form a (10, 9) sequence
        seq = np.tile(features[i], (T, 1))
        X.append(seq)
        y.append(labels[i])
            
    X = np.array(X, dtype=np.float32) # Shape: (N, 10, 9)
    y = np.array(y, dtype=np.int32)
    
    print(f"Generated {len(X)} sequences of shape {X.shape}.")
    return X, y, 2

def build_unimodal_handwriting_model(input_shape, num_classes):
    inputs = tf.keras.layers.Input(shape=input_shape, name="input_handwriting")
    
    # Exact architecture from fusion model
    branch = ModalityBranch(encoder_units=32, gru_units=64, name="handwriting_branch")
    
    hm = branch(inputs)
    
    # Add a temporary classification head for pre-training
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax', name="trait_classifier")(hm)
    
    model = tf.keras.models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model, branch

def train_handwriting_model():
    csv_path = "data/handwriting/raw/handwriting_personality_large_dataset.csv"
    if not os.path.exists(csv_path):
        print(f"[ERROR] Handwriting data not found at {csv_path}")
        sys.exit(1)
        
    X, y, num_classes = load_and_prepare_handwriting_data(csv_path, T=10)
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model, branch_layer = build_unimodal_handwriting_model(input_shape=(10, 9), num_classes=num_classes)
    model.summary()
    
    checkpoint_dir = "results/checkpoints/"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print("Pre-training Handwriting ModalityBranch on Personality Trait Proxy (Neuroticism)...")
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=128,
        callbacks=[early_stop]
    )
    
    # We save ONLY the weights of the ModalityBranch so it can be cleanly loaded into fusion later
    branch_weights_path = os.path.join(checkpoint_dir, "handwriting_encoder.weights.h5")
    
    # Create a tiny model just for the branch to save it natively
    branch_input = tf.keras.layers.Input(shape=(10, 9))
    branch_output = branch_layer(branch_input)
    save_model = tf.keras.models.Model(branch_input, branch_output)
    save_model.save_weights(branch_weights_path)
    
    print(f"\n[SUCCESS] Unimodal Handwriting encoder weights saved to {branch_weights_path}")

if __name__ == "__main__":
    train_handwriting_model()
