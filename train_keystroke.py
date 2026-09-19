import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Import ModalityBranch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from DL_models import ModalityBranch

def load_and_prepare_keystroke_data(csv_path, T=10):
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} records. Preparing sequences of length {T}...")
    
    h_cols = [c for c in df.columns if c.startswith('H.')]
    ud_cols = [c for c in df.columns if c.startswith('UD.')]
    
    # Pre-calculate 5D features for all rows
    h_vals = df[h_cols].values
    ud_vals = df[ud_cols].values
    
    dwell_mean = np.mean(h_vals, axis=1)
    dwell_std = np.std(h_vals, axis=1)
    flight_mean = np.mean(ud_vals, axis=1)
    flight_std = np.std(ud_vals, axis=1)
    
    total_time = np.sum(h_vals, axis=1) + np.sum(ud_vals, axis=1)
    num_chars = h_vals.shape[1]
    speed = np.divide(num_chars, total_time, out=np.zeros_like(total_time), where=total_time!=0)
    
    features_5d = np.column_stack((dwell_mean, dwell_std, flight_mean, flight_std, speed))
    
    df['features_5d'] = list(features_5d)
    
    X = []
    y = []
    
    for subject, group in df.groupby('subject'):
        feats = np.stack(group['features_5d'].values)
        
        # Pad from 5D to 7D
        features_7d = np.pad(feats, ((0,0), (0,2)), 'constant')
        
        for i in range(len(features_7d) - T + 1):
            X.append(features_7d[i:i+T])
            y.append(subject)
            
    X = np.array(X, dtype=np.float32)
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_classes = len(le.classes_)
    
    print(f"Generated {len(X)} sequences of shape {X.shape}. Unique subjects: {num_classes}")
    return X, y_encoded, num_classes

def build_unimodal_keystroke_model(input_shape, num_classes):
    inputs = tf.keras.layers.Input(shape=input_shape, name="input_keystroke")
    
    # Exact architecture from fusion model
    branch = ModalityBranch(encoder_units=32, gru_units=64, name="keystroke_branch")
    
    hm = branch(inputs)
    
    # Add a temporary classification head for pre-training
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax', name="subject_classifier")(hm)
    
    model = tf.keras.models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model, branch

def train_keystroke_model():
    csv_path = "data/keystroke/raw/DSL-StrongPasswordData.csv"
    if not os.path.exists(csv_path):
        print(f"[ERROR] Raw keystroke data not found at {csv_path}")
        sys.exit(1)
        
    X, y, num_classes = load_and_prepare_keystroke_data(csv_path, T=10)
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model, branch_layer = build_unimodal_keystroke_model(input_shape=(10, 7), num_classes=num_classes)
    model.summary()
    
    checkpoint_dir = "results/checkpoints/"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # We train the full classification model
    print("Pre-training Keystroke ModalityBranch on Subject Classification (Identity Proxy)...")
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=128,
        callbacks=[early_stop]
    )
    
    # We save ONLY the weights of the ModalityBranch so it can be cleanly loaded into fusion later
    branch_weights_path = os.path.join(checkpoint_dir, "keystroke_encoder.weights.h5")
    
    # Create a tiny model just for the branch to save it natively
    branch_input = tf.keras.layers.Input(shape=(10, 7))
    branch_output = branch_layer(branch_input)
    save_model = tf.keras.models.Model(branch_input, branch_output)
    save_model.save_weights(branch_weights_path)
    
    print(f"\n[SUCCESS] Unimodal Keystroke encoder weights saved to {branch_weights_path}")

if __name__ == "__main__":
    train_keystroke_model()
