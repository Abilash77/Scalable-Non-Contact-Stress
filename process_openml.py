import pandas as pd
import numpy as np
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the CSV
df = pd.read_csv('data/SaYoPillow/raw/dataset.csv')

# Drop any rows where target is NaN (already done during fetch, but just in case)
target_col = 'stresse_a_cet_instant'
df = df.dropna(subset=[target_col])

# The target in this dataset might be string or float. Let's binarize it.
# Often stress is 0 or 1, or string 'Oui' / 'Non'.
le = LabelEncoder()
y = le.fit_transform(df[target_col].astype(str))

# Keep only 2 classes if it's multiclass, or just use as is (assuming binary)
# To make it strictly binary stress vs non-stress:
if len(np.unique(y)) > 2:
    # If multiclass, binarize by median or just map >0 to 1
    y = (y > 0).astype(int)

# Features: exclude the target and anything that leaks (like 'Stress_dernier_mois')
drop_cols = [target_col, 'Stress_dernier_mois', 'evenements_stressant_derniers_mois', 'Surpoids']
X_df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# One-hot encode categoricals, fill NaNs
for col in X_df.columns:
    if X_df[col].dtype == 'object':
        X_df[col] = X_df[col].fillna('Missing')
    else:
        X_df[col] = X_df[col].fillna(X_df[col].median())

X_encoded = pd.get_dummies(X_df, drop_first=True).astype(float)

# Scale
scaler = StandardScaler()
X = scaler.fit_transform(X_encoded)

# Split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Save arrays
out_dir = 'data/openml_stress/processed'
os.makedirs(out_dir, exist_ok=True)

np.save(os.path.join(out_dir, 'X_train.npy'), X_train)
np.save(os.path.join(out_dir, 'y_train.npy'), y_train)
np.save(os.path.join(out_dir, 'X_val.npy'), X_val)
np.save(os.path.join(out_dir, 'y_val.npy'), y_val)
np.save(os.path.join(out_dir, 'X_test.npy'), X_test)
np.save(os.path.join(out_dir, 'y_test.npy'), y_test)

# Build inventory
inventory = {
    "total_samples": len(X),
    "features": X.shape[1],
    "splits": {
        "train": len(X_train),
        "val": len(X_val),
        "test": len(X_test)
    },
    "feature_names": X_encoded.columns.tolist(),
    "target_mapping": {str(cls): int(idx) for idx, cls in enumerate(le.classes_)}
}

with open('results/dataset_inventory.json', 'w') as f:
    json.dump(inventory, f, indent=4)

print("Processing complete. Inventory saved to results/dataset_inventory.json")
