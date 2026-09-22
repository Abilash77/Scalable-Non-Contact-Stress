"""
Keyboard stress model training (Freihaut & Goeritz 2021).

    python scripts/preprocess_freihaut.py   # raw (N,10,7) arrays + participant ids
    python train.py --retrain               # model selection + one test evaluation

Protocol
  * participant-independent train / val / test (from preprocessing)
  * scaler (log + z-score) fitted on TRAIN only, saved next to the checkpoint
  * class weights from TRAIN only
  * every candidate / hyper-parameter is scored on VALIDATION only (ROC-AUC)
  * decision threshold chosen on VALIDATION (max balanced accuracy)
  * the selected model is evaluated ONCE on the untouched TEST split
"""
import argparse
import datetime
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from keyboard_stress_features import (  # noqa: E402
    FEATURE_NAMES, KEYSTROKES_PER_SUBWINDOW, KeyboardFeatureTransform, summary_features,
)

DATA_DIR = os.path.join(ROOT, 'data', 'processed', 'freihaut')
CP_DIR = os.path.join(ROOT, 'results', 'checkpoints')
META_FILE = 'keyboard_stress_model_metadata.json'
SCALER_FILE = 'keyboard_stress_scaler.json'
SEED = 42


def load_split(name):
    X = np.load(os.path.join(DATA_DIR, f'X_{name}.npy'))
    y = np.load(os.path.join(DATA_DIR, f'y_{name}.npy'))
    pid = np.load(os.path.join(DATA_DIR, f'pid_{name}.npy'))
    trial = np.load(os.path.join(DATA_DIR, f'trial_{name}.npy'))
    return X, y, pid, trial


def validate_preprocessing():
    needed = [f'{k}_{s}.npy' for k in ('X', 'y', 'pid', 'trial') for s in ('train', 'val', 'test')]
    missing = [n for n in needed if not os.path.exists(os.path.join(DATA_DIR, n))]
    if missing:
        sys.exit(f"[ERROR] Missing preprocessed arrays {missing}. Run: python scripts/preprocess_freihaut.py")
    with open(os.path.join(ROOT, 'results', 'freihaut_leakage_audit.json')) as f:
        if not json.load(f).get('PASS'):
            sys.exit("[ERROR] Leakage audit failed; fix preprocessing first.")
    splits = {s: load_split(s) for s in ('train', 'val', 'test')}
    p = {s: set(splits[s][2]) for s in splits}
    assert not (p['train'] & p['val']) and not (p['train'] & p['test']) and not (p['val'] & p['test']), \
        "participant leakage between splits"
    return splits


def metrics(y, prob, thr):
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                                 f1_score, precision_score, recall_score, roc_auc_score)
    pred = (prob >= thr).astype(int)
    return {
        'accuracy': float(accuracy_score(y, pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y, pred)),
        'precision': float(precision_score(y, pred, zero_division=0)),
        'recall': float(recall_score(y, pred, zero_division=0)),
        'f1': float(f1_score(y, pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y, prob)),
        'confusion_matrix': confusion_matrix(y, pred, labels=[0, 1]).tolist(),
        'n': int(len(y)),
    }


def pick_threshold(y, prob):
    from sklearn.metrics import balanced_accuracy_score
    grid = np.unique(np.quantile(prob, np.linspace(0.05, 0.95, 91)))
    scores = [balanced_accuracy_score(y, (prob >= t).astype(int)) for t in grid]
    return float(grid[int(np.argmax(scores))])


def trial_level(y, prob, trial):
    """Average window probabilities per trial (secondary metric)."""
    ids = np.unique(trial)
    ty = np.array([y[trial == t][0] for t in ids])
    tp = np.array([prob[trial == t].mean() for t in ids])
    return ty, tp


# ---------------------------------------------------------------------------
# Candidates. Each returns (fitted_model, val_prob, save_fn)
# ---------------------------------------------------------------------------

def sklearn_candidates(cw):
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    cands = []
    for C in (0.01, 0.1, 1.0):
        cands.append((f'logreg_C{C}', LogisticRegression(C=C, class_weight='balanced', max_iter=2000)))
    for depth in (2, 3):
        cands.append((f'hgb_d{depth}', HistGradientBoostingClassifier(
            max_depth=depth, learning_rate=0.05, max_iter=200, l2_regularization=1.0,
            min_samples_leaf=40, class_weight='balanced', random_state=SEED)))
    for leaf in (10, 40):
        cands.append((f'rf_leaf{leaf}', RandomForestClassifier(
            n_estimators=400, min_samples_leaf=leaf, max_features='sqrt',
            class_weight='balanced', n_jobs=-1, random_state=SEED)))
    return cands


def build_gru():
    import tensorflow as tf
    inp = tf.keras.Input(shape=(10, 7), name='keyboard_sequence')
    x = tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(1e-3))(inp)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.GRU(16, kernel_regularizer=tf.keras.regularizers.L2(1e-3))(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    m = tf.keras.Model(inp, out, name='keyboard_gru')
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='binary_crossentropy')
    return m


def train_gru(Xtr, ytr, Xva, yva, cw, epochs):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(SEED)
    m = build_gru()
    m.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=epochs, batch_size=64, verbose=0,
          class_weight=cw, callbacks=[tf.keras.callbacks.EarlyStopping(
              monitor='val_loss', patience=10, restore_best_weights=True)])
    return m, m.predict(Xva, verbose=0).ravel()


def fusion_inputs(Xk):
    n = len(Xk)
    mask = np.zeros((n, 5), np.float32)
    mask[:, 2] = 1.0
    return {'input_audio': np.zeros((n, 10, 169), np.float32), 'input_face': np.zeros((n, 10, 12), np.float32),
            'input_keystroke': Xk, 'input_handwriting': np.zeros((n, 10, 9), np.float32),
            'input_eye': np.zeros((n, 10, 5), np.float32), 'input_mask': mask}


def train_fusion(Xtr, ytr, Xva, yva, cw, epochs):
    """Original RA-HMSD fusion architecture, keyboard branch only (other inputs masked)."""
    import tensorflow as tf
    from DL_models import custom_fusion_loss, get_model
    tf.keras.utils.set_random_seed(SEED)
    shapes = {'audio': (10, 169), 'face': (10, 12), 'keystroke': (10, 7), 'handwriting': (10, 9), 'eye': (10, 5)}
    m = get_model('fusion', input_shapes=shapes, num_classes=2)
    m.compile(optimizer='adam', loss=custom_fusion_loss(class_weights=cw))
    ydict = lambda y: {k: y for k in ('fusion_output', 'unimodal_audio', 'unimodal_face',  # noqa: E731
                                     'unimodal_keystroke', 'unimodal_handwriting', 'unimodal_eye')}
    m.fit(fusion_inputs(Xtr), ydict(ytr), validation_data=(fusion_inputs(Xva), ydict(yva)),
          epochs=epochs, batch_size=64, verbose=0,
          callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_fusion_output_loss', mode='min',
                                                      patience=10, restore_best_weights=True)])
    return m, m.predict(fusion_inputs(Xva), verbose=0)['fusion_output'][:, 1]


def main(epochs, retrain):
    meta_path = os.path.join(CP_DIR, META_FILE)
    if os.path.exists(meta_path) and not retrain:
        print(f"[MODEL] Trained keyboard stress model already exists ({meta_path}). Use --retrain to overwrite.")
        return

    splits = validate_preprocessing()
    Xtr_raw, ytr, pid_tr, _ = splits['train']
    Xva_raw, yva, pid_va, trial_va = splits['val']
    Xte_raw, yte, pid_te, trial_te = splits['test']
    print(f"[DATA] train={Xtr_raw.shape} ({len(set(pid_tr))} participants)  "
          f"val={Xva_raw.shape} ({len(set(pid_va))})  test={Xte_raw.shape} ({len(set(pid_te))})")

    scaler = KeyboardFeatureTransform().fit(Xtr_raw)  # TRAIN ONLY
    Xtr, Xva = scaler.transform(Xtr_raw), scaler.transform(Xva_raw)
    Str, Sva = summary_features(Xtr), summary_features(Xva)

    from sklearn.utils.class_weight import compute_class_weight
    w = compute_class_weight('balanced', classes=np.array([0, 1]), y=ytr)
    cw = {0: float(w[0]), 1: float(w[1])}
    print(f"[DATA] class weights (train): {cw}")

    # ---- majority baseline (majority class of TRAIN) ------------------------
    maj = int(np.bincount(ytr).argmax())
    results = {}

    # ---- candidates, validation only ----------------------------------------
    fitted = {}
    for name, model in sklearn_candidates(cw):
        model.fit(Str, ytr)
        p = model.predict_proba(Sva)[:, 1]
        fitted[name] = (model, p, 'summary', 'joblib')
    m, p = train_gru(Xtr, ytr, Xva, yva, cw, epochs)
    fitted['gru_sequence'] = (m, p, 'sequence', 'keras')
    m, p = train_fusion(Xtr, ytr, Xva, yva, cw, epochs)
    fitted['fusion_keyboard_only'] = (m, p, 'sequence', 'fusion_weights')

    from sklearn.metrics import roc_auc_score
    print("\n[VALIDATION] model selection (ROC-AUC, balanced accuracy @ val-tuned threshold)")
    for name, (_, p, _, _) in fitted.items():
        thr = pick_threshold(yva, p)
        results[name] = {'val_roc_auc': float(roc_auc_score(yva, p)), 'val_threshold': thr,
                         'val_balanced_accuracy': metrics(yva, p, thr)['balanced_accuracy']}
        print(f"  {name:22s} AUC={results[name]['val_roc_auc']:.4f}  "
              f"balAcc={results[name]['val_balanced_accuracy']:.4f}  thr={thr:.3f}")
    best = max(results, key=lambda n: results[n]['val_roc_auc'])
    model, pva, input_kind, fmt = fitted[best]
    thr = results[best]['val_threshold']
    print(f"\n[SELECTED] {best} (by validation ROC-AUC)")

    # ---- single evaluation on untouched TEST --------------------------------
    Xte = scaler.transform(Xte_raw)
    if fmt == 'joblib':
        pte = model.predict_proba(summary_features(Xte))[:, 1]
    elif fmt == 'keras':
        pte = model.predict(Xte, verbose=0).ravel()
    else:
        pte = model.predict(fusion_inputs(Xte), verbose=0)['fusion_output'][:, 1]
    test = metrics(yte, pte, thr)
    test_default = metrics(yte, pte, 0.5)
    ty, tp = trial_level(yte, pte, trial_te)
    test_trial = metrics(ty, tp, thr)
    majority = metrics(yte, np.full(len(yte), float(maj)), 0.5)
    majority['roc_auc'] = 0.5
    majority['predicted_class'] = maj

    # ---- save ----------------------------------------------------------------
    os.makedirs(CP_DIR, exist_ok=True)
    model_file = {'joblib': 'keyboard_stress_model.joblib', 'keras': 'keyboard_stress_model.keras',
                  'fusion_weights': 'keyboard_stress_fusion.weights.h5'}[fmt]
    if fmt == 'joblib':
        import joblib
        joblib.dump(model, os.path.join(CP_DIR, model_file))
    elif fmt == 'keras':
        model.save(os.path.join(CP_DIR, model_file))
    else:
        model.save_weights(os.path.join(CP_DIR, model_file))
    with open(os.path.join(CP_DIR, SCALER_FILE), 'w') as f:
        json.dump(scaler.to_dict(), f, indent=2)

    meta = {
        'dataset_name': 'Freihaut & Goeritz (2021) keyboard stress',
        'is_trained': True,
        'is_prototype': False,
        'trained_modalities': ['keyboard'],
        'model_name': best,
        'model_format': fmt,
        'model_file': model_file,
        'scaler_file': SCALER_FILE,
        'input_kind': input_kind,
        'input_shape': [10, 7],
        'feature_names': FEATURE_NAMES,
        'feature_notes': 'error_rate is not observable in free typing and is zeroed for training and runtime',
        'keystrokes_per_subwindow': KEYSTROKES_PER_SUBWINDOW,
        'decision_threshold': thr,
        'threshold_selection': 'validation split, max balanced accuracy',
        'selection_metric': 'validation ROC-AUC',
        'class_mapping': {'0': 'NON-STRESS', '1': 'STRESS'},
        'class_weights_train': cw,
        'training_timestamp': datetime.datetime.now().isoformat(timespec='seconds'),
        'participants': {'train': len(set(pid_tr)), 'val': len(set(pid_va)), 'test': len(set(pid_te))},
        'windows': {'train': int(len(ytr)), 'val': int(len(yva)), 'test': int(len(yte))},
        'validation_candidates': results,
        'test_metrics': test,
        'test_metrics_threshold_0_5': test_default,
        'test_metrics_trial_level': test_trial,
        'majority_baseline_test': majority,
    }
    with open(os.path.join(CP_DIR, META_FILE), 'w') as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(ROOT, 'results', 'final_evaluation.json'), 'w') as f:
        json.dump({k: meta[k] for k in ('model_name', 'decision_threshold', 'participants', 'windows',
                                        'validation_candidates', 'test_metrics', 'test_metrics_threshold_0_5',
                                        'test_metrics_trial_level', 'majority_baseline_test')}, f, indent=2)

    print("\n" + "=" * 60)
    print("FINAL TEST EVALUATION (untouched test split, evaluated once)")
    print("=" * 60)
    print(f"MODEL: {best}   threshold={thr:.3f}")
    for k in ('accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1', 'roc_auc'):
        print(f"TEST {k.upper():18s}: {test[k]:.4f}")
    print(f"CONFUSION MATRIX [[TN FP][FN TP]]: {test['confusion_matrix']}")
    print(f"TRIAL-LEVEL  acc={test_trial['accuracy']:.4f}  balAcc={test_trial['balanced_accuracy']:.4f}  "
          f"AUC={test_trial['roc_auc']:.4f}  (n={test_trial['n']} trials)")
    print(f"MAJORITY BASELINE (class {maj}): acc={majority['accuracy']:.4f}  balAcc={majority['balanced_accuracy']:.4f}")
    print(f"Saved: {os.path.join(CP_DIR, model_file)}, {SCALER_FILE}, {META_FILE}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--retrain", action="store_true", help="Overwrite existing trained model")
    args = ap.parse_args()
    main(args.epochs, args.retrain)
