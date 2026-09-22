import numpy as np

from keyboard_stress_features import (
    KEYSTROKES_PER_SUBWINDOW, browser_events_to_raw, compute_subwindow_features, events_to_pairs,
)


def extract_keystroke_features(events):
    """
    7D keyboard features of the most recent complete 4-keystroke sub-window,
    computed with exactly the same code as the Freihaut training data
    (src/keyboard_stress_features.py).

    events: [{'key', 'code' (optional), 'action': 'press'|'release', 'time': seconds}]
    Features: mean_dwell, std_dwell, mean_flight, std_flight (ms),
              typing_speed (keys/s), backspace_freq, error_rate (always 0 at runtime)
    Returns zeros when fewer than 4 complete keystrokes are available.
    """
    if not events:
        return np.zeros(7, dtype=np.float32)
    pairs = events_to_pairs(browser_events_to_raw(events))
    n_full = (len(pairs) // KEYSTROKES_PER_SUBWINDOW) * KEYSTROKES_PER_SUBWINDOW
    for end in range(n_full, 0, -KEYSTROKES_PER_SUBWINDOW):
        feats = compute_subwindow_features(pairs[end - KEYSTROKES_PER_SUBWINDOW:end])
        if feats is not None:
            return feats
    return np.zeros(7, dtype=np.float32)
