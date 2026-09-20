# FREIHAUT & GÖRITZ FEATURE MAPPING

## ORIGINAL 11 FEATURES
The 11 keyboard typing features extracted by Freihaut & Göritz (2021) are natively derived from trial-level aggregated raw JSON `KeyDown` and `KeyUp` events.

| Original Feature | Definition (From Source Code) | Raw fields required | Can derive from released raw events? | Current 7D feature equivalent? | Exact transformation |
|---|---|---|---|---|---|
| **Backspace_presses** | Count of 'Backspace' KeyDown events in a trial | `key`, `eventType` | YES | `backspace_freq` | Divide raw backspace count by total keystrokes in trial |
| **Writing_Time** | Total time from first keystroke to task end | `time`, `eventType` | YES | N/A (Aggregate) | Use total window time |
| **Time_per_Keystroke** | `Writing_Time` / `Number_of_keypresses` | `time`, `eventType` | YES | `typing_speed` | Inverse of `Time_per_Keystroke` yields keystrokes/sec |
| **Dwelltime_Mean** | Mean time between KeyDown and KeyUp of the same key | `code`, `time`, `eventType` | YES | `mean_dwell` | Direct mapping (extract millisecond mean) |
| **Dwelltime_SD** | Std dev of Dwell time | `code`, `time`, `eventType` | YES | `std_dwell` | Direct mapping |
| **Latency_Mean** | Mean time between KeyUp of previous key and KeyDown of next key | `code`, `time`, `eventType` | YES | `mean_flight` | Direct mapping (Latency = Flight time) |
| **Latency_SD** | Std dev of Latency time | `code`, `time`, `eventType` | YES | `std_flight` | Direct mapping |
| **Trial_Onset_Mean** | Mean time between trial start and first keystroke across tasks | `time` | YES | N/A | Ignored for continuous 7D windows |
| **Trial_Onset_SD** | Std dev of Trial Onset times | `time` | YES | N/A | Ignored |
| **Trial_Time_Mean** | Mean total duration of trial tasks | `time` | YES | N/A | Ignored |
| **Trial_Time_SD** | Std dev of total trial durations | `time` | YES | N/A | Ignored |

## COMPATIBILITY WITH 10-TIMESTEP 7D MODEL

The raw data (`HS_PatternTyping`, `LS_PatternTyping`, `PR_PatternTyping`) contains temporally sequential `KeyDown` and `KeyUp` events with exact milliseconds `time`.

**Temporal Representation Generation:**
Because the raw JSON contains the exact chronological keystrokes, we do NOT have to repeat the 11 static aggregated features 10 times.
We can sequentially group the raw keystrokes into temporal sub-windows (e.g., $N$ keystrokes or $N$ seconds per step), compute the 7D vector for each step, and construct the required `(10, 7)` temporal tensor per trial.

**Feature Subset:**
We will use the defensible 7-feature subset (`mean_dwell, std_dwell, mean_flight, std_flight, typing_speed, backspace_freq, error_rate`) naturally derived from the raw sequence, discarding the 4 static trial-level macro features (`Trial_Onset_Mean`, `Trial_Onset_SD`, `Trial_Time_Mean`, `Trial_Time_SD`) which are incompatible with continuous windowing. The error rate will be derived from the `isCorrect` field provided in the raw data.
