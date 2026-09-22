#!/usr/bin/env python3
"""
One-time extraction of Freihaut & Goeritz pattern-typing keyboard events.

Loads the raw lab + online study JSON (online file is ~288 MB gzip) once and
stores ONLY the keyboard events of the PatternTyping trials, together with the
participant id, study, task and experimental condition, in a compact pickle:

    data/processed/freihaut/pattern_typing_trials.pkl.gz

No features, labels or splits are computed here.
"""
import gzip
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(__file__))
from preprocess_freihaut import _convert_online_tracker_data  # noqa: E402

RAW = "data/freihaut_git"
OUT = "data/processed/freihaut/pattern_typing_trials.pkl.gz"


def _key_events(events):
    return [e for e in events if isinstance(e, dict) and e.get("eventType") in ("KeyDown", "KeyUp")]


def main():
    trials = []
    lab = json.load(gzip.open(os.path.join(RAW, "Data-Analysis-Lab-Study", "Data", "LabStudy_RawData.json.gz"), "rt"))
    for pid, pdata in lab.items():
        completed = "BfiNeuroticism" in pdata.get("phaseFinishedTimestamps", {})
        for task in ("PR_PatternTyping", "LS_PatternTyping", "HS_PatternTyping"):
            ev = pdata.get(task)
            if isinstance(ev, list):
                trials.append({"study": "lab", "pid": f"lab_{pid}", "task": task, "completed": completed,
                               "condition": None, "events": _key_events(ev)})
    print(f"lab trials: {len(trials)}")
    del lab

    ids = pickle.load(open(os.path.join(RAW, "Data-Analysis-Online-Study", "Data", "filtered_online_study_ids_studyLevel"), "rb"))
    online = json.load(gzip.open(os.path.join(RAW, "Data-Analysis-Online-Study", "Data", "OnlineStudy_RawData.json.gz"), "rt"))
    n0 = len(trials)
    for pid, pdata in online.items():
        try:
            cond = pdata["ExperimentMetaData"]["condition"]
        except (KeyError, TypeError):
            cond = None
        for task in ("Pr_PatternTyping", "Con_PatternTyping"):
            try:
                td = pdata[task]["data"]["TrackerData"]
            except (KeyError, TypeError):
                continue
            raw = list(td.values()) if isinstance(td, dict) else td
            n_practice = sum(1 for e in raw if isinstance(e, dict) and e.get("isPractice"))
            ev = _convert_online_tracker_data(td) if isinstance(td, dict) else _key_events(td)
            trials.append({"study": "online", "pid": f"online_{pid}", "task": task, "completed": pid in ids,
                           "condition": cond, "n_practice_events": n_practice, "events": ev})
    print(f"online trials: {len(trials) - n0}")
    with gzip.open(OUT, "wb") as f:
        pickle.dump(trials, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
