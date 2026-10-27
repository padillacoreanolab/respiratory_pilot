"""Shared paths, recording lists and loaders for the real-data detector validation.

Signals are loaded with the SAME loaders the pipelines use:
  rank    : load_rank_resp_signal_interactions(h5, target_rate=400, prefer_raw=True)
  valence : load_clean_resp_signal(h5, target_rate=400)
and cached (read-only copies) so later stages do not reread the 20 kHz raw data.
"""
import glob
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESP = os.path.dirname(HERE)
sys.path[:0] = [HERE, RESP]
warnings.filterwarnings("ignore")

FS = 400
DROPBOX = "C:/Users/thoma/UF Dropbox/Thomas Heeps/Padilla-Coreano Lab/2025/ECG_cohort1/Aim1/AIM1/Day1_new"
RANK_DIR = DROPBOX + "/CM interactions/cm_h5/baseline_cagemate_interactions_h5_outputs"
VALENCE_DIR = DROPBOX + "/Resp_h5"
# the 7 recordings used by rank-featureextraction-3sec.ipynb (cell 12); drops per EXCLUSIONS_LOG
RANK_TRIALS = {
    "CM_1_1_d1_2": ("CM_s1_1_d1_2_20250623_111352_merged.h5", "1_1", "Subordinate"),
    "CM_1_2_sub1_1": ("CM_s1_2_sub1_1_20250623_133932_merged.h5", "1_2", "Dominant"),
    "CM_2_3_d2_4": ("CM_s2_3_d2_4_20250623_151153_merged.h5", "2_3", "Subordinate"),
    "CM_2_4_sub2_3": ("CM_s2_4_sub2_3_20250623_143348_merged.h5", "2_4", "Dominant"),
    "CM_3_5_d3_6": ("CM_s3_5_d3_6_20250623_160001_merged.h5", "3_5", "Subordinate"),
    "CM_4_7_d4_8": ("CM_s4_7_d4_8_20250623_193718_merged.h5", "4_7", "Subordinate"),
    "CM_4_8_sub4_7": ("CM_s4_8_sub4_7_20250623_182649_merged.h5", "4_8", "Dominant"),
}
CACHE = os.environ.get("BREATH_VALIDATION_CACHE",
                       "C:/Users/thoma/AppData/Local/Temp/claude/c--Users-thoma-Code-ResearchCode-respiratory-pilot/"
                       "8e717e36-2ecb-467d-937f-f42ae203126f/scratchpad/signal_cache")


def valence_trials():
    out = {}
    for p in sorted(glob.glob(VALENCE_DIR + "/*.h5")):
        b = os.path.basename(p)
        cond = b.split("_")[0]                       # RI1 / RI2 / BLRI
        subj = "_".join(b.split("_")[1:3]).lstrip("s")
        out[f"{cond}_{subj}"] = (b, subj, cond)
    return out


def all_recordings():
    rec = {k: dict(cohort="rank", file=os.path.join(RANK_DIR, f), subject=s, label=r) for k, (f, s, r) in RANK_TRIALS.items()}
    for k, (f, s, c) in valence_trials().items():
        rec[k] = dict(cohort="valence", file=os.path.join(VALENCE_DIR, f), subject=s, label=c)
    return rec


def load_signal(key, rec):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"{key}.npz")
    if os.path.exists(path):
        d = np.load(path)
        return d["signal"], d["time"]
    from resp_helper_functions import load_clean_resp_signal, load_rank_resp_signal_interactions
    if rec["cohort"] == "rank":
        sig, t, fs, _ = load_rank_resp_signal_interactions(rec["file"], target_rate=FS, prefer_raw=True)
    else:
        sig, t, fs, _ = load_clean_resp_signal(rec["file"], target_rate=FS, verbose=False)
    assert fs == FS, (key, fs)
    np.savez(path, signal=sig, time=t)
    return sig, t


def dev_mask(n, fs=FS):
    """Development samples = odd-numbered minutes (60-120 s, 180-240 s, ...)."""
    return ((np.arange(n) // int(60 * fs)) % 2) == 1


def match_idx(a, b, tol):
    """Boolean per element of a: a nearby element of b exists within tol samples."""
    a = np.asarray(a, float)
    b = np.sort(np.asarray(b, float)[np.isfinite(b)])
    if not len(b) or not len(a):
        return np.zeros(len(a), bool)
    i = np.clip(np.searchsorted(b, a), 1, len(b) - 1)
    return np.minimum(np.abs(b[i] - a), np.abs(b[i - 1] - a)) <= tol
