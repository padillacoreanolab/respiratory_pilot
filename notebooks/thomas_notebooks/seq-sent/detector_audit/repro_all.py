"""Check that the current, unmodified pipeline reproduces EVERY feature in the saved rank pickle."""
import os, sys
from multiprocessing import Pool
import numpy as np, pandas as pd
from audit_stage1 import DATA_DIR, TRIAL_FILES, FEATURE_COLS, FS
from resp_helper_functions import load_rank_resp_signal_interactions, fit_bm_session, extract_features_from_windows

def one(args):
    trial, win = args
    sig, _, fs, _ = load_rank_resp_signal_interactions(os.path.join(DATA_DIR, TRIAL_FILES[trial]), target_rate=FS, prefer_raw=True)
    bm = fit_bm_session(sig, fs, data_type="rodentAirflow")
    rows = extract_features_from_windows(bm, list(zip(win.Start, win.Stop)), min_breaths_per_window=5)
    return pd.DataFrame(rows).assign(Trial=trial)

if __name__ == "__main__":
    orig = pd.read_csv(sys.argv[1])
    with Pool(4) as p:
        rep = pd.concat(p.map(one, [(t, orig[orig.Trial == t]) for t in TRIAL_FILES]), ignore_index=True)
    m = orig.merge(rep, on=["Trial", "Start", "Stop"], suffixes=("", "_rep"))
    print(f"windows: saved={len(orig)} reproduced={len(rep)} matched={len(m)}")
    bad = {f: int((~np.isclose(m[f], m[f + "_rep"], rtol=1e-9, atol=1e-12, equal_nan=True)).sum()) for f in FEATURE_COLS}
    print("features with any mismatch:", {k: v for k, v in bad.items() if v} or "NONE - all 27 features identical in all windows")
