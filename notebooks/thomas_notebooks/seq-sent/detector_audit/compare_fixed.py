"""Re-run the rank pipeline with the FIXED breathmetrics.py and compare with the saved pickle features."""
import sys
from multiprocessing import Pool
import numpy as np, pandas as pd
from audit_stage1 import TRIAL_FILES, FEATURE_COLS
from repro_all import one

if __name__ == "__main__":
    orig = pd.read_csv(sys.argv[1])
    with Pool(4) as p:
        new = pd.concat(p.map(one, [(t, orig[orig.Trial == t]) for t in TRIAL_FILES]), ignore_index=True)
    m = orig.merge(new, on=["Trial", "Start", "Stop"], how="left", suffixes=("", "_new"), indicator=True)
    print(f"saved windows={len(orig)}; still pass >=5 breaths with fixed code={int((m._merge == 'both').sum())}")
    m = m[m._merge == "both"]
    rows = []
    for f in FEATURE_COLS:
        a, b = m[f], m[f + "_new"]
        ok = a.notna() & b.notna()
        rel = ((b - a) / a.abs().replace(0, np.nan))[ok]
        rows.append(dict(feature=f, pct_windows_changed=100 * (~np.isclose(a, b, equal_nan=True)).mean(),
                         median_abs_change_pct=100 * rel.abs().median(), spearman=a[ok].corr(b[ok], method="spearman")))
    print(pd.DataFrame(rows).round(2).to_string(index=False))
    print("\nn_breaths change per window:", (m.n_breaths_new - m.n_breaths).value_counts().sort_index().to_dict())
