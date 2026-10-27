"""Preprocess cohort 2 CM h5s with Thomas's pipeline and compute 10-s window QC metrics."""
import os, sys, glob
import numpy as np
import pandas as pd
from scipy.signal import welch

sys.path.insert(0, r"C:\Users\thoma\Code\ResearchCode\respiratory_pilot\notebooks\thomas_notebooks\resp_classification")
import resp_helper_functions as rhf

H5 = r"C:\Users\thoma\UF Dropbox\Thomas Heeps\Padilla-Coreano Lab\2026\BLA_ChR_resp_cohort2\BLA_ChR_resp_CM_hc\h5_outputs"
OUT = os.path.dirname(os.path.abspath(__file__))
FS = 400
WIN = 10  # s, same window as mask_signal_loss in signal_loss.ipynb

rows, signals = [], {}
for p in sorted(glob.glob(os.path.join(H5, "*.h5"))):
    name = os.path.basename(p)[:-3]
    sig, t, *_ = rhf.load_rank_resp_signal_interactions(p, target_rate=FS, prefer_raw=True)
    signals[name] = sig.astype(np.float32)
    n = int(WIN * FS)
    for w in range(len(sig) // n):
        x = sig[w * n:(w + 1) * n]
        f, pxx = welch(x, fs=FS, nperseg=4 * FS)
        tot = pxx[(f >= 0.1) & (f <= 20)].sum()
        band = pxx[(f >= 1) & (f <= 15)].sum()
        rows.append(dict(session=name, win=w, t0=w * WIN, std=x.std(),
                         p99=np.percentile(np.abs(x - np.median(x)), 99),
                         maxabs=np.abs(x - np.median(x)).max(),
                         band_frac=band / tot if tot > 0 else 0,
                         peak_hz=f[(f >= 0.5) & (f <= 20)][np.argmax(pxx[(f >= 0.5) & (f <= 20)])]))
df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "qc_windows.csv"), index=False)
np.savez_compressed(os.path.join(OUT, "processed_400hz.npz"), **signals)
pd.set_option("display.width", 200)
print(df.groupby("session")[["std", "p99", "band_frac", "peak_hz"]].median().round(2))
print(df[["std", "p99", "maxabs", "band_frac", "peak_hz"]].describe(percentiles=[.05, .1, .25, .5, .75, .9, .95]).round(2))
