"""Rank candidate windows for slides 5-7 from computed detector outputs (Aim1, good signal only)."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deck_common import FS, D, aim1_signal, cohort3_signal, trace_detector, wb  # noqa: E402

PP = D["PEAK_PARAMS"]
W = 4 * FS
s5, s6, s7 = [], [], []
keys = [k for k in sorted(wb.all_recordings()) if k not in wb.EXCLUDE]
for k in keys:
    sig, _ = aim1_signal(k)
    T = trace_detector(sig)
    ex = wb.bad_signal_mask(sig, FS)[0]
    rA = pd.DataFrame(T["rows_A"])
    rejA = rA[rA.reason != "counted"] if len(rA) else rA
    pk, tr = T["pk_final"], T["troughs"]
    # slide 7: every adjacent final pair, margin on the validation-stage y (Pass-A baseline)
    yA, lr = T["y_A"], T["lr_B"]
    need = lambda i: max(PP["prominence_frac"] * lr[i], PP["noise_mult"] * T["noise_ptp"])  # noqa: E731
    for a, b in zip(pk[:-1], pk[1:]):
        if b - a > 0.5 * FS or ex[a:b + 1].any() or np.isnan(yA[a:b + 1]).any():
            continue
        t = a + int(np.argmin(yA[a:b]))
        L = a if yA[a] <= yA[b] else b
        s7.append(dict(key=k, a=a / FS, b=b / FS, ratio=(yA[L] - yA[t]) / need(L), delta=yA[L] - yA[t], need=need(L)))
    for s in range(0, len(sig) - W, FS):
        e = s + W
        if ex[s:e].any() or not T["valid"][s:e].all():
            continue
        lrw = T["lr_A"][s:e]
        nrej = int(((rejA.idx >= s) & (rejA.idx < e)).sum()) if len(rejA) else 0
        s5.append(dict(key=k, start=s / FS, lr_ratio=np.nanmax(lrw) / np.nanmin(lrw), rejA=nrej,
                       nA=int(((T["pkA"] >= s) & (T["pkA"] < e)).sum())))
        bl = T["bl_A"][s:e]
        pkw = pk[(pk >= s) & (pk < e)]
        amp = np.median(T["seg"][pkw] - T["bl_A"][pkw]) if len(pkw) else np.nan
        s6.append(dict(key=k, start=s / FS, bl_range=np.ptp(bl), amp=amp, ratio=np.ptp(bl) / amp,
                       same=int(len(pkw) == int(((T["pkA"] >= s) & (T["pkA"] < e)).sum()))))
s5, s6, s7 = map(pd.DataFrame, (s5, s6, s7))
pd.set_option("display.width", 200)
print("SLIDE 5 (LR ratio, Pass-A rejections):")
print(s5[s5.rejA >= 2].sort_values("lr_ratio", ascending=False).head(15).to_string())
print(s5.sort_values("lr_ratio", ascending=False).head(10).to_string())
print("\nSLIDE 6 (baseline excursion / breath amplitude):")
print(s6.sort_values("ratio", ascending=False).head(15).to_string())
print("\nSLIDE 7 closest accepted pairs:")
print(s7.sort_values("ratio").head(10).to_string())
print("typical ratio:", s7.ratio.median())
for key in sorted({k for k, *_ in wb.ORIGINAL_PANELS}):
    T = trace_detector(cohort3_signal(key))
    r = pd.DataFrame(T["rows_B"])
    print(key, r.reason.value_counts().to_dict())
