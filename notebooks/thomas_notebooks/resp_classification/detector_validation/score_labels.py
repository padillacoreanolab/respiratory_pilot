"""Score human labels against the detectors.   python score_labels.py <out_dir>

Needs <out_dir>/labeling/labels.csv (item_id,label from label_breaths.html) and answer_key.csv.
Per class: fraction labeled breath (Wilson 95% CI). From these, for disputed peaks:
  legacy_only labeled breath   -> breaths the adaptive detector misses
  adaptive_only labeled breath -> breaths the legacy detector misses
  adaptive_only labeled not    -> adaptive false positives, etc.
Precision/recall are then estimated by re-weighting each class by its total count in the
evaluation data (detection_per_recording.csv), i.e. stratified estimates, not raw ratios."""
import os
import sys

import numpy as np
import pandas as pd

OUT = sys.argv[1]
L = os.path.join(OUT, "labeling")
key = pd.read_csv(os.path.join(L, "answer_key.csv"))
lab = pd.read_csv(os.path.join(L, "labels.csv"))
d = key.merge(lab, on="item_id")
d = d[d.label != "unsure"]


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan, np.nan
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return p, c - h, c + h


rows = []
for cls, g in d.groupby("cls"):
    p, lo, hi = wilson((g.label == "breath").sum(), len(g))
    rows.append(dict(cls=cls, n_labeled=len(g), frac_breath=p, ci_lo=lo, ci_hi=hi))
t = pd.DataFrame(rows).set_index("cls")
print(t.round(3).to_string())

det = pd.read_csv(os.path.join(OUT, "detection_per_recording.csv"))
n_both = (det.n_adaptive - det.adaptive_only).sum()
n_lo, n_ao = det.legacy_only.sum(), det.adaptive_only.sum()
pb = t.frac_breath.get("both", np.nan)
plo = t.frac_breath.get("legacy_only", np.nan)
pao = t.frac_breath.get("adaptive_only", np.nan)
true_both, true_lo, true_ao = n_both * pb, n_lo * plo, n_ao * pao
real = true_both + true_lo + true_ao       # real breaths found by at least one detector
est = dict(
    legacy_precision=(true_both + true_lo) / (n_both + n_lo),
    adaptive_precision=(true_both + true_ao) / (n_both + n_ao),
    legacy_recall_vs_union=(true_both + true_lo) / real,
    adaptive_recall_vs_union=(true_both + true_ao) / real)
print("\nStratified estimates (recall is relative to breaths found by at least one detector):")
for k, v in est.items():
    print(f"  {k}: {v:.3f}")
pd.DataFrame([est]).to_csv(os.path.join(L, "label_scores.csv"), index=False)
t.to_csv(os.path.join(L, "label_class_rates.csv"))
