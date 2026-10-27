"""Find real examples for each slide from computed detector / QC outputs (not by eye).
Writes explore_examples.txt next to this file."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deck_common import FS, D, aim1_signal, cohort3_signal, trace_detector, wb  # noqa: E402

out = []
p = lambda *a: (out.append(" ".join(str(x) for x in a)), print(*a))  # noqa: E731

snap = wb.load_snapshot("v1")
keys = [k for k in sorted(wb.all_recordings()) if k not in wb.EXCLUDE]
p("Aim1 recordings:", len(keys))
tot_pk, tot_min, tot_ex, n_aff = 0, 0.0, 0.0, 0
rows_b, blocks_all, qc = [], [], []
for k in keys:
    sig, t = aim1_signal(k)
    T = trace_detector(sig)
    same = np.array_equal(T["pk_final"], snap[k])
    ex, reason = wb.bad_signal_mask(sig, FS)
    keep = ~ex
    pk = T["pk_final"][keep[T["pk_final"]]]
    used = T["valid"] & keep
    tot_pk += len(pk)
    tot_min += used.sum() / FS / 60
    tot_ex += ex.sum() / FS / 60
    n_aff += ex.any()
    p(f"{k:15s} snapshot_match={same} peaks_kept={len(pk)} analysed_min={used.sum() / FS / 60:.3f} "
      f"pkA={len(T['pkA'])} pkB_raw={len(T['pkB_raw'])} final={len(T['pk_final'])}")
    for r in T["rows_B"]:
        rows_b.append(dict(r, key=k, excluded=bool(ex[r["idx"]])))
    b = T["blocks"].copy()
    b["key"] = k
    blocks_all.append(b)
    for rs in ["dead", "artifact", "degraded"]:
        m = np.r_[False, reason == rs, False]
        d = np.diff(m.astype(int))
        for a, z in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
            qc.append(dict(key=k, reason=rs, start_s=a / FS, stop_s=z / FS, dur_s=(z - a) / FS))
p(f"TOTAL peaks={tot_pk} analysed_min={tot_min:.3f} excluded_min={tot_ex:.3f} affected={n_aff}/{len(keys)} "
  f"rate={tot_pk / (tot_min * 60):.4f}")

rb = pd.DataFrame(rows_b)
p("\nPass-B rejection reasons (all Aim1):\n", rb.groupby(["reason", "excluded"]).size().to_string())
st = rb[(rb.reason == "shallow_trough") & ~rb.excluded].copy()
st["ratio"] = st.prominence / st.required
p("\nshallow_trough (not QC-excluded), closest to threshold first:\n",
  st.sort_values("ratio", ascending=False).head(25).assign(t=lambda d: d.idx / FS).to_string())
sw = rb[(rb.reason == "spike_width") & ~rb.excluded]
p("\nspike_width examples:\n", sw.assign(t=lambda d: d.idx / FS).head(15).to_string())
tc = rb[(rb.reason == "too_close") & ~rb.excluded]
p("\ntoo_close examples:\n", tc.assign(t=lambda d: d.idx / FS).head(15).to_string())

B = pd.concat(blocks_all, ignore_index=True)
p("\nComponent A decisions (Aim1):\n", B.decision.value_counts().to_string())
p("\nlow_amp (periodic) blocks:\n", B[B.decision == "low_amp (periodic)"].head(40).to_string())
p("\nflat_or_dropout blocks (first 20):\n", B[B.decision == "flat_or_dropout"].head(20).to_string())

Q = pd.DataFrame(qc)
p("\nFrozen QC core stretches by reason:\n", Q.groupby("reason").size().to_string())
p("\nartifact stretches:\n", Q[Q.reason == "artifact"].to_string())
p("\ndegraded stretches (longest 20):\n", Q[Q.reason == "degraded"].sort_values("dur_s", ascending=False).head(20).to_string())
p("\ndead stretches (longest 15):\n", Q[Q.reason == "dead"].sort_values("dur_s", ascending=False).head(15).to_string())

# cohort 3 Section 1 sessions
for key, t0, t1, note in wb.ORIGINAL_PANELS:
    sig = cohort3_signal(key)
    T = trace_detector(sig)
    s0, s1 = int(t0 * FS), int(t1 * FS)
    pk = T["pk_final"][(T["pk_final"] >= s0) & (T["pk_final"] < s1)]
    leg, lx, thr = wb._cohort3_legacy_fit(key, sig)
    lp = leg[(leg >= s0) & (leg < s1)]
    p(f"C3 {key} {t0}-{t1}: NEW={len(pk)} OLD={len(lp)} thr={thr:.1f} noise={T['noise_ptp']:.2f}")
    b = T["blocks"]
    bw = b[(b.start_s >= t0 - 1) & (b.start_s < t1)]
    if (bw.decision != "ok").any():
        p(bw.to_string())
sig = cohort3_signal("11_3_s_CM_11_4_d_20260729_122040")
p("11_3 abs max of 400 Hz signal:", float(np.max(np.abs(sig))), "argmax t:", int(np.argmax(np.abs(sig))) / FS)
open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "explore_examples.txt"), "w").write("\n".join(out))
