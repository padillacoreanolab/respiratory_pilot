"""Synthetic ground-truth benchmark: each downstream quantity scored separately, legacy vs
adaptive, on identical signals.   python synthetic_benchmark.py <out_dir>"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]
warnings.filterwarnings("ignore")
from breathmetrics import bmObject  # noqa: E402
from synthetic import make_signal, match, regular  # noqa: E402

FS = 400
TOL = int(0.03 * FS)


def scenarios():
    rng = np.random.default_rng(0)
    out = {}
    out["clean_6Hz"] = (regular(400, 6, A=100), {})
    out["drift+dc"] = (regular(400, 6, A=100), dict(drift=[(150, 0.3), (60, 1.1)], dc=3000))
    amp = regular(400, 6, A=100)
    for i in range(130, 270):
        amp[i] = dict(amp[i], A=float(rng.uniform(10, 40)))
    out["amplitude_10-40%"] = (amp, {})
    out["pauses"] = ([dict(Ti=.06, Te=.1, A=100, Tpe=.08 if i % 2 else 0, Tpi=.04 if i % 3 == 0 else 0)
                      for i in range(400)], {})
    T = rng.uniform(0.07, 0.5, 400)
    out["irregular_2-14Hz"] = ([dict(Ti=t * .4, Te=t * .6, A=float(a)) for t, a in zip(T, rng.uniform(30, 150, 400))], {})
    mix = [dict(c, A=c["A"] * (0.2 if 150 <= i < 250 else 1)) for i, c in enumerate(out["pauses"][0])]
    out["pauses+amp+drift"] = (mix, dict(drift=[(120, 0.4)], dc=1000))
    return out


def fit(x, det):
    bm = bmObject(x, FS, "rodentAirflow")
    if det == "legacy":
        bm.estimateAllFeatures()
    else:
        bm.estimateAllFeatures(detector="adaptive_prominence")
    return bm


def score(bm, tr):
    peaks = np.asarray(bm.inhalePeaks, float)
    tp, fp, fn, _, pairs = match(tr["peak"], peaks, TOL)
    on = np.asarray(bm.inhaleOnsets, float)
    _, _, _, e_on, _ = match(tr["inhale_onset"], on, TOL)
    # rate over the whole signal, from detected onsets (as extract_bout_features does)
    det_ibi = np.diff(np.sort(on[np.isfinite(on)])) / FS
    true_ibi = np.diff(tr["inhale_onset"]) / FS
    # per-breath durations / volumes for breaths matched by peak
    ti_true = (np.where(tr["inhale_pause_onset"] >= 0, tr["inhale_pause_onset"], tr["exhale_onset"]) - tr["inhale_onset"]) / FS
    idur = np.asarray(bm.inhaleDurations, float).ravel()
    ivol = np.asarray(bm.inhaleVolumes, float).ravel()
    d_err, v_err = [], []
    for ti, j in pairs:
        if j < len(idur) and np.isfinite(idur[j]):
            d_err.append(idur[j] - ti_true[ti])
        if j < len(ivol) and np.isfinite(ivol[j]):
            v_err.append(ivol[j] / tr["inhale_area"][ti] - 1)
    # pauses: exhale pauses, matched by onset time
    epo = np.asarray(bm.exhalePauseOnsets, float).ravel()
    t_ep = tr["exhale_pause_onset"][tr["exhale_pause_onset"] >= 0]
    ptp_, pfp, pfn, _, _ = match(t_ep, epo, TOL)
    return dict(peak_recall=tp / max(tp + fn, 1), peak_precision=tp / max(tp + fp, 1),
                onset_med_abs_err_ms=1000 * np.median(np.abs(e_on)) / FS if len(e_on) else np.nan,
                rate_err_pct=100 * (np.mean(true_ibi) / np.mean(det_ibi) - 1) if len(det_ibi) else np.nan,
                inhale_dur_med_abs_err_ms=1000 * np.median(np.abs(d_err)) if d_err else np.nan,
                inhale_vol_med_rel_err_pct=100 * np.median(v_err) if v_err else np.nan,
                exh_pause_recall=ptp_ / max(len(t_ep), 1) if len(t_ep) else np.nan,
                exh_pause_false=pfp)


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for name, (cyc, kw) in scenarios().items():
        x, tr, _ = make_signal(FS, cyc, noise_sd=1.0, seed=1, **kw)
        for det in ["legacy", "adaptive_prominence"]:
            try:
                r = score(fit(x, det), tr)
            except Exception as ex:  # noqa: BLE001
                r = dict(error=f"{type(ex).__name__}: {ex}")
            rows.append(dict(scenario=name, detector=det, **r))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "synthetic_benchmark.csv"), index=False)
    pd.set_option("display.width", 250)
    print(df.round(3).to_string(index=False))
