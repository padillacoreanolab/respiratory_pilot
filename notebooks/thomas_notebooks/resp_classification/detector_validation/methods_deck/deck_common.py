"""Shared loaders and an instrumented (read-only) replay of the workbench detector.

The detector code is NOT copied: cells A-C are executed straight from
peak_detection_workbench.ipynb, so every figure uses exactly the notebook's functions.
`trace_detector` calls those same functions in the same order as the notebook's
detect_breaths and keeps the intermediate arrays; it asserts that its final peaks equal
detect_breaths' peaks, so the replay cannot drift from the real detector.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DV = os.path.dirname(HERE)                       # detector_validation/
RESP = os.path.dirname(DV)                       # resp_classification/
sys.path[:0] = [DV, RESP]
os.environ.setdefault("BREATH_VALIDATION_CACHE", os.path.join(DV, "workbench_cache"))

import workbench_support as wb                   # noqa: E402  (loads/plots only; no detection)

FS = 400
NOTEBOOK = os.path.join(DV, "peak_detection_workbench.ipynb")
LAB = "C:/Users/thoma/UF Dropbox/Thomas Heeps/Padilla-Coreano Lab/"
AIM1_CM = LAB + "2025/ECG_cohort1/Aim1/AIM1/Day1_new/CM interactions/cm_h5/baseline_cagemate_interactions_h5_outputs/"
AIM1_VAL = LAB + "2025/ECG_cohort1/Aim1/AIM1/Day1_new/Resp_h5/"
AIM1_REC = LAB + "2025/ECG_cohort1/Aim1/AIM1/Day1_new/CM interactions/Baseline cagemate interactions/"


def load_notebook_detector():
    """Execute notebook cells A, B, C (by their banner comments) into one namespace."""
    nb = json.load(open(NOTEBOOK, encoding="utf-8"))
    ns = {}
    found = []
    for c in nb["cells"]:
        src = "".join(c["source"])
        if c["cell_type"] == "code" and any(f"# CELL {k}:" in src for k in "ABC"):
            exec(compile(src, f"workbench-cell-{src.split('# CELL ')[1][0]}", "exec"), ns)
            found.append(src.split("# CELL ")[1][0])
    assert found == ["A", "B", "C"], found
    return ns


D = load_notebook_detector()


def trace_detector(signal, fs=FS):
    """Replay of D['detect_breaths'] keeping every intermediate, per valid segment."""
    P = D["DETECTOR_PARAMS"]
    PP = D["PEAK_PARAMS"]
    x = P["polarity"] * np.asarray(signal, float)
    n = len(x)
    valid, reason, low_amp, segments, noise_ptp, blocks = D["validity_mask"](x, fs)
    sn = max(int(np.floor(fs / 1000.0 * P["smooth_ms"])), 1)
    sn += 1 - sn % 2
    keys = ["seg", "hp", "bl_A", "y_A", "bl_final", "y_final", "lr_A", "lr_B"]
    arr = {k: np.full(n, np.nan) for k in keys}
    out = dict(pkA=[], pkB_raw=[], pk_final=[], troughs=[], cand_A=[], cand_B=[], cycle_means_A=[],
               cycle_means_final=[], rows_A=[], rows_B=[])
    for s0, s1 in segments:
        seg = D["running_mean"](x[s0:s1], sn)
        hp = D["_highpass"](seg, fs, P["scale_highpass_hz"])
        pkA, rowsA, lrA = D["select_peaks"](hp, fs, noise_ptp)
        blA = D["_cycle_mean_baseline"](seg, pkA)
        y = seg - blA
        pk, rej, lr = D["select_peaks"](y, fs, noise_ptp, scale_signal=hp)
        pk_raw = pk.copy()
        rej_before = len(rej)
        need_at = lambda i: max(PP["prominence_frac"] * lr[i], PP["noise_mult"] * noise_ptp)  # noqa: E731
        pk = D["_validate_cycles"](y, pk, need_at, rej)
        blF = blA
        yF = y
        if len(pk) >= 2:
            blF = D["_cycle_mean_baseline"](seg, pk)
            yF = seg - blF
        tr = D["_troughs_between"](yF, pk) if len(pk) > 1 else np.array([], int)
        for k, v in zip(keys, [seg, hp, blA, y, blF, yF, lrA, lr]):
            arr[k][s0:s1] = v
        out["pkA"].append(pkA + s0)
        out["pkB_raw"].append(pk_raw + s0)
        out["pk_final"].append(pk + s0)
        out["troughs"].append(tr + s0)
        for a, b, cm in _cycle_means(seg, pkA):
            out["cycle_means_A"].append((a + s0, b + s0, cm))
        for a, b, cm in _cycle_means(seg, pk):
            out["cycle_means_final"].append((a + s0, b + s0, cm))
        out["rows_A"] += [dict(r, idx=r["idx"] + s0) for r in rowsA]
        out["rows_B"] += [dict(r, idx=r["idx"] + s0, stage="validate" if i >= rej_before else "select")
                          for i, r in enumerate(rej)]
    cat = lambda v: np.concatenate(v).astype(int) if len(v) else np.array([], int)  # noqa: E731
    for k in ["pkA", "pkB_raw", "pk_final", "troughs"]:
        out[k] = cat(out[k])
    out.update(arr)
    out.update(valid=valid, reason=reason, low_amp=low_amp, segments=segments, noise_ptp=noise_ptp,
               blocks=blocks, x=x, fs=fs)
    ref = D["detect_breaths"](signal, fs)
    assert np.array_equal(ref["peaks"], out["pk_final"]), "replay diverged from detect_breaths (peaks)"
    assert np.array_equal(ref["troughs"], out["troughs"]), "replay diverged from detect_breaths (troughs)"
    assert np.allclose(ref["baseline"], out["bl_final"], equal_nan=True), "replay diverged (baseline)"
    out["ref"] = ref
    return out


def _cycle_means(seg, pk):
    if len(pk) < 2:
        return []
    c = np.concatenate([[0.0], np.cumsum(seg)])
    return [(int(a), int(b), float((c[b] - c[a]) / (b - a))) for a, b in zip(pk[:-1], pk[1:])]


def aim1_signal(key):
    """Cached 400 Hz Aim1 signal (same loaders as the pipelines)."""
    recs = wb.all_recordings()
    return wb.load_signal(key, recs[key])


def cohort3_signal(key):
    return np.asarray(wb._cohort3_signal(key), float)
