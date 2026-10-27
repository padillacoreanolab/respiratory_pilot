"""Support code for peak_detection_workbench.ipynb: data loading, window selection,
snapshots, plotting and summaries. NOTHING here detects peaks - the notebook passes its
own detect_breaths() in, so editing the notebook's detector cells changes every result.

Data: the same 400 Hz signals the pipelines use (rank: load_rank_resp_signal_interactions;
valence: load_clean_resp_signal), cached locally in workbench_cache/ after the first load.
"""
import glob
import json
import os
import re
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "workbench_cache")
SNAPS = os.path.join(HERE, "workbench_snapshots")
os.environ.setdefault("BREATH_VALIDATION_CACHE", CACHE)
sys.path[:0] = [HERE, os.path.dirname(HERE)]
from real_common import FS, all_recordings, load_signal  # noqa: E402

# documented drops (docs/EXCLUSIONS_LOG.md): broken / missing respiration
EXCLUDE = {"RI1_5_13", "RI2_5_13", "BLRI_5_13"}
WINDOW_S = 4.0
N_WINDOWS = 40
MATCH_TOL_S = 0.025

# ----------------------------------------------------------------------------- cohorts
# group -> (long label for headings / figure titles, short label for panel titles)
# Overall numbering, by recording date (filenames) - the Dropbox folder names use BLA-only numbers:
#   cohort 1 = Aim1, Jun 2025                 (Dropbox 2025/ECG_cohort1)
#   cohort 2 = BLA pilot, Jan 2026             (Dropbox 2025/BLA_ChR_resp_pilot1 (SS) = "BLA cohort 1")
#   cohort 3 = BLA cohort 2, Jul 2026          (Dropbox 2026/BLA_ChR_resp_cohort2)
GROUPS = {
    "aim1_cagemate": ("Cohort 1 - Aim1 (Jun 2025): cagemate / rank", "C1 Aim1 cagemate"),
    "aim1_blri": ("Cohort 1 - Aim1 (Jun 2025): valence, BLRI baseline", "C1 Aim1 BLRI"),
    "aim1_ri1": ("Cohort 1 - Aim1 (Jun 2025): valence, RI1 positive", "C1 Aim1 RI1"),
    "aim1_ri2": ("Cohort 1 - Aim1 (Jun 2025): valence, RI2 negative", "C1 Aim1 RI2"),
    "c2_cm": ("Cohort 2 - BLA pilot (Jan 2026): cagemate", "C2 BLA pilot cagemate"),
    "c2_ri1_d1": ("Cohort 2 - BLA pilot (Jan 2026): RI1 day 1", "C2 BLA pilot RI1 d1"),
    "c2_ri1_d2": ("Cohort 2 - BLA pilot (Jan 2026): RI1 day 2", "C2 BLA pilot RI1 d2"),
    "c2_ri2_d1": ("Cohort 2 - BLA pilot (Jan 2026): RI2 post day 1", "C2 BLA pilot RI2 d1"),
    "c2_ri2_d2": ("Cohort 2 - BLA pilot (Jan 2026): RI2 post day 2", "C2 BLA pilot RI2 d2"),
    "c3_cm": ("Cohort 3 (Jul 2026): cagemate", "C3 cagemate"),
    "c3_ri1": ("Cohort 3 (Jul 2026): RI1 days 1-2", "C3 RI1"),
}
_STYPE_GROUP = {(2, "CM"): "c2_cm", (2, "RI1 day1"): "c2_ri1_d1", (2, "RI1 day2"): "c2_ri1_d2",
                (2, "RI2 post d1"): "c2_ri2_d1", (2, "RI2 post d2"): "c2_ri2_d2",
                (3, "CM"): "c3_cm", (3, "RI1 day1-2"): "c3_ri1"}


def group_of(key):
    """Recording key -> GROUPS key. Keys are the Aim1 workbench keys (CM_..., BLRI_..., RI1_...,
    RI2_...), inventory keys "c<cohort>|<session type>|<file>", or raw cohort-3 session names
    (the Section 1 sessions, e.g. 11_3_s_CM_11_4_d_...)."""
    if "|" in key:
        c, st, f = key.split("|", 2)
        if c == "c1":                          # cohort 1: same groups as the Aim1 workbench keys
            return {"CM": "aim1_cagemate", "BLRI": "aim1_blri", "RI1": "aim1_ri1", "RI2": "aim1_ri2"}[f.split("_")[0]]
        return _STYPE_GROUP[(int(c[1:]), st)]
    for prefix, g in (("CM_", "aim1_cagemate"), ("BLRI_", "aim1_blri"), ("RI1_", "aim1_ri1"), ("RI2_", "aim1_ri2")):
        if key.startswith(prefix):
            return g
    return "c3_cm"


def display_name(key):
    return key.split("|")[-1]


def status_of(key):
    """Whether the current analyses use this recording (docs/EXCLUSIONS_LOG.md)."""
    if "|" in key:
        c, st, f = key.split("|", 2)
        return _status(int(c[1:]), st, f)
    if key.startswith("CM_"):
        return "USED (rank)"
    if key.split("_")[0] in ("BLRI", "RI1", "RI2"):
        return "USED (valence)"
    return _status(3, "CM", key)


def cohort_label(key, short=False):
    return GROUPS[group_of(key)][1 if short else 0]


# ----------------------------------------------------------------------------- data
def recordings():
    """{key: (signal, time)} for every usable rank + valence recording (cached)."""
    os.makedirs(CACHE, exist_ok=True)
    out = {}
    for key, rec in all_recordings().items():
        if key in EXCLUDE:
            continue
        out[key] = load_signal(key, rec)
    return out


# ----------------------------------------------------------------------------- bad signal
# FROZEN rules (independent of the editable detector, so the same data is excluded and the
# same windows are shown after any edit). Evaluated on 1 s blocks, 0.25 s hop, using the
# 5-95% range relative to the recording median ("rel"):
DEAD_REL = 0.10          # rel < 0.10 -> dead / dropout   (rule of the 2026-09-30 audit)
ARTIFACT_REL = 8.0       # rel > 8    -> gross artifact
SPIKE_FULL_REL = 5.0     # FULL block range (max-min) > 5x the median 5-95% range -> brief spike
                         # (a ~50 ms spike is invisible to the 5-95% range; 99.9th pct of all blocks = 5.3)
DEGRADED_REL = 0.50      # 5 s running median of rel < 0.50 -> sustained degraded / decoupled signal
BAD_BUFFER_S = 10.0      # exclude this much on each side of dead / artifact stretches
# Why the buffer: around dropouts the signal fades in and out (e.g. CM_1_2 ~277-307 s,
# RI2_2_4 ~281-310 s), and the flattest parts are only the core of the failure.
# Thresholds were set by inspecting the 40 plotted windows (2026-10-09): 0.35 / no spike rule
# left RI2_2_4 287-291 s (degraded, ~47% amplitude, leads into a dropout) and a ~10000-count
# spike in RI1_2_3 at 151.8 s in the plots. Cost: ~10% of data excluded (RI2_2_4 ~50%).


def bad_signal_mask(sig, fs=FS):
    """Returns (exclude, reason): bool per sample, and 'dead' / 'artifact' (incl. spikes) / 'degraded' /
    'buffer' / '' per sample. Excluded samples are left out of plots AND summaries."""
    from scipy.ndimage import binary_dilation, median_filter
    n, w, h = len(sig), int(fs), int(0.25 * fs)
    starts = np.arange(0, n - w + 1, h)
    r = np.array([np.subtract(*np.percentile(sig[s:s + w], [95, 5])) for s in starts])
    med = np.median(r)
    rel = r / med
    full = np.array([np.ptp(sig[s:s + w]) for s in starts]) / med
    relm = median_filter(rel, size=int(5 / 0.25) + 1)
    dead, art, deg = np.zeros(n, bool), np.zeros(n, bool), np.zeros(n, bool)
    for s, v, vm, f in zip(starts, rel, relm, full):
        dead[s:s + w] |= v < DEAD_REL
        art[s:s + w] |= (v > ARTIFACT_REL) | (f > SPIKE_FULL_REL)
        deg[s:s + w] |= vm < DEGRADED_REL
    core = dead | art
    buf = binary_dilation(core, structure=np.ones(2 * int(BAD_BUFFER_S * fs) + 1, bool)) if core.any() else core
    reason = np.full(n, "", dtype=object)
    reason[buf] = "buffer"
    reason[deg] = "degraded"
    reason[art] = "artifact"
    reason[dead] = "dead"
    return buf | deg, reason


def bad_intervals(recs, fs=FS):
    """Table of excluded stretches (written to workbench_bad_intervals.csv)."""
    rows = []
    for key, (sig, t) in recs.items():
        ex, reason = bad_signal_mask(sig, fs)
        m = np.r_[False, ex, False]
        d = np.diff(m.astype(int))
        for a, b in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
            rs = pd.Series(reason[a:b]).value_counts()
            rows.append(dict(key=key, start_s=a / fs, stop_s=b / fs, dur_s=(b - a) / fs,
                             contains=", ".join(f"{k} {v / fs:.1f}s" for k, v in rs.items())))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "workbench_bad_intervals.csv"), index=False)
    return df


def select_windows(recs, n=N_WINDOWS, win_s=WINDOW_S, fs=FS):
    """n windows spread evenly over recordings (round robin) and over time within each
    recording, using only good signal (bad_signal_mask). Deterministic; saved to
    workbench_windows.csv."""
    path = os.path.join(HERE, "workbench_windows.csv")
    keys = sorted(recs, key=lambda k: (not k.startswith("CM_"), k))   # rank first, then valence
    R = len(keys)
    k_per = np.full(R, n // R)
    extra = n - k_per.sum()
    k_per[np.round(np.linspace(0, R - 1, extra)).astype(int)] += 1 if extra else 0
    rows = []
    W = int(win_s * fs)
    for key, k in zip(keys, k_per):
        sig, t = recs[key]
        good = ~bad_signal_mask(sig, fs)[0]
        cand = np.array([s for s in range(0, len(sig) - W, int(0.5 * fs)) if good[s:s + W].all()])
        if not len(cand) or k == 0:
            continue
        for j in range(k):
            target = (j + 0.5) / k * (len(sig) - W)
            s = int(cand[np.argmin(np.abs(cand - target))])
            rows.append(dict(key=key, start_s=s / fs, stop_s=(s + W) / fs))
    win = pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)
    win.insert(0, "panel", np.arange(1, len(win) + 1))
    win.to_csv(path, index=False)
    return win


# ----------------------------------------------------------------------------- running
def run_all(detect_fn, recs=None):
    """Run the notebook's detect_breaths on every whole recording (detection needs context)."""
    recs = recs or recordings()
    out = {}
    for key, (sig, t) in recs.items():
        out[key] = detect_fn(sig, FS)
        out[key]["signal"] = sig
        out[key]["excluded"] = bad_signal_mask(sig, FS)[0]   # frozen bad-signal screen
    return out


def _match(a, b, tol):
    a, b = np.asarray(a, float), np.sort(np.asarray(b, float))
    if not len(a) or not len(b):
        return np.zeros(len(a), bool)
    i = np.clip(np.searchsorted(b, a), 1, len(b) - 1)
    return np.minimum(np.abs(b[i] - a), np.abs(b[i - 1] - a)) <= tol


# ----------------------------------------------------------------------------- snapshots
def save_snapshot(results, name, params=None, overwrite=False):
    os.makedirs(SNAPS, exist_ok=True)
    p = os.path.join(SNAPS, f"{name}.npz")
    if os.path.exists(p) and not overwrite:
        print(f"snapshot '{name}' exists - pass overwrite=True to replace it")
        return
    np.savez_compressed(p, **{f"peaks__{k}": r["peaks"] for k, r in results.items()})
    if params is not None:
        json.dump(params, open(os.path.join(SNAPS, f"{name}_params.json"), "w"), indent=1, default=str)
    print(f"saved snapshot '{name}' ({len(results)} recordings)")


def load_snapshot(name):
    p = os.path.join(SNAPS, f"{name}.npz")
    if not os.path.exists(p):
        return None
    d = np.load(p)
    return {k[len("peaks__"):]: d[k] for k in d.files}


def list_snapshots():
    return sorted(f[:-4] for f in os.listdir(SNAPS) if f.endswith(".npz")) if os.path.isdir(SNAPS) else []


def legacy_peaks(key):
    """Legacy BreathMetrics peaks saved by compare_real.py (comparison only; not computed here)."""
    p = os.path.join(HERE, "outputs_v1", "recordings", f"{key}.npz")
    return np.load(p)["peaks_l"] if os.path.exists(p) else None


# ----------------------------------------------------------------------------- plots / tables
def plot_windows(results, windows, compare_to="v1", show_legacy=False, show_near_misses=True,
                 per_figure=10, fs=FS):
    snap = load_snapshot(compare_to) if compare_to else None
    tol = int(MATCH_TOL_S * fs)
    rows = []
    for f0 in range(0, len(windows), per_figure):
        chunk = windows.iloc[f0:f0 + per_figure]
        fig, axes = plt.subplots(len(chunk), 1, figsize=(17, 2.5 * len(chunk)), squeeze=False)
        for ax, w in zip(axes[:, 0], chunk.itertuples()):
            r = results[w.key]
            s0, s1 = int(w.start_s * fs), int(w.stop_s * fs)
            tt = np.arange(s0, s1) / fs
            y = r["smoothed"][s0:s1]
            ax.plot(tt, y if np.isfinite(y).any() else r["signal"][s0:s1], color="#167fa0", lw=1)
            ax.plot(tt, r["baseline"][s0:s1], color="0.55", ls="--", lw=0.8)
            inv = ~r["valid"][s0:s1]
            if inv.any():
                ax.fill_between(tt, np.nanmin(y), np.nanmax(y), where=inv, color="0.8", alpha=.6, step="mid")
            yv = lambda p: r["smoothed"][p] if np.isfinite(r["smoothed"][p]) else r["signal"][p]  # noqa: E731
            pk = r["peaks"][(r["peaks"] >= s0) & (r["peaks"] < s1)]
            tr = r["troughs"][(r["troughs"] >= s0) & (r["troughs"] < s1)]
            ax.plot(pk / fs, [yv(p) for p in pk], "^", color="#1f3fbf", ms=6)
            ax.plot(tr / fs, [yv(p) for p in tr], "v", color="#8a63d2", ms=4)
            added = removed = np.array([], int)
            if snap is not None and w.key in snap:
                sp = snap[w.key]
                sp = sp[(sp >= s0) & (sp < s1)]
                added = pk[~_match(pk, sp, tol)]
                removed = sp[~_match(sp, pk, tol)]
                ax.plot(added / fs, [yv(p) for p in added], "o", mfc="none", mec="limegreen", ms=15, mew=2.5)
                ax.plot(removed / fs, [yv(int(p)) for p in removed], "x", color="red", ms=13, mew=3)
            nm = 0
            if show_near_misses and len(r["rejected"]):
                rj = r["rejected"]
                rj = rj[(rj.idx >= s0) & (rj.idx < s1) & (rj.prominence >= 0.5 * rj.required)]
                nm = len(rj)
                ax.plot(rj.idx / fs, [yv(int(p)) for p in rj.idx], ".", color="darkorange", ms=7)
            if show_legacy:
                lp = legacy_peaks(w.key)
                if lp is not None:
                    lp = lp[(lp >= s0) & (lp < s1)]
                    ax.plot(lp / fs, [yv(p) for p in lp], "o", mfc="none", mec="0.35", ms=10, mew=1)
            ax.set_xlim(s0 / fs, s1 / fs)
            ax.set_title(f"#{w.panel}  [{cohort_label(w.key, short=True)}]  {w.key}   "
                         f"{w.start_s:.1f}-{w.stop_s:.1f} s   |  peaks {len(pk)}"
                         + (f"  (vs '{compare_to}': +{len(added)} / -{len(removed)})" if snap is not None else "")
                         + (f"  |  near-misses {nm}" if show_near_misses else ""), fontsize=9, loc="left")
            ax.tick_params(labelsize=7)
            rows.append(dict(panel=w.panel, key=w.key, start_s=w.start_s, peaks=len(pk), added=len(added),
                             removed=len(removed), near_misses=nm))
        handles = [plt.Line2D([], [], color="#167fa0", label="signal (smoothed)"),
                   plt.Line2D([], [], color="0.55", ls="--", label="local baseline"),
                   plt.Line2D([], [], marker="^", ls="", color="#1f3fbf", label="peak"),
                   plt.Line2D([], [], marker="v", ls="", color="#8a63d2", label="trough")]
        if snap is not None:
            handles += [plt.Line2D([], [], marker="o", ls="", mfc="none", mec="limegreen", mew=2.5, ms=12,
                                   label=f"ADDED vs '{compare_to}'"),
                        plt.Line2D([], [], marker="x", ls="", color="red", mew=3, ms=10, label=f"REMOVED vs '{compare_to}'")]
        if show_near_misses:
            handles.append(plt.Line2D([], [], marker=".", ls="", color="darkorange", ms=10,
                                      label="near-miss (rejected, prominence >= 50% of required)"))
        if show_legacy:
            handles.append(plt.Line2D([], [], marker="o", ls="", mfc="none", mec="0.35", ms=10, label="legacy peak"))
        fig.legend(handles=handles, loc="upper center", ncol=len(handles), fontsize=8, frameon=False)
        fig.tight_layout(rect=(0, 0, 1, 1 - 0.25 / len(chunk)))
        plt.show()
    return pd.DataFrame(rows)


def summary(results, compare_to="v1", fs=FS):
    """Whole-recording effect of the current detector vs a snapshot (all valid signal)."""
    snap = load_snapshot(compare_to) if compare_to else None
    tol = int(MATCH_TOL_S * fs)
    rows = []
    for key, r in results.items():
        keep = ~r["excluded"]                       # bad / dead / degraded signal left out
        pk = r["peaks"][keep[r["peaks"]]]
        used = r["valid"] & keep
        row = dict(key=key, cohort=cohort_label(key, short=True), analysed_min=used.sum() / fs / 60, excluded_pct=100 * np.mean(~keep), peaks=len(pk),
                   rate_hz=len(pk) / max(used.sum() / fs, 1e-9),
                   detector_invalid_pct=100 * np.mean(~r["valid"] & keep))
        if snap is not None and key in snap:
            sp = snap[key]
            sp = sp[keep[sp]]
            row.update(snapshot_peaks=len(sp), added=int((~_match(pk, sp, tol)).sum()),
                       removed=int((~_match(sp, pk, tol)).sum()))
            row["change_pct"] = 100 * (len(pk) - len(sp)) / max(len(sp), 1)
        rows.append(row)
    df = pd.DataFrame(rows)
    tot = dict(key="TOTAL", analysed_min=df.analysed_min.sum(), peaks=df.peaks.sum(),
               rate_hz=df.peaks.sum() / (df.analysed_min.sum() * 60))
    if "snapshot_peaks" in df:
        tot.update(snapshot_peaks=df.snapshot_peaks.sum(), added=df.added.sum(), removed=df.removed.sum(),
                   change_pct=100 * (df.peaks.sum() - df.snapshot_peaks.sum()) / df.snapshot_peaks.sum())
    return pd.concat([df, pd.DataFrame([tot])], ignore_index=True).round(3)


def seed_cache_from(path):
    """Copy already-cached signals (e.g. from the validation run) to avoid re-reading raw H5s."""
    os.makedirs(CACHE, exist_ok=True)
    n = 0
    for f in os.listdir(path):
        if f.endswith(".npz") and not os.path.exists(os.path.join(CACHE, f)):
            shutil.copy(os.path.join(path, f), CACHE)
            n += 1
    return n


# ----------------------------------------------------------------------------- the original problem
# The windows from Sequoia's QC notebook (START_HERE_load_and_inspect_features.ipynb, cohort 3,
# 2026-07-29) that first showed the undercounting: Sequoia's screenshot was Sequoia's run of Thomas's legacy
# BreathMetrics peak-finder; her prominence-fix figure is Sequoia's own solution. Same sessions and
# time windows, re-made from the raw H5s with this pipeline's loader
# (load_rank_resp_signal_interactions: 20 kHz -> 400 Hz, 0.1-20 Hz).
# Time 0 = first respiration sample (v2 H5s have no duplicate samples, so seconds should match
# Sequoia's recording-clock seconds closely). Amplitude units differ from Sequoia's ADC-count plots.
COHORT3_DIR = ("C:/Users/thoma/UF Dropbox/Thomas Heeps/Padilla-Coreano Lab/2026/BLA_ChR_resp_cohort2/"
               "BLA_ChR_resp_CM_hc/h5_outputs")
ORIGINAL_PANELS = [
    ("11_3_s_CM_11_4_d_20260729_122040", 300.0, 301.5, "Sequoia's run of the legacy code found 4"),
    ("11_3_s_CM_11_4_d_20260729_122040", 550.0, 551.5, "Sequoia's run of the legacy code found 15"),
    ("11_3_s_CM_11_4_d_20260729_122040", 442.0, 443.5, "from Sequoia's prominence-fix figure"),
    ("10_3_s_CM_10_2_d_20260729_174121", 50.0, 51.5, "from Sequoia's prominence-fix figure"),
    ("11_2_s_CM_11_4_d_20260729_172306", 490.0, 491.5, "Sequoia's figure: masked signal loss"),
    ("11_2_s_CM_11_4_d_20260729_172306", 521.0, 522.5, "from Sequoia's prominence-fix figure"),
    # extra window (not in Sequoia's notebook): inside her masked 482-516 s signal loss
    ("11_2_s_CM_11_4_d_20260729_172306", 499.8, 501.3, "flaw demo: inside Sequoia's masked signal loss"),
]


def _cohort3_signal(key):
    p = os.path.join(CACHE, f"cohort3_{key}.npz")
    if os.path.exists(p):
        return np.load(p)["signal"]
    from resp_helper_functions import load_rank_resp_signal_interactions
    sig, t, fs, _ = load_rank_resp_signal_interactions(os.path.join(COHORT3_DIR, key + ".h5"), target_rate=FS)
    np.savez(p, signal=sig)
    return sig


def cohort3_section1_recordings():
    """{session: (signal, time)} for the cohort-3 sessions used in Section 1 (cached), so they can
    go through the same bad-signal report as the Aim1 recordings."""
    out = {}
    for key in sorted({k for k, *_ in ORIGINAL_PANELS}):
        sig = _cohort3_signal(key)
        out[key] = (sig, np.arange(len(sig)) / FS)
    return out


def _cohort3_legacy_peaks(key, sig):
    """Legacy BreathMetrics (the method that undercounted) - reference only, cached."""
    p = os.path.join(CACHE, f"cohort3_{key}_legacy.npz")
    if os.path.exists(p):
        return np.load(p)["peaks"]
    from breathmetrics import bmObject
    bm = bmObject(sig, FS, "rodentAirflow")
    bm.estimateAllFeatures()
    pk = np.asarray(bm.inhalePeaks, int)
    np.savez(p, peaks=pk)
    return pk


def plot_original_problem(detect_fn, fs=FS):
    """Legacy (green, numbered) vs YOUR detector (blue) on the windows that showed the problem."""
    keys = sorted({k for k, *_ in ORIGINAL_PANELS})
    sig = {k: _cohort3_signal(k) for k in keys}
    leg = {k: _cohort3_legacy_peaks(k, sig[k]) for k in keys}
    res = {k: detect_fn(sig[k], fs) for k in keys}
    fig, axes = plt.subplots(len(ORIGINAL_PANELS), 1, figsize=(17, 2.7 * len(ORIGINAL_PANELS)))
    rows = []
    for ax, (k, t0, t1, note) in zip(axes, ORIGINAL_PANELS):
        r, s0, s1 = res[k], int(t0 * fs), int(t1 * fs)
        tt = np.arange(s0, s1) / fs
        y = r["smoothed"][s0:s1]
        y = np.where(np.isfinite(y), y, sig[k][s0:s1])
        ax.plot(tt, y, color="#167fa0", lw=1)
        inv = ~r["valid"][s0:s1]
        if inv.any():
            ax.fill_between(tt, y.min(), y.max(), where=inv, color="0.8", alpha=.6, step="mid")
        yv = lambda p: r["smoothed"][p] if np.isfinite(r["smoothed"][p]) else sig[k][p]  # noqa: E731
        lp = leg[k][(leg[k] >= s0) & (leg[k] < s1)]
        ap = r["peaks"][(r["peaks"] >= s0) & (r["peaks"] < s1)]
        ax.plot(lp / fs, [yv(p) for p in lp], "o", mfc="none", mec="green", ms=13, mew=1.8)
        for i, p in enumerate(lp, 1):
            ax.annotate(str(i), (p / fs, yv(p)), xytext=(0, 9), textcoords="offset points", ha="center",
                        fontsize=8, color="green")
        ax.plot(ap / fs, [yv(p) for p in ap], "^", color="#1f3fbf", ms=6)
        ax.set_xlim(t0, t1)
        ax.set_title(f"[{cohort_label(k, short=True)}]  {k[:5]} {t0:.0f}-{t1:.1f} s  |  legacy {len(lp)} peaks (green, numbered)   vs   "
                     f"your detector {len(ap)} peaks (blue)   [{note}]", fontsize=9, loc="left")
        ax.tick_params(labelsize=7)
        rows.append(dict(session=k, start_s=t0, stop_s=t1, legacy_peaks=len(lp), your_peaks=len(ap),
                         invalid_frac=float(inv.mean()), note=note))
    fig.tight_layout()
    plt.show()
    return pd.DataFrame(rows)


def _cohort3_legacy_fit(key, sig):
    """Legacy BreathMetrics fit: peaks, its own input signal, and its session-wide cutoff
    (mean + std/2). Reference only, cached."""
    p = os.path.join(CACHE, f"cohort3_{key}_legacyfit.npz")
    if os.path.exists(p):
        d = np.load(p)
        return d["peaks"], d["x"], float(d["thr"])
    from breathmetrics import bmObject
    bm = bmObject(sig, FS, "rodentAirflow")
    bm.estimateAllFeatures()
    x = bm.baselineCorrectedRespiration
    thr = float(np.mean(x) + np.std(x) / 2)
    pk = np.asarray(bm.inhalePeaks, int)
    np.savez(p, peaks=pk, x=x, thr=thr)
    return pk, x, thr


_NEW_CACHE = {}


def plot_old_vs_new(detect_fn, i, fs=FS):
    """ONE window from ORIGINAL_PANELS: OLD (legacy) on top, NEW (your detector) below,
    same time axis. Returns a one-row summary."""
    key, t0, t1, note = ORIGINAL_PANELS[i]
    sig = _cohort3_signal(key)
    lp, lx, thr = _cohort3_legacy_fit(key, sig)
    ck = (key, id(detect_fn))
    if ck not in _NEW_CACHE:
        _NEW_CACHE.clear() if len(_NEW_CACHE) > 6 else None
        _NEW_CACHE[ck] = detect_fn(sig, fs)
    r = _NEW_CACHE[ck]
    s0, s1 = int(t0 * fs), int(t1 * fs)
    tt = np.arange(s0, s1) / fs
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(15, 5.6), sharex=True)
    # OLD
    y1 = lx[s0:s1]
    a1.plot(tt, y1, color="0.35", lw=1)
    p1 = lp[(lp >= s0) & (lp < s1)]
    a1.plot(p1 / fs, lx[p1], "o", mfc="none", mec="green", ms=12, mew=2)
    for j, p in enumerate(p1, 1):
        a1.annotate(str(j), (p / fs, lx[p]), xytext=(0, 8), textcoords="offset points", ha="center", color="green")
    lo, hi = np.nanmin(y1), np.nanmax(y1)
    pad = 0.15 * (hi - lo + 1e-9)
    if thr <= hi + pad:
        a1.axhline(thr, color="red", ls="--", lw=1.2)
        a1.text(t0, thr, f" old cutoff = {thr:.0f}", color="red", va="bottom", fontsize=9)
    else:
        a1.text(t0 + 0.01 * (t1 - t0), hi + 0.6 * pad,
                f"old cutoff = {thr:.0f} (above the top of this plot: no peak here can reach it)",
                color="red", fontsize=9, va="center")
    a1.set_ylim(lo - pad, hi + 1.2 * pad)
    a1.set_title(f"OLD: legacy BreathMetrics peak-finder (the code in Sequoia's screenshot): "
                 f"{len(p1)} peaks counted (green, numbered)", loc="left", fontsize=10)
    # NEW
    y2 = r["smoothed"][s0:s1]
    y2 = np.where(np.isfinite(y2), y2, sig[s0:s1])
    a2.plot(tt, y2, color="#167fa0", lw=1)
    inv = ~r["valid"][s0:s1]
    if inv.any():
        a2.fill_between(tt, np.nanmin(y2), np.nanmax(y2), where=inv, color="0.8", alpha=.7, step="mid")
        a2.text(t0 + 0.01 * (t1 - t0), np.nanmax(y2), "grey = marked invalid (no breaths detected inside)",
                fontsize=9, va="top")
    p2 = r["peaks"][(r["peaks"] >= s0) & (r["peaks"] < s1)]
    yv = lambda p: r["smoothed"][p] if np.isfinite(r["smoothed"][p]) else sig[p]  # noqa: E731
    a2.plot(p2 / fs, [yv(p) for p in p2], "^", color="#1f3fbf", ms=9)
    for j, p in enumerate(p2, 1):
        a2.annotate(str(j), (p / fs, yv(p)), xytext=(0, 8), textcoords="offset points", ha="center", color="#1f3fbf")
    a2.set_title(f"NEW: adaptive detector (cells A-C): {len(p2)} peaks counted (blue, numbered)",
                 loc="left", fontsize=10)
    a2.set_xlabel("recording time (s)")
    for a in (a1, a2):
        a.margins(y=0.25)
        a.tick_params(labelsize=8)
    fig.suptitle(f"{cohort_label(key)}   |   {key}   {t0:.1f}-{t1:.1f} s", fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    plt.show()
    return explain_window(key, s0, s1, r, lp, lx, thr, fs)



def explain_window(key, s0, s1, r, lp, lx, thr, fs=FS):
    """The numbers behind one OLD/NEW pair: prints both rules with this session's values and
    returns one row per candidate peak (OLD height vs cutoff, NEW prominence vs requirement)."""
    from IPython.display import display
    P = r.get("params", {})
    frac, mult = P.get("prominence_frac", np.nan), P.get("noise_mult", np.nan)
    m, sd = float(np.mean(lx)), float(np.std(lx))
    print(f"OLD rule  count if height > cutoff,  cutoff = mean + std/2 = {m:.1f} + {sd:.1f}/2 = {thr:.1f}"
          f"   (mean/std over the WHOLE {len(lx) / fs:.0f} s session)")
    print(f"NEW rule  count if prominence >= max({frac:.2f} x local range, {mult:g} x noise floor "
          f"{r['noise_ptp']:.1f} = {mult * r['noise_ptp']:.1f}),  half-width >= 15 ms,  >= 60 ms from a taller peak")
    b = r["blocks"]
    bw = b[(b.start_s >= s0 / fs - 0.75) & (b.start_s < s1 / fs)]
    flagged = bw[bw.decision != "ok"]
    if len(flagged) and len(flagged) < len(bw) + 1:
        print(f"Component A blocks in this window: rel = range / median range = {bw.rel.min():.3f}-{bw.rel.max():.3f}; "
              f"ACF {np.nanmin(bw.acf) if bw.acf.notna().any() else float('nan'):.2f}-"
              f"{np.nanmax(bw.acf) if bw.acf.notna().any() else float('nan'):.2f}; decisions: "
              + ", ".join(f"{k} x{v}" for k, v in bw.decision.value_counts().items()))
    c = r["candidates"]
    c = c[(c.idx >= s0) & (c.idx < s1)]
    if not len(c):
        b = r["blocks"]
        b = b[(b.start_s >= s0 / fs - 0.75) & (b.start_s < s1 / fs)]
        print("NEW: no candidates - Component A marked this stretch invalid, so no peaks are searched:")
        out = pd.DataFrame(dict(block_start_s=b.start_s.round(2), range=b.range.round(1),
                                median_range=b.median_range.round(1),
                                rel=[f"{x:.1f} / {y:.1f} = {x / y:.3f}" for x, y in zip(b.range, b.median_range)],
                                acf=b.acf.round(2), decision=b.decision))
        display(out.reset_index(drop=True))
        return None
    tol = int(MATCH_TOL_S * fs)
    old_counted = _match(c.idx.to_numpy(), lp, tol)
    rows = []
    for (_, row), oc in zip(c.iterrows(), old_counted):
        h = float(lx[int(row.idx)])
        above = h > thr
        old = "counted" if oc else ("NOT counted: above cutoff but dropped by consensus/alternation" if above
                                    else "NOT counted: below cutoff")
        bound = "range" if row.req_from_range >= row.req_from_noise else "noise"
        new = {"counted": "counted", "low_prominence": "NOT counted: prominence too small",
               "spike_width": "NOT counted: too narrow (spike)", "too_close": "NOT counted: < 60 ms from a taller peak",
               "shallow_trough": "NOT counted: trough to neighbour too shallow"}.get(row.reason, row.reason)
        rows.append({"t (s)": round(row.idx / fs, 3),
                     "OLD height": round(h, 1), "OLD: height > cutoff?": f"{h:.0f} > {thr:.0f}? {'yes' if above else 'no'}",
                     "OLD result": old,
                     "NEW prominence": round(row.prominence, 1),
                     "NEW required = max(frac x range, mult x noise)":
                         f"max({frac:.2f} x {row.local_range:.0f}, {mult * r['noise_ptp']:.1f}) = {row.required:.1f} [{bound}]",
                     "NEW: prom >= required?": f"{row.prominence:.0f} >= {row.required:.0f}? {'yes' if row.prominence >= row.required else 'no'}",
                     "width (ms)": round(1000 * row.width_s, 1) if np.isfinite(row.width_s) else np.nan,
                     "NEW result": new})
    t = pd.DataFrame(rows)
    big = t[(t["NEW result"] == "counted") | (t["NEW prominence"] >= 0.3 * c.required.values)]
    hidden = len(t) - len(big)
    if hidden:
        print(f"({hidden} tiny wiggles with prominence < 30% of the requirement not listed)")
    pd.set_option("display.max_colwidth", 80)
    return big.reset_index(drop=True)


# ----------------------------------------------------------------------------- bad-signal report
_REASON_COLOR = {"dead": "#d62728", "artifact": "#9467bd", "degraded": "#ff7f0e", "buffer": "#f2c9a0"}


def bad_signal_report(recs, group=None, fs=FS, review_pct=10.0, drop_pct=25.0, row_height=2.8, low_group=0.25,
                      plots_only=False, per_figure=12):
    """Per recording: excluded stretches (frozen bad_signal_mask) as a list, plus a whole-session plot
    with the stretches shaded. group: a GROUPS key to show only that cohort (None = all).
    Every recording in the group is plotted, clean ones included. Sorted by % excluded.
    vs_group = the recording's median 1 s range / the median of that over the group; below
    low_group flags a recording whose WHOLE session is low (the within-recording rules can't see it).
    plots_only: skip the printed list (returns None). per_figure: recordings per figure, so a large
    group is drawn as several smaller images instead of one very tall one."""
    if group is not None:
        recs = {k: v for k, v in recs.items() if group_of(k) == group}
        if not plots_only:
            print(f"{GROUPS[group][0]}: {len(recs)} recordings\n")
    rows = []
    masks = {}
    for key, (sig, t) in recs.items():
        ex, reason = bad_signal_mask(sig, fs)
        masks[key] = (ex, reason)
        n1 = len(sig) // fs
        blocks = np.asarray(sig[: n1 * fs], float).reshape(n1, fs)
        rows.append(dict(key=key, recording=display_name(key), cohort=cohort_label(key, short=True),
                         status=status_of(key), minutes=len(sig) / fs / 60,
                         median_1s_range=float(np.median(np.percentile(blocks, 95, 1) - np.percentile(blocks, 5, 1))),
                         excluded_pct=100 * ex.mean(),
                         dead_s=(reason == "dead").sum() / fs, artifact_s=(reason == "artifact").sum() / fs,
                         degraded_s=(reason == "degraded").sum() / fs, buffer_s=(reason == "buffer").sum() / fs))
    df = pd.DataFrame(rows).sort_values("excluded_pct", ascending=False).reset_index(drop=True)
    df["flag"] = np.where(df.excluded_pct >= drop_pct, "CONSIDER DROPPING",
                          np.where(df.excluded_pct >= review_pct, "review", "ok"))
    df["vs_group"] = df.median_1s_range / df.median_1s_range.median()
    low = df.vs_group < low_group
    df.loc[low, "flag"] = [f"{f}, WHOLE-SESSION LOW ({v:.2f}x group)".replace("ok, ", "")
                           for f, v in zip(df.flag[low], df.vs_group[low])]
    if not plots_only:
        print(f"Excluded stretches per recording (flag: >= {drop_pct:.0f}% consider dropping, "
              f">= {review_pct:.0f}% review)\n")
    for row in ([] if plots_only else df.itertuples()):
        ex, reason = masks[row.key]
        m = np.r_[False, ex, False]
        d = np.diff(m.astype(int))
        parts = []
        for a, b in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
            core = pd.Series(reason[a:b])
            core = core[core != "buffer"].value_counts()
            what = "+".join(core.index) if len(core) else "buffer"
            parts.append(f"{a / fs:.0f}-{b / fs:.0f} s ({what})")
        line = (f"- {row.recording:44s} {row.excluded_pct:5.1f}% excluded "
                f"({row.excluded_pct * row.minutes * 60 / 100 / 60:.1f} of {row.minutes:.1f} min)  [{row.flag}]  [{row.status}]")
        print(line)
        if parts:
            print("      " + "; ".join(parts))
    n_fig = -(-len(df) // per_figure)
    for f0 in range(0, len(df), per_figure):
        show = df.iloc[f0:f0 + per_figure]
        fig, axes = plt.subplots(len(show), 1, figsize=(18, row_height * len(show)), squeeze=False)
        for ax, row in zip(axes[:, 0], show.itertuples()):
            sig = recs[row.key][0]
            ex, reason = masks[row.key]
            h = int(0.25 * fs)
            nb = len(sig) // h
            blk = sig[: nb * h].reshape(nb, h)
            tt = (np.arange(nb) + 0.5) * h / fs
            ax.fill_between(tt, blk.min(1), blk.max(1), color="#167fa0", lw=0)
            lo, hi = np.percentile(blk.min(1), 0.5), np.percentile(blk.max(1), 99.5)
            for rname, col in _REASON_COLOR.items():
                mm = reason == rname
                if mm.any():
                    ax.fill_between(np.arange(len(sig)) / fs, lo, hi, where=mm, color=col, alpha=0.45, lw=0, step="mid")
            ax.set_ylim(lo, hi)
            ax.set_xlim(0, len(sig) / fs)
            ax.set_yticks([])
            ax.set_title(f"[{row.cohort}]  {row.recording}: {row.excluded_pct:.1f}% excluded  [{row.flag}]  [{row.status}]",
                         loc="left", fontsize=12)
            ax.tick_params(labelsize=10)
        axes[-1, 0].set_xlabel("recording time (s)", fontsize=11)
        handles = [plt.Line2D([], [], color=c, lw=8, alpha=0.5, label=n) for n, c in _REASON_COLOR.items()]
        top = 1 - 0.55 / (row_height * len(show))
        fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=11, frameon=False,
                   bbox_to_anchor=(0.5, 1 - 0.25 / (row_height * len(show))))
        if group is not None:
            part = f"   (part {f0 // per_figure + 1} of {n_fig})" if n_fig > 1 else ""
            fig.suptitle(GROUPS[group][0] + part, fontsize=14, x=0.01, ha="left",
                         y=1 - 0.05 / (row_height * len(show)))
        fig.tight_layout(rect=(0, 0, 1, top))
        plt.show()
        plt.close(fig)
    return None if plots_only else df.drop(columns="key").round(2)


def group_recordings(group):
    """{inventory key: (signal, None)} for every recording of a cohort-2 / cohort-3 group.
    Only this group is loaded (cached as float32 at 400 Hz after the first load); unreadable
    files are skipped with a message."""
    os.makedirs(CACHE, exist_ok=True)
    inv, _ = cohort_inventory()
    out = {}
    for key, path, c, st in zip(inv.key, inv.path, inv.cohort, inv.session_type):
        if _STYPE_GROUP.get((int(c), st)) != group:      # cohort-1 rows have no entry here
            continue
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", key)
        p = os.path.join(CACHE, f"all_{safe}.npz")
        if not os.path.exists(p):
            res = _quality_one((key, path))          # loads the raw H5 and writes the cache
            if res.get("error"):
                print(f"skipped {display_name(key)}: {res['error']}")
                continue
        out[key] = (np.load(p)["signal"].astype(float), None)
    return out



# ============================================================================= cohorts
# Overall cohort numbering (evidence: recording dates in filenames, Dropbox folders, and
# docs/EXCLUSIONS_LOG.md). NOTE the folder names use BLA-only numbering: "BLA_ChR_resp_pilot1"
# is BLA cohort 1 (= overall cohort 2) and "BLA_ChR_resp_cohort2" is BLA cohort 2 (= overall cohort 3).
LAB = "C:/Users/thoma/UF Dropbox/Thomas Heeps/Padilla-Coreano Lab/"
COHORTS = {
    1: dict(name="Cohort 1 - Aim1 (Dropbox 2025/ECG_cohort1)", recorded="2025-06-21 to 06-23"),
    2: dict(name="Cohort 2 - BLA pilot (Dropbox 2025/BLA_ChR_resp_pilot1 (SS) = BLA cohort 1)",
            recorded="2026-01-06 to 01-20"),
    3: dict(name="Cohort 3 (Jul 2026; Dropbox 2026/BLA_ChR_resp_cohort2)", recorded="2026-07-11 to 07-29"),
}
COHORT_FOLDERS = [   # (cohort, session type, h5 folder, raw .rec folder or None)
    (1, "CM", "2025/ECG_cohort1/Aim1/AIM1/Day1_new/CM interactions/cm_h5/baseline_cagemate_interactions_h5_outputs", None),
    (1, "valence (BLRI/RI1/RI2)", "2025/ECG_cohort1/Aim1/AIM1/Day1_new/Resp_h5", None),
    (2, "CM", "2025/BLA_ChR_resp_pilot1 (SS)/BLA_resp_ChR_cm_hc/h5_outputs", "2025/BLA_ChR_resp_pilot1 (SS)/BLA_resp_ChR_cm_hc"),
    (2, "RI1 day1", "2025/BLA_ChR_resp_pilot1 (SS)/BLA_ChR_resp_RI1_hc_day1/h5_outputs", "2025/BLA_ChR_resp_pilot1 (SS)/BLA_ChR_resp_RI1_hc_day1"),
    (2, "RI1 day2", "2025/BLA_ChR_resp_pilot1 (SS)/BLA_ChR_resp_RI1_hc_day2/h5_outputs", "2025/BLA_ChR_resp_pilot1 (SS)/BLA_ChR_resp_RI1_hc_day2"),
    (2, "RI2 post d1", "2025/BLA_ChR_resp_pilot1 (SS)/BLA_ChR_resp_RI2_hc_post_d1/h5_outputs", "2025/BLA_ChR_resp_pilot1 (SS)/BLA_ChR_resp_RI2_hc_post_d1"),
    (2, "RI2 post d2", "2025/BLA_ChR_resp_pilot1 (SS)/BLA_ChR_resp_RI2_hc_post_d2/h5_outputs", "2025/BLA_ChR_resp_pilot1 (SS)/BLA_ChR_resp_RI2_hc_post_d2"),
    (3, "CM", "2026/BLA_ChR_resp_cohort2/BLA_ChR_resp_CM_hc/h5_outputs", "2026/BLA_ChR_resp_cohort2/BLA_ChR_resp_CM_hc"),
    (3, "RI1 day1-2", "2026/BLA_ChR_resp_cohort2/BLA_ChR_resp_RI1_hc_day1_2/h5_outputs", "2026/BLA_ChR_resp_cohort2/BLA_ChR_resp_RI1_hc_day1_2"),
]
NOT_CHECKED = [   # respiration folders not used by the rank / valence pipelines
    (2, "OFT", "2025/BLA_ChR_resp_pilot1 (SS)/BLA_ChR_resp_OFT/BLA_OFT_h5_outputs"),
    (3, "OFT", "2026/BLA_ChR_resp_cohort2/OFT"),
]
# recordings used by the current analyses / documented drops (docs/EXCLUSIONS_LOG.md)
_RANK_USED = {"CM_s1_1_d1_2_20250623_111352", "CM_s1_2_sub1_1_20250623_133932", "CM_s2_3_d2_4_20250623_151153",
              "CM_s2_4_sub2_3_20250623_143348", "CM_s3_5_d3_6_20250623_160001", "CM_s4_7_d4_8_20250623_193718",
              "CM_s4_8_sub4_7_20250623_182649"}
_RANK_DROPS = {"CM_s3_6_sub3_5_20250623_174348": "documented drop: empty BORIS (subject 3_6)",
               "CM_s3_5_d3_6_20250623_170708": "documented drop: BORIS-less duplicate of 3_5"}
QUALITY_VERSION = f"dead{DEAD_REL}_art{ARTIFACT_REL}_spk{SPIKE_FULL_REL}_deg{DEGRADED_REL}_buf{BAD_BUFFER_S}"


def _stem(f):
    b = os.path.basename(f)
    for suf in (".rec.h5", ".h5", ".rec"):
        if b.endswith(suf):
            return b[: -len(suf)]
    return b


def _status(cohort, stype, stem):
    if cohort == 1 and stype == "CM":
        s = stem.replace("_merged", "")
        return "USED (rank)" if s in _RANK_USED else _RANK_DROPS.get(s, "not used")
    if cohort == 1:
        if "_s5_13_" in stem:
            return "documented drop: subject 5_13"
        return "USED (valence)"
    if cohort == 2:
        return "not used (BLA cohort excluded: no baseline, EXCLUSIONS_LOG)"
    return "not used yet (new cohort; Sequoia's QC)"


def cohort_inventory():
    """Every respiration H5 per cohort / session type, plus raw .rec recordings without an H5."""
    rows, missing = [], []
    for cohort, stype, h5dir, recdir in COHORT_FOLDERS:
        h5s = sorted(glob.glob(os.path.join(LAB, h5dir, "*.h5")))
        seen = {}
        for f in h5s:                       # same recording saved twice (x.h5 and x.rec.h5): keep one
            st = _stem(f)
            if st in seen:
                missing.append(dict(cohort=cohort, session_type=stype, raw_recording=os.path.basename(f),
                                    problem=f"duplicate H5 of {os.path.basename(seen[st])} (same recording); ignored"))
                continue
            seen[st] = f
        h5s = list(seen.values())
        for f in h5s:
            st = _stem(f)
            dt = re.search(r"(20\d{6})", st)
            rows.append(dict(cohort=cohort, session_type=stype, key=f"c{cohort}|{stype}|{st}", file=st,
                             date=dt.group(1) if dt else "", path=f, status=_status(cohort, stype, st)))
        if recdir:
            recs = {_stem(r) for r in glob.glob(os.path.join(LAB, recdir, "*.rec"))}
            have = {_stem(f) for f in h5s}
            for r in sorted(recs - have):
                missing.append(dict(cohort=cohort, session_type=stype, raw_recording=r,
                                    problem="raw .rec has no H5 of the same name"))
            for h in sorted(have - recs):
                missing.append(dict(cohort=cohort, session_type=stype, raw_recording=h,
                                    problem="H5 has no raw .rec of the same name"))
    for cohort, stype, d in NOT_CHECKED:
        n_h5 = len(glob.glob(os.path.join(LAB, d, "*.h5")))
        n_rec = len(glob.glob(os.path.join(LAB, d, "*.rec")))
        missing.append(dict(cohort=cohort, session_type=stype, raw_recording=f"{n_rec} .rec, {n_h5} .h5",
                            problem="not checked (not used by rank/valence pipelines)"))
    return pd.DataFrame(rows), pd.DataFrame(missing)


def _quality_one(args):
    """Worker: load (cached, float32), run the frozen bad-signal screen, summarise."""
    key, path = args
    import warnings
    warnings.filterwarnings("ignore")
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", key)
    p = os.path.join(CACHE, f"all_{safe}.npz")
    try:
        if os.path.exists(p):
            d = np.load(p)
            sig, ch = d["signal"].astype(float), str(d["channel"])
        else:
            import h5py
            from resp_helper_functions import load_rank_resp_signal_interactions
            sig, t, fs, meta = load_rank_resp_signal_interactions(path, target_rate=FS)
            if sig is None:
                return dict(key=key, error="could not load respiration (no resp/analog dataset)")
            ch = ""
            with h5py.File(path, "r") as h:
                if "analog" in h:
                    ch = next((k for k in h["analog"].keys() if "ECU_Ain1" in k), "")
            np.savez(p, signal=np.asarray(sig, np.float32), channel=ch)
            sig = np.asarray(sig, float)
        ex, reason = bad_signal_mask(sig, FS)
        m = np.r_[False, ex, False]
        dd = np.diff(m.astype(int))
        parts = []
        for a, b in zip(np.where(dd == 1)[0], np.where(dd == -1)[0]):
            core = pd.Series(reason[a:b])
            core = core[core != "buffer"].value_counts()
            parts.append(f"{a / FS:.0f}-{b / FS:.0f} s ({'+'.join(core.index) if len(core) else 'buffer'})")
        n1 = len(sig) // FS
        blocks = sig[: n1 * FS].reshape(n1, FS)
        med_range = float(np.median(np.percentile(blocks, 95, axis=1) - np.percentile(blocks, 5, axis=1)))
        ch_stem = ch.split(".analog")[0] if ch else ""
        return dict(key=key, minutes=len(sig) / FS / 60, excluded_pct=100 * ex.mean(),
                    dead_s=(reason == "dead").sum() / FS, artifact_s=(reason == "artifact").sum() / FS,
                    degraded_s=(reason == "degraded").sum() / FS, median_1s_range=med_range,
                    stretches="; ".join(parts), internal_channel=ch_stem, error="")
    except Exception as e:  # noqa: BLE001
        return dict(key=key, error=f"{type(e).__name__}: {e}")


def cohort_quality(max_workers=4):
    """Run the frozen bad-signal screen on every recording of every cohort (cached; the first
    run loads ~210 raw 20 kHz files and takes several minutes)."""
    from concurrent.futures import ProcessPoolExecutor
    os.makedirs(CACHE, exist_ok=True)
    inv, missing = cohort_inventory()
    qpath = os.path.join(CACHE, f"cohort_quality_{QUALITY_VERSION}.csv")
    done = pd.read_csv(qpath, keep_default_na=False) if os.path.exists(qpath) else pd.DataFrame(columns=["key"])
    todo = [(k, p) for k, p in zip(inv.key, inv.path) if k not in set(done.key)]
    if todo:
        print(f"screening {len(todo)} recordings (cached afterwards) ...")
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            new = list(ex.map(_quality_one, todo))
        done = pd.concat([done, pd.DataFrame(new)], ignore_index=True)
        done.to_csv(qpath, index=False)
    q = inv.merge(done, on="key", how="left")
    for col in ["minutes", "excluded_pct", "dead_s", "artifact_s", "degraded_s", "median_1s_range"]:
        q[col] = pd.to_numeric(q[col], errors="coerce")
    for col in ["error", "stretches", "internal_channel"]:
        q[col] = q[col].fillna("").astype(str)
    # whole-session amplitude vs the other recordings of the same cohort + session type
    q["vs_group_median"] = q.median_1s_range / q.groupby(["cohort", "session_type"]).median_1s_range.transform("median")
    q["name_mismatch"] = [bool(c) and c != f for c, f in zip(q.internal_channel, q.file)]
    return q, missing


def cohort_bad_signal_report(review_pct=10.0, drop_pct=25.0, low_group=0.25, plot=True):
    """Section 6: per cohort - inventory, missing recordings, and per recording the excluded
    stretches; flagged recordings are plotted whole with the stretches shaded."""
    q, missing = cohort_quality()

    def flag(r):
        if r.error:
            return "UNREADABLE"
        f = []
        if r.excluded_pct >= drop_pct:
            f.append("CONSIDER DROPPING")
        elif r.excluded_pct >= review_pct:
            f.append("review")
        if r.vs_group_median < low_group:
            f.append(f"WHOLE-SESSION LOW AMPLITUDE ({r.vs_group_median:.2f}x group median)")
        return ", ".join(f) if f else "ok"

    q["flag"] = [flag(r) for r in q.itertuples()]
    for c in sorted(COHORTS):
        g = q[q.cohort == c]
        print("=" * 110)
        print(f"{COHORTS[c]['name']}   recorded {COHORTS[c]['recorded']}")
        print("   " + ", ".join(f"{t}: {n}" for t, n in g.groupby("session_type", sort=False).size().items())
              + f"   (total {len(g)} respiration H5)")
        for r in missing[missing.cohort == c].itertuples():
            print(f"   ! {r.session_type}: {r.raw_recording} - {r.problem}")
        nm = g[g.name_mismatch]
        if len(nm):
            print(f"   ! {len(nm)} H5 whose internal channel name differs from the file name, e.g. "
                  f"{nm.file.iloc[0]} (file) vs {nm.internal_channel.iloc[0]} (channel)")
        for stype, gg in g.groupby("session_type", sort=False):
            print(f"\n   -- {stype} ({len(gg)})")
            for r in gg.sort_values("excluded_pct", ascending=False).itertuples():
                if r.error:
                    print(f"   - {r.file:44s} UNREADABLE: {r.error}   [{r.status}]")
                    continue
                print(f"   - {r.file:44s} {r.excluded_pct:5.1f}% excluded of {r.minutes:4.1f} min  "
                      f"[{r.flag}]  [{r.status}]")
                if r.stretches:
                    print(f"        {r.stretches}")
        if plot:
            show = g[(g.flag != "ok") & (g.error == "")].sort_values("excluded_pct", ascending=False)
            if len(show):
                fig, axes = plt.subplots(len(show), 1, figsize=(17, 1.35 * len(show) + 0.6), squeeze=False)
                for ax, r in zip(axes[:, 0], show.itertuples()):
                    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", r.key)
                    sig = np.load(os.path.join(CACHE, f"all_{safe}.npz"))["signal"].astype(float)
                    ex, reason = bad_signal_mask(sig, FS)
                    h = int(0.25 * FS)
                    nb = len(sig) // h
                    blk = sig[: nb * h].reshape(nb, h)
                    tt = (np.arange(nb) + 0.5) * h / FS
                    ax.fill_between(tt, blk.min(1), blk.max(1), color="#167fa0", lw=0)
                    lo, hi = np.percentile(blk.min(1), 0.5), np.percentile(blk.max(1), 99.5)
                    for rname, col in _REASON_COLOR.items():
                        mm2 = reason == rname
                        if mm2.any():
                            ax.fill_between(np.arange(len(sig)) / FS, lo, hi, where=mm2, color=col, alpha=0.45,
                                            lw=0, step="mid")
                    ax.set_ylim(lo, hi)
                    ax.set_xlim(0, len(sig) / FS)
                    ax.set_yticks([])
                    ax.set_title(f"[{r.session_type}] {r.file}: {r.excluded_pct:.1f}% excluded  [{r.flag}]  "
                                 f"[{r.status}]", loc="left", fontsize=8.5)
                    ax.tick_params(labelsize=7)
                axes[-1, 0].set_xlabel("recording time (s)")
                handles = [plt.Line2D([], [], color=col, lw=8, alpha=0.5, label=n) for n, col in _REASON_COLOR.items()]
                fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=9, frameon=False)
                fig.suptitle(f"{COHORTS[c]['name']}: flagged recordings only", y=1.0, fontsize=11)
                fig.tight_layout(rect=(0, 0, 1, 1 - 0.35 / len(show)))
                plt.show()
    summary = q.groupby(["cohort", "session_type"], sort=False).agg(
        h5=("key", "size"), readable=("error", lambda e: int((e == "").sum())),
        mean_excluded_pct=("excluded_pct", "mean"),
        consider_dropping=("flag", lambda f: int(f.str.contains("DROPPING").sum())),
        review=("flag", lambda f: int(f.str.contains("review").sum())),
        whole_session_low=("flag", lambda f: int(f.str.contains("LOW AMPLITUDE").sum())),
        name_mismatch=("name_mismatch", "sum")).round(1)
    return summary, q
