"""Builds every figure for the 8-slide methods deck from real recordings.

Run in biopipeline-env from any folder:
    C:/Users/thoma/miniconda3/envs/biopipeline-env/python.exe make_figures.py

- Raw stages (slides 1-2) are read from the source 20 kHz H5 and recomputed with the exact
  steps of resp_helper_functions.preprocess_resp_to_target_rate; the final stage is checked
  against the cached 400 Hz workbench signal.
- Detector internals come from deck_common.trace_detector, which runs the notebook's own
  cells A-C and asserts its peaks/troughs/baseline equal detect_breaths().
- Frozen QC masks come from workbench_support.bad_signal_mask (unchanged).
Writes figures/*.png + *.svg, figures/manifest.txt and figures/facts.json.
"""
import json
import os
import sys

import h5py
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from scipy.signal import butter, filtfilt, find_peaks, resample_poly  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from deck_common import (AIM1_CM, AIM1_REC, D, FS, aim1_signal, cohort3_signal,  # noqa: E402
                         trace_detector, wb)

REPO = os.path.abspath(os.path.join(HERE, *[".."] * 5))
sys.path.insert(0, os.path.join(REPO, "src", "file_conversion"))
from ekg_resp_to_h5 import read_trodes_extracted_data_file  # noqa: E402  (reader only)
import neurokit2 as nk  # noqa: E402

OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)
DETECTOR_VERSION = "workbench cells A-C (= adaptive_breath_detection.py v1.0.0; snapshot v1 params)"
QC_VERSION = wb.QUALITY_VERSION

# ------------------------------------------------------------------ style (4 semantic colours)
SIG = "#262626"        # signal (dark neutral)
BLUE = "#1F5FA8"       # accepted / provisional peaks
ORANGE = "#E07B00"     # rejected candidates, troughs
RED = "#D9534F"        # invalid / excluded core (translucent)
GRAY = "#9A9A9A"       # buffer / faint elements
plt.rcParams.update({
    "font.family": "Arial", "font.size": 16, "axes.labelsize": 16, "axes.titlesize": 16,
    "xtick.labelsize": 14, "ytick.labelsize": 14, "axes.spines.top": False,
    "axes.spines.right": False, "axes.linewidth": 1.2, "xtick.major.width": 1.2,
    "ytick.major.width": 1.2, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "savefig.facecolor": "white", "figure.facecolor": "white", "legend.frameon": False,
    "svg.fonttype": "none",
})
MANIFEST, FACTS = [], {}


def save(fig, name, **prov):
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=300)
    fig.savefig(os.path.join(OUT, name + ".svg"))
    plt.close(fig)
    MANIFEST.append(dict(file=name + ".png / .svg", **prov))
    print("wrote", name)


def tag(ax, text, loc="right"):
    ax.text(1.0 if loc == "right" else 0.0, 1.01, text, transform=ax.transAxes, fontsize=11,
            color="#555555", ha=loc, va="bottom")


# ================================================================== slide 1 + 2: raw H5
RAW_FILE = AIM1_CM + "CM_s2_4_sub2_3_20250623_143348_merged.h5"
REC_DIR = AIM1_REC + "CM_s2_4_sub2_3_20250623_143348.rec/CM_s2_4_sub2_3_20250623_143348_merged.analog/"
with h5py.File(RAW_FILE, "r") as f:
    raw32 = f["resp"][:]
    rmeta = dict(f["resp_metadata"].attrs)
FS_RAW = float(rmeta["sampling_frequency"])
FACTS["s1_h5_dtype"] = f"{raw32.dtype}, shape {raw32.shape}"
FACTS["s1_h5_fs"] = FS_RAW
raw = np.asarray(raw32, np.float64).flatten()
FACTS["s1_raw_integer_valued"] = bool(np.all(raw[:200000] == np.round(raw[:200000])))

# exact preprocess_resp_to_target_rate stages, kept separately
b, a = butter(N=4, Wn=(400 / 2) / (FS_RAW / 2), btype="low")
lp = filtfilt(b, a, raw)
dec = resample_poly(lp, up=1, down=int(FS_RAW // 400))
bp = nk.signal_filter(dec, lowcut=0.1, highcut=20, method="butterworth", sampling_rate=400, order=2)
cached, _ = aim1_signal("CM_2_4_sub2_3")
FACTS["s2_final_equals_cache_maxabsdiff"] = float(np.max(np.abs(bp - cached)))
assert np.allclose(bp, cached, atol=1e-6), "recomputed 400 Hz signal differs from workbench cache"

# duplicate-tick evidence from the Trodes export of the same session
ts = np.asarray(read_trodes_extracted_data_file(REC_DIR + "CM_s2_4_sub2_3_20250623_143348_merged.timestamps.dat")["data"]["time"]).astype(np.int64)
ain = read_trodes_extracted_data_file(REC_DIR + "CM_s2_4_sub2_3_20250623_143348_merged.analog_ECU_Ain1.dat")["data"]["voltage"]
FACTS["s1_export_n"] = int(ts.size)
FACTS["s1_h5_n"] = int(raw.size)
FACTS["s1_h5_equals_export"] = bool(ts.size == raw.size and np.array_equal(np.asarray(ain).ravel()[:1000000], raw32.ravel()[:1000000].astype(np.int16)))
dup = np.r_[False, np.diff(ts) == 0]
FACTS["s1_n_duplicate_ticks"] = int(dup.sum())
FACTS["s1_index_time_stretch_pct"] = float(100 * dup.sum() / (ts.size - dup.sum()))
FACTS["s1_index_minus_tick_time_end_s"] = float(raw.size / FS_RAW - (ts[-1] - ts[0]) / FS_RAW)

# --- slide 1 raw trace with zoom
T0, T1 = 120.0, 123.0
Z0, Z1 = 121.20, 121.50
i0, i1 = int(T0 * FS_RAW), int(T1 * FS_RAW)
tr = np.arange(i0, i1) / FS_RAW
fig = plt.figure(figsize=(9.6, 4.6))
ax = fig.add_axes([0.10, 0.17, 0.60, 0.70])
az = fig.add_axes([0.77, 0.17, 0.21, 0.70])
ax.plot(tr, raw[i0:i1], color=SIG, lw=1.2)
ylo, yhi = raw[i0:i1].min(), raw[i0:i1].max()
pad = 0.06 * (yhi - ylo)
ax.set_ylim(ylo - pad, yhi + pad)
ax.add_patch(Rectangle((Z0, ylo - pad * 0.6), Z1 - Z0, yhi - ylo + pad * 1.2, fill=False, ec=BLUE, lw=1.5))
ax.set_xlim(T0, T1)
ax.set_xlabel("time (s, index / 20 000)")
ax.set_ylabel("ADC counts")
ax.set_title("ECU_Ain1 · raw · 20 000 Hz")
j0, j1 = int(Z0 * FS_RAW), int(Z1 * FS_RAW)
az.plot(np.arange(j0, j1) / FS_RAW, raw[j0:j1], color=SIG, lw=1.4)
az.set_xlim(Z0, Z1)
az.set_xticks([Z0, Z1])
az.set_xticklabels([f"{Z0:.1f}", f"{Z1:.1f}"])
az.set_title("300 ms zoom", color=BLUE)
for s in az.spines.values():
    s.set_edgecolor(BLUE)
az.set_xlabel("time (s)")
tag(ax, "Aim1 rank · CM_s2_4_sub2_3_20250623 · /resp", "left")
fig.text(0.985, 0.955, "ADC counts ≠ volts", ha="right", fontsize=12, color="#555555")
save(fig, "s1_raw_trace", dataset="Aim1 cagemate (cohort 1)", recording="CM_s2_4_sub2_3_20250623_143348_merged",
     source=RAW_FILE + " :: /resp", interval=f"{T0}-{T1} s (zoom {Z0}-{Z1} s), index time",
     fs="20000 Hz", stage="raw ADC counts as stored in H5 (float32 of int16 values)", algorithm="none")

# --- slide 1 duplicate-tick inset
fig, ax = plt.subplots(figsize=(2.9, 1.9))
t_idx = np.arange(ts.size) / FS_RAW
step = 2000
ax.plot(t_idx[::step], np.cumsum(dup)[::step] / 1000, color=SIG, lw=2)
ax.set_xlabel("index time (s)", fontsize=10)
ax.set_ylabel("repeated\nticks (k)", fontsize=10)
ax.tick_params(labelsize=9)
fig.tight_layout(pad=0.3)
save(fig, "s1_duplicate_ticks", dataset="Aim1 cagemate", recording="CM_s2_4_sub2_3_20250623_143348_merged",
     source=REC_DIR + "...merged.timestamps.dat", interval="whole session",
     fs="20000 Hz ticks", stage="Trodes export timestamps; cumulative count of samples whose tick equals the previous tick",
     algorithm="np.diff(ticks)==0")

# --- slide 2 stack
T0, T1 = 120.0, 124.0
i0, i1 = int(T0 * FS_RAW), int(T1 * FS_RAW)
n0, n1 = int(T0 * 400), int(T1 * 400)
fig, axs = plt.subplots(3, 1, figsize=(9.7, 5.75), sharex=True)
axs[0].plot(np.arange(i0, i1) / FS_RAW, raw[i0:i1], color=SIG, lw=1.0)
axs[0].set_title("Raw · 20 kHz")
axs[1].plot(np.arange(n0, n1) / 400, dec[n0:n1], color=SIG, lw=1.6)
axs[1].set_title("Antialiased + decimated · 400 Hz")
lo = min(raw[i0:i1].min(), dec[n0:n1].min())
hi = max(raw[i0:i1].max(), dec[n0:n1].max())
for k in (0, 1):
    axs[k].set_ylim(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo))
    axs[k].set_ylabel("ADC counts")
axs[2].plot(np.arange(n0, n1) / 400, bp[n0:n1], color=SIG, lw=1.6)
axs[2].axhline(0, color=GRAY, lw=0.8, zorder=0)
axs[2].set_title("Band-passed · 400 Hz  (zero-centred)")
axs[2].set_ylabel("ADC units")
axs[2].set_xlabel("time (s)")
axs[2].set_xlim(T0, T1)
tag(axs[0], "Aim1 rank · CM_s2_4_sub2_3 · same samples in all panels")
fig.tight_layout(h_pad=0.6)
save(fig, "s2_preprocessing_stack", dataset="Aim1 cagemate", recording="CM_s2_4_sub2_3_20250623_143348_merged",
     source=RAW_FILE + " :: /resp ; check vs workbench_cache/CM_2_4_sub2_3.npz", interval=f"{T0}-{T1} s",
     fs="20000 Hz (top) / 400 Hz (middle, bottom)",
     stage="top raw; middle butter(4, 200 Hz) filtfilt + resample_poly(1,50); bottom nk.signal_filter 0.1-20 Hz order 2",
     algorithm="resp_helper_functions.preprocess_resp_to_target_rate (stages recomputed; final max |diff| vs cache "
               f"{FACTS['s2_final_equals_cache_maxabsdiff']:.2e})")
del raw, lp, raw32

# ================================================================== slide 3: frozen QC
QC_COL = {"dead": RED, "artifact": RED, "degraded": RED}


def qc_shade(ax, reason, t, lo, hi):
    core = np.isin(reason, ["dead", "artifact", "degraded"])
    buf = reason == "buffer"
    ax.fill_between(t, lo, hi, where=buf, color=GRAY, alpha=0.30, lw=0, step="mid")
    ax.fill_between(t, lo, hi, where=core, color=RED, alpha=0.22, lw=0, step="mid")


def runs(mask):
    m = np.r_[False, mask, False]
    d = np.diff(m.astype(int))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


qc_rows = []
for k in [k for k in sorted(wb.all_recordings()) if k not in wb.EXCLUDE]:
    s_, _ = aim1_signal(k)
    ex, _r = wb.bad_signal_mask(s_, FS)
    qc_rows.append(dict(key=k, n=len(s_), ex=int(ex.sum())))
qdf = pd.DataFrame(qc_rows)
FACTS["s3_excluded_min"] = float(qdf.ex.sum() / FS / 60)
FACTS["s3_recordings_affected"] = f"{int((qdf.ex > 0).sum())}/{len(qdf)}"

MAIN = ("RI2_2_4", 270.0, 395.0)
sig, _ = aim1_signal(MAIN[0])
ex, reason = wb.bad_signal_mask(sig, FS)
s0, s1 = int(MAIN[1] * FS), int(MAIN[2] * FS)
t = np.arange(s0, s1) / FS
fig, ax = plt.subplots(figsize=(9.7, 2.75))
lo, hi = sig[s0:s1].min(), sig[s0:s1].max()
lo, hi = lo - 0.05 * (hi - lo), hi + 0.22 * (hi - lo)
qc_shade(ax, reason[s0:s1], t, lo, hi)
ax.plot(t, sig[s0:s1], color=SIG, lw=0.7)
ax.set_ylim(lo, hi)
ax.set_xlim(MAIN[1], MAIN[2])
ax.set_ylabel("ADC units")
ax.set_xlabel("time (s)")
labelled = set()
for rs in ["degraded", "artifact", "dead"]:
    for a_, z_ in runs(reason[s0:s1] == rs):
        if (z_ - a_) / FS < 0.5:
            continue
        xm = (s0 + (a_ + z_) / 2) / FS
        ax.text(xm, hi - 0.02 * (hi - lo), rs, ha="center", va="top", fontsize=13, color="#8B1E1A")
        labelled.add(rs)
# one ±10 s buffer bracket: the first buffer run
for a_, z_ in runs(reason[s0:s1] == "buffer"):
    if abs((z_ - a_) / FS - 10) < 0.6:
        xa, xz = (s0 + a_) / FS, (s0 + z_) / FS
        yb = lo + 0.06 * (hi - lo)
        ax.annotate("", (xa, yb), (xz, yb), arrowprops=dict(arrowstyle="<->", color="#555555", lw=1.2))
        ax.text((xa + xz) / 2, yb + 0.03 * (hi - lo), "10 s buffer", ha="center", fontsize=11, color="#555555")
        break
FACTS["s3_main_labels"] = sorted(labelled)
tag(ax, "Aim1 valence · RI2_2_4 · 400 Hz · red = core exclusion, gray = ±10 s buffer")
fig.tight_layout()
save(fig, "s3_qc_main", dataset="Aim1 valence RI2", recording="RI2_2_4", source="workbench_cache/RI2_2_4.npz",
     interval=f"{MAIN[1]}-{MAIN[2]} s", fs="400 Hz", stage="band-passed 400 Hz input; frozen bad_signal_mask overlay",
     algorithm=f"workbench_support.bad_signal_mask ({QC_VERSION})")

ZOOMS = [("a  dead", "CM_1_1_d1_2", 476.0, 494.0), ("b  artifact", "RI1_2_3", 149.5, 154.5),
         ("c  degraded", "RI2_1_1", 341.0, 349.0)]
fig, axs = plt.subplots(1, 3, figsize=(9.7, 2.55))
for ax, (lab, key, z0, z1) in zip(axs, ZOOMS):
    s_, _ = aim1_signal(key)
    _ex, rs_ = wb.bad_signal_mask(s_, FS)
    a_, z_ = int(z0 * FS), int(z1 * FS)
    tt = np.arange(a_, z_) / FS
    lo, hi = s_[a_:z_].min(), s_[a_:z_].max()
    lo, hi = lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo)
    qc_shade(ax, rs_[a_:z_], tt, lo, hi)
    ax.plot(tt, s_[a_:z_], color=SIG, lw=0.9)
    ax.set_xlim(z0, z1)
    ax.set_ylim(lo, hi)
    ax.set_title(lab, fontsize=15)
    ax.set_yticks([])
    ax.set_xlabel("time (s)", fontsize=13)
    ax.tick_params(labelsize=12)
    ax.text(0.0, -0.36, key, transform=ax.transAxes, fontsize=10, color="#555555")
    present = sorted(set(rs_[a_:z_]) - {""})
    MANIFEST.append(dict(file=f"s3_qc_zooms.png panel '{lab}'", dataset="Aim1", recording=key,
                         source=f"workbench_cache/{key}.npz", interval=f"{z0}-{z1} s", fs="400 Hz",
                         stage="band-passed 400 Hz; frozen QC overlay; reasons present: " + ", ".join(present),
                         algorithm=f"workbench_support.bad_signal_mask ({QC_VERSION})"))
fig.tight_layout(w_pad=1.2)
save(fig, "s3_qc_zooms", dataset="see panel rows", recording="CM_1_1_d1_2 / RI1_2_3 / RI2_1_1",
     source="workbench_cache", interval="see panel rows", fs="400 Hz", stage="frozen QC", algorithm=QC_VERSION)

# ================================================================== slide 4: detector validity
EX4 = [("valid · low_amp (periodic)", "aim1", "BLRI_3_6", 13.5, 16.5, 15.00),
       ("invalid · flat_or_dropout", "c3", "11_2_s_CM_11_4_d_20260729_172306", 489.5, 492.5, 490.50)]
fig = plt.figure(figsize=(9.9, 5.7))
gs = fig.add_gridspec(4, 2, width_ratios=[2.9, 1], height_ratios=[1, 0.16, 1, 0.16], hspace=0.55, wspace=0.28)
acf_axes = []
for r, (lab, coh, key, w0, w1, blk) in enumerate(EX4):
    s_ = aim1_signal(key)[0] if coh == "aim1" else cohort3_signal(key)
    T = trace_detector(s_)
    bl = T["blocks"]
    row = bl[np.isclose(bl.start_s, blk)].iloc[0]
    a_, z_ = int(w0 * FS), int(w1 * FS)
    tt = np.arange(a_, z_) / FS
    ax = fig.add_subplot(gs[2 * r, 0])
    x = T["x"][a_:z_]
    ax.plot(tt, x, color=SIG, lw=1.4)
    ax.axvspan(blk, blk + 1.0, color=BLUE, alpha=0.08, lw=0)
    ax.axvline(blk, color=BLUE, lw=0.8, ls=":")
    ax.axvline(blk + 1.0, color=BLUE, lw=0.8, ls=":")
    ax.set_xlim(w0, w1)
    ax.set_ylabel("ADC units", fontsize=14)
    ax.set_title(lab, fontsize=15)
    src = "Aim1 valence · " + key if coh == "aim1" else "BLA cohort 2 · 11_2 (Section 1)"
    tag(ax, f"{src} · rel = {row.rel:.3f}")
    ax.tick_params(labelbottom=False)
    sx = fig.add_subplot(gs[2 * r + 1, 0], sharex=ax)
    v, la = T["valid"][a_:z_], T["low_amp"][a_:z_]
    sx.fill_between(tt, 0, 1, where=~v, color=RED, alpha=0.35, lw=0, step="mid")
    sx.fill_between(tt, 0, 1, where=v & la, color=GRAY, alpha=0.45, lw=0, step="mid")
    sx.fill_between(tt, 0, 1, where=v & ~la, color="white", lw=0, step="mid")
    sx.set_yticks([])
    sx.set_ylim(0, 1)
    sx.spines["left"].set_visible(False)
    labs = []
    if (~v).any():
        labs.append(("invalid", RED))
    if (v & la).any():
        labs.append(("valid · low_amp", "#555555"))
    if (v & ~la).any():
        labs.append(("valid", "#555555"))
    sx.text(1.005, 0.5, " / ".join(l for l, _c in labs), transform=sx.transAxes, fontsize=11,
            va="center", ha="left", color=labs[0][1])
    sx.tick_params(labelsize=12)
    if r == 1:
        sx.set_xlabel("time (s)", fontsize=14)
    # autocorrelation exactly as _acf_peak (on the 1 s block of x, NaN-free)
    bseg = T["x"][int(blk * FS):int(blk * FS) + FS]
    bseg = bseg[np.isfinite(bseg)]
    c = bseg - bseg.mean()
    ac = np.correlate(c, c, "full")[len(c) - 1:]
    ac = ac / ac[0]
    lags = np.arange(len(ac)) / FS
    l0, l1 = int(0.06 * FS), int(0.5 * FS)
    acmax = float(ac[l0:l1].max())
    assert np.isclose(acmax, D["_acf_peak"](bseg, FS)) and np.isclose(acmax, row.acf)
    axa = fig.add_subplot(gs[2 * r, 1])
    axa.plot(lags[l0:l1] * 1000, ac[l0:l1], color=SIG, lw=1.6)
    k = l0 + int(np.argmax(ac[l0:l1]))
    axa.plot(lags[k] * 1000, ac[k], "o", color=BLUE if acmax >= 0.5 else ORANGE, ms=9, zorder=5)
    axa.text(lags[k] * 1000, ac[k] + 0.12, f"{acmax:.2f}", ha="center", fontsize=13,
             color=BLUE if acmax >= 0.5 else ORANGE)
    axa.axhline(0.5, color=SIG, lw=1.0, ls="--")
    axa.set_ylim(-1, 1.15)
    axa.set_xlim(60, 500)
    axa.set_xticks([60, 250, 500])
    axa.set_ylabel("ρ", fontsize=15)
    axa.set_title("ACF of block", fontsize=13)
    axa.tick_params(labelsize=12)
    if r == 1:
        axa.set_xlabel("lag (ms)", fontsize=13)
    FACTS[f"s4_{key}"] = dict(block=blk, rel=float(row.rel), acf=acmax, decision=row.decision,
                              noise_ptp=float(T["noise_ptp"]))
    MANIFEST.append(dict(file=f"s4_validity.png row {r + 1}", dataset="Aim1 valence BLRI" if coh == "aim1" else "BLA cohort 2 (overall cohort 3) cagemate",
                         recording=key, source=("workbench_cache/" + key + ".npz") if coh == "aim1" else f"workbench_cache/cohort3_{key}.npz",
                         interval=f"{w0}-{w1} s; ACF block {blk}-{blk + 1} s", fs="400 Hz",
                         stage="detector input x (polarity +1, unsmoothed 400 Hz); validity_mask decisions",
                         algorithm=DETECTOR_VERSION + f"; block decision '{row.decision}', rel {row.rel:.4f}, acf {acmax:.3f}"))
save(fig, "s4_validity", dataset="see rows", recording="BLRI_3_6 / 11_2", source="workbench_cache",
     interval="see rows", fs="400 Hz", stage="Component A", algorithm=DETECTOR_VERSION)

# ================================================================== slides 5 & 6: BLRI_4_7
K56, W0, W1 = "BLRI_4_7", 530.0, 534.5
sig, _ = aim1_signal(K56)
T = trace_detector(sig)
a_, z_ = int(W0 * FS), int(W1 * FS)
segs = [(p, q) for p, q in T["segments"] if p <= a_ and q >= z_]
assert segs, "window not inside one valid segment"
S0 = segs[0][0]
assert not wb.bad_signal_mask(sig, FS)[0][a_:z_].any(), "window touches frozen-QC exclusion"
tt = np.arange(a_, z_) / FS
PP = D["PEAK_PARAMS"]
N = T["noise_ptp"]
hp = T["hp"]
# all local maxima of hp with their prominence / width, exactly as select_peaks computes them (segment-wide)
segS, segE = segs[0]
hseg = hp[segS:segE]
cand, pr = find_peaks(hseg, prominence=0, wlen=max(int(PP["prominence_wlen_s"] * FS), 3), width=0, rel_height=0.5)
cand_abs = cand + segS
inwin = (cand_abs >= a_) & (cand_abs < z_)
lrA = T["lr_A"]
need_t = np.maximum(PP["prominence_frac"] * lrA, PP["noise_mult"] * N)
rowsA = pd.DataFrame(T["rows_A"])
rA = rowsA[(rowsA.idx >= a_) & (rowsA.idx < z_)]
pkA = T["pkA"][(T["pkA"] >= a_) & (T["pkA"] < z_)]
rejA = rA[rA.reason != "counted"]
FACTS["s5_window"] = f"{K56} {W0}-{W1}"
FACTS["s5_n_local_maxima"] = int(inwin.sum())
FACTS["s5_n_pkA"] = int(len(pkA))
FACTS["s5_rejected"] = rejA[["idx", "prominence", "required", "width_s", "reason"]].assign(t=lambda d: d.idx / FS).round(3).to_dict("records")
FACTS["s5_noise_ptp"] = float(N)
FACTS["s5_need_range"] = [float(np.nanmin(need_t[a_:z_])), float(np.nanmax(need_t[a_:z_]))]

fig = plt.figure(figsize=(9.6, 4.55))
ax = fig.add_axes([0.09, 0.47, 0.89, 0.45])
ap = fig.add_axes([0.09, 0.12, 0.89, 0.26], sharex=ax)
ax.plot(tt, hp[a_:z_], color=SIG, lw=1.5)
ci = cand_abs[inwin]
ax.plot(ci / FS, hp[ci], "o", ms=5, color=GRAY, mfc="none", mew=1.2, zorder=3)
ax.plot(pkA / FS, hp[pkA], "^", ms=10, color=BLUE, zorder=4)
if len(rejA):
    ax.plot(rejA.idx / FS, hp[rejA.idx.to_numpy()], "x", ms=10, mew=2.5, color=ORANGE, zorder=5)
ax.set_ylabel("ADC units")
ax.set_title("Pass A input: hp = HP₁Hz(smoothed)")
ax.tick_params(labelbottom=False)
ax.text(0.995, 0.97, "LOCAL prominence threshold", transform=ax.transAxes, ha="right", va="top", fontsize=12,
        color="#555555")
tag(ax, f"Aim1 valence · {K56}")
prom = pr["prominences"][inwin]
okw = (pr["widths"][inwin] / FS) >= PP["min_peak_width_s"]
is_acc = np.isin(ci, pkA)
ap.plot(tt, need_t[a_:z_], color=SIG, lw=2.0, label="need(t)")
ap.plot(tt, PP["prominence_frac"] * lrA[a_:z_], color=GRAY, lw=1.2, ls="--", label="0.20·LR(t)")
ap.axhline(PP["noise_mult"] * N, color=GRAY, lw=1.2, ls=":", label="2N")
for xi, p_, acc in zip(ci, prom, is_acc):
    ap.vlines(xi / FS, 1, p_, color=BLUE if acc else ORANGE, lw=1.6 if acc else 1.2, alpha=1 if acc else 0.9)
    ap.plot(xi / FS, p_, "o", ms=4.5, color=BLUE if acc else ORANGE)
ap.set_yscale("log")
ap.set_ylim(1, max(prom.max(), np.nanmax(need_t[a_:z_])) * 2)
ap.set_ylabel("prominence", fontsize=14)
ap.set_xlabel("time (s)")
ap.set_xlim(W0, W1)
ap.legend(loc="upper right", fontsize=11, ncol=3, handlelength=1.6, bbox_to_anchor=(1.0, 1.25))
save(fig, "s5_prominence", dataset="Aim1 valence BLRI", recording=K56, source=f"workbench_cache/{K56}.npz",
     interval=f"{W0}-{W1} s", fs="400 Hz",
     stage="hp = 1 Hz Butterworth(2) filtfilt of 5-sample running mean; Pass A select_peaks; prominence stems from find_peaks(wlen=400, rel_height=0.5)",
     algorithm=DETECTOR_VERSION)

# single-peak geometry: an accepted Pass A peak with a logged rejected neighbour if possible
pick = None
for r_ in rejA.itertuples():
    near = pkA[np.abs(pkA - r_.idx) < 0.25 * FS]
    if len(near):
        pick = (int(near[np.argmin(np.abs(near - r_.idx))]), r_)
        break
if pick is None:
    pick = (int(pkA[len(pkA) // 2]), None)
p_abs, rj = pick
j = int(np.where(cand_abs == p_abs)[0][0])
lb, rb = pr["left_bases"][j] + segS, pr["right_bases"][j] + segS
wh, lips, rips = pr["width_heights"][j], pr["left_ips"][j] + segS, pr["right_ips"][j] + segS
pv, pp_ = hp[p_abs], pr["prominences"][j]
base_level = pv - pp_
z0 = min(lb, p_abs - int(0.12 * FS), rj.idx if rj is not None else p_abs) - int(0.03 * FS)
z1 = max(rb, p_abs + int(0.12 * FS), rj.idx if rj is not None else p_abs) + int(0.03 * FS)
z0, z1 = max(z0, p_abs - int(0.2 * FS)), min(z1, p_abs + int(0.2 * FS))
fig, ax = plt.subplots(figsize=(5.0, 2.65))
ax.plot(np.arange(z0, z1) / FS * 1000 - p_abs / FS * 1000, hp[z0:z1], color=SIG, lw=1.8)
xr = lambda i: i / FS * 1000 - p_abs / FS * 1000  # noqa: E731
ax.plot(0, pv, "^", ms=11, color=BLUE, zorder=5)
ax.annotate("", (0, base_level), (0, pv), arrowprops=dict(arrowstyle="<->", color=BLUE, lw=1.8))
ax.text(4, (pv + base_level) / 2, f"prominence\n{pp_:.0f}", fontsize=11, color=BLUE, va="center")
ax.hlines(base_level, xr(max(lb, z0)), xr(min(rb, z1 - 1)), color=GRAY, lw=1.2, ls="--")
ax.annotate("", (xr(lips), wh), (xr(rips), wh), arrowprops=dict(arrowstyle="<->", color=SIG, lw=1.6))
ax.text(xr(rips) + 4, wh, f"width {1000 * (rips - lips) / FS:.0f} ms", fontsize=11, va="center")
if rj is not None and z0 <= rj.idx < z1:
    ax.plot(xr(rj.idx), hp[rj.idx], "x", ms=11, mew=2.5, color=ORANGE, zorder=5)
    ax.text(xr(rj.idx), hp[rj.idx] - 0.12 * (hp[z0:z1].max() - hp[z0:z1].min()),
            f"{rj.reason}\n{rj.prominence:.0f} < {rj.required:.0f}" if rj.reason == "low_prominence" else rj.reason,
            fontsize=10, color=ORANGE, ha="center", va="top")
ax.set_xlabel("ms from peak", fontsize=13)
ax.set_ylabel("ADC units", fontsize=13)
ax.tick_params(labelsize=11)
ax.set_xlim(xr(z0), xr(z1))
tag(ax, f"{K56} · t = {p_abs / FS:.3f} s · need {need_t[p_abs]:.0f}")
fig.tight_layout(pad=0.4)
FACTS["s5_peak_geometry"] = dict(t=p_abs / FS, prominence=float(pp_), width_ms=float(1000 * (rips - lips) / FS),
                                 need=float(need_t[p_abs]),
                                 neighbour=None if rj is None else dict(t=rj.idx / FS, reason=rj.reason,
                                                                        prominence=rj.prominence, required=rj.required))
save(fig, "s5_peak_geometry", dataset="Aim1 valence BLRI", recording=K56, source=f"workbench_cache/{K56}.npz",
     interval=f"{z0 / FS:.3f}-{z1 / FS:.3f} s", fs="400 Hz",
     stage="hp (Pass A input); scipy find_peaks prominence/base/width_heights/ips of this peak", algorithm=DETECTOR_VERSION)

# --- slide 6: two passes, same samples
pkB = T["pkB_raw"][(T["pkB_raw"] >= a_) & (T["pkB_raw"] < z_)]
cm = [c_ for c_ in T["cycle_means_A"] if c_[1] > a_ and c_[0] < z_]
FACTS["s6"] = dict(n_pkA=int(len(pkA)), n_pkB=int(len(pkB)), baseline_range=float(np.ptp(T["bl_A"][a_:z_])))
fig, axs = plt.subplots(3, 1, figsize=(9.7, 5.75), sharex=True)
axs[0].plot(tt, hp[a_:z_], color=SIG, lw=1.5)
axs[0].plot(pkA / FS, hp[pkA], "^", ms=9, color=BLUE)
axs[0].set_title(f"PASS A · provisional peaks on hp   (n = {len(pkA)})")
seg = T["seg"]
axs[1].plot(tt, seg[a_:z_], color=SIG, lw=1.5)
for p_ in pkA:
    axs[1].axvline(p_ / FS, color=BLUE, lw=0.8, ls=":", alpha=0.8)
axs[1].plot([((p + q) / 2) / FS for p, q, _ in cm], [m for *_, m in cm], "o", ms=6, color=BLUE, zorder=5)
axs[1].plot(tt, T["bl_A"][a_:z_], color=SIG, lw=2.2, ls=(0, (5, 3)), alpha=0.75)
# bracket over one cycle in the middle of the window
pa, pb, pm = cm[len(cm) // 2]
yb = seg[a_:z_].max() * 1.05
axs[1].annotate("", (pa / FS, yb), (pb / FS, yb), arrowprops=dict(arrowstyle="|-|", color=BLUE, lw=1.2,
                                                                   mutation_scale=4))
axs[1].set_title("CYCLE MEANS · baseline b(t) on seg")
axs[1].text(0.995, 0.03, "cycle-mean baseline assumption", transform=axs[1].transAxes, ha="right",
            fontsize=11, color="#555555")
y = T["y_A"]
axs[2].plot(tt, y[a_:z_], color=SIG, lw=1.5)
axs[2].axhline(0, color=GRAY, lw=0.8, zorder=0)
axs[2].plot(pkB / FS, y[pkB], "^", ms=9, color=BLUE)
axs[2].set_title(f"PASS B · y = seg − b   (n = {len(pkB)})")
axs[2].set_xlabel("time (s)")
for ax in axs:
    ax.set_ylabel("ADC units", fontsize=14)
axs[2].set_xlim(W0, W1)
tag(axs[0], f"Aim1 valence · {K56} · same samples in all panels")
fig.tight_layout(h_pad=0.5)
save(fig, "s6_two_passes", dataset="Aim1 valence BLRI", recording=K56, source=f"workbench_cache/{K56}.npz",
     interval=f"{W0}-{W1} s", fs="400 Hz",
     stage="top hp + Pass A pkA; middle seg (5-sample running mean, polarity +1) + cycle means + interpolated baseline from pkA; bottom y_A = seg - b_A + Pass B peaks before trough validation",
     algorithm=DETECTOR_VERSION)

# ================================================================== slide 7: trough validation
K7, W0, W1 = "BLRI_1_1", 527.5, 532.5
sig, _ = aim1_signal(K7)
T = trace_detector(sig)
a_, z_ = int(W0 * FS), int(W1 * FS)
assert not wb.bad_signal_mask(sig, FS)[0][a_:z_].any()
tt = np.arange(a_, z_) / FS
yF, yA, lrB = T["y_final"], T["y_A"], T["lr_B"]
need = lambda i: max(PP["prominence_frac"] * lrB[i], PP["noise_mult"] * T["noise_ptp"])  # noqa: E731
pk = T["pk_final"][(T["pk_final"] >= a_) & (T["pk_final"] < z_)]
tr = T["troughs"][(T["troughs"] >= a_) & (T["troughs"] < z_)]
rB = pd.DataFrame(T["rows_B"])
rB = rB[(rB.idx >= a_) & (rB.idx < z_)]
pairs = []
for p, q in zip(pk[:-1], pk[1:]):
    t_ = p + int(np.argmin(yA[p:q]))
    L = p if yA[p] <= yA[q] else q
    pairs.append(dict(p=p, q=q, t=t_, L=L, delta=yA[L] - yA[t_], need=need(L), ratio=(yA[L] - yA[t_]) / need(L)))
pairs = pd.DataFrame(pairs)
pB = pairs.loc[pairs.ratio.idxmin()]
pA = pairs.iloc[(pairs.ratio - pairs.ratio.median()).abs().argsort().iloc[0]]
FACTS["s7"] = dict(window=f"{K7} {W0}-{W1}", n_final=int(len(pk)), n_pkB_raw=int(((T["pkB_raw"] >= a_) & (T["pkB_raw"] < z_)).sum()),
                   rejected_reasons=rB.reason.value_counts().to_dict(),
                   shallow_trough_in_window=int((rB.reason == "shallow_trough").sum()),
                   pairA=dict(t=[pA.p / FS, pA.q / FS], delta=float(pA.delta), need=float(pA.need), ratio=float(pA.ratio)),
                   pairB=dict(t=[pB.p / FS, pB.q / FS], delta=float(pB.delta), need=float(pB.need), ratio=float(pB.ratio)))
fig, ax = plt.subplots(figsize=(9.7, 3.0))
ax.plot(tt, yF[a_:z_], color=SIG, lw=1.4)
ax.axhline(0, color=GRAY, lw=0.8, zorder=0)
cands = rB.idx.to_numpy()
ax.plot(cands / FS, yF[cands], "o", ms=6, mfc="none", mec=GRAY, mew=1.2, zorder=3)
ax.plot(pk / FS, yF[pk], "^", ms=8, color=BLUE, zorder=4)
ax.plot(tr / FS, yF[tr], "v", ms=8, color=ORANGE, zorder=4)
st = rB[rB.reason == "shallow_trough"]
if len(st):
    ax.plot(st.idx / FS, yF[st.idx.to_numpy()], "x", ms=10, mew=2.5, color=ORANGE)
for prr, lab in ((pA, "A"), (pB, "B")):
    for v in (prr.p, prr.q):
        ax.axvline(v / FS, color=SIG, lw=0.9, ls=":")
    ax.text((prr.p + prr.q) / 2 / FS, np.nanmax(yF[a_:z_]) * 1.12, lab, ha="center", fontsize=14, fontweight="bold")
ax.set_ylim(np.nanmin(yF[a_:z_]) * 1.15, np.nanmax(yF[a_:z_]) * 1.3)
ax.set_xlim(W0, W1)
ax.set_ylabel("y (ADC units)", fontsize=14)
ax.set_xlabel("time (s)")
tag(ax, f"Aim1 valence · {K7} · final y, final peaks ▲, final troughs ▼, logged Pass B candidates ○")
fig.tight_layout()
save(fig, "s7_trough_main", dataset="Aim1 valence BLRI", recording=K7, source=f"workbench_cache/{K7}.npz",
     interval=f"{W0}-{W1} s", fs="400 Hz",
     stage="final y = seg - baseline(final peaks); final peaks; troughs recomputed on final y; gray circles = all logged Pass B candidates (counted + rejected)",
     algorithm=DETECTOR_VERSION)

fig, axs = plt.subplots(1, 2, figsize=(9.7, 2.6))
for ax, prr, lab in ((axs[0], pA, "A · typical accepted pair"), (axs[1], pB, "B · closest accepted pair in Aim1")):
    p, q, t_, L = int(prr.p), int(prr.q), int(prr.t), int(prr.L)
    z0, z1 = p - int(0.05 * FS), q + int(0.05 * FS)
    xx = (np.arange(z0, z1) - t_) / FS * 1000
    ax.plot(xx, yA[z0:z1], color=SIG, lw=1.8)
    ax.plot([(p - t_) / FS * 1000, (q - t_) / FS * 1000], [yA[p], yA[q]], "^", ms=10, color=BLUE)
    ax.plot(0, yA[t_], "v", ms=10, color=ORANGE)
    xL = (L - t_) / FS * 1000
    ax.hlines(yA[L], xL, 0, color=GRAY, lw=1, ls=":")
    ax.annotate("", (0, yA[t_]), (0, yA[L]), arrowprops=dict(arrowstyle="<->", color=SIG, lw=1.6))
    ax.text(4, (yA[L] + yA[t_]) / 2, f"Δ = {prr.delta:.0f}", fontsize=12, va="center")
    xn = xL + (6 if xL < 0 else -6)
    ax.plot([xn, xn], [yA[L], yA[L] - prr.need], color=BLUE, lw=5, alpha=0.45, solid_capstyle="butt")
    ax.text(xn + (-4 if xL < 0 else 4), yA[L] - prr.need, f"need(L) = {prr.need:.0f}", fontsize=12,
            ha="right" if xL < 0 else "left", va="center", color=BLUE)
    ax.text(xL, yA[L] + 0.06 * (np.nanmax(yA[z0:z1]) - np.nanmin(yA[z0:z1])), "L", ha="center", fontsize=13,
            fontweight="bold")
    ax.set_title(f"{lab}   ({prr.delta:.0f} ≥ {prr.need:.0f} → keep)", fontsize=14)
    ax.set_xlabel("ms from trough", fontsize=13)
    ax.set_ylabel("y (ADC units)", fontsize=12)
    ax.tick_params(labelsize=11)
    ax.set_ylim(np.nanmin(yA[z0:z1]) * 1.15, np.nanmax(yA[z0:z1]) * 1.25)
    tag(ax, f"{K7} {p / FS:.3f}-{q / FS:.3f} s")
fig.tight_layout(w_pad=2.0)
save(fig, "s7_trough_pairs", dataset="Aim1 valence BLRI", recording=K7, source=f"workbench_cache/{K7}.npz",
     interval=f"pair A {pA.p / FS:.3f}-{pA.q / FS:.3f} s; pair B {pB.p / FS:.3f}-{pB.q / FS:.3f} s", fs="400 Hz",
     stage="y_A (validation-stage signal = seg - baseline from Pass A peaks), the array _validate_cycles tests; "
           "need(L) = max(0.20 LR_hp(L), 2N)", algorithm=DETECTOR_VERSION)

# ================================================================== slide 8: OLD vs NEW (Section 1)
K8 = "11_3_s_CM_11_4_d_20260729_122040"


def old_new(key, t0, t1):
    s_ = cohort3_signal(key)
    lp_, lx_, thr_ = wb._cohort3_legacy_fit(key, s_)
    T_ = trace_detector(s_)
    a, z = int(t0 * FS), int(t1 * FS)
    return s_, lp_[(lp_ >= a) & (lp_ < z)], T_["pk_final"][(T_["pk_final"] >= a) & (T_["pk_final"] < z)], thr_, T_


rows8 = [(300.0, 301.5), (550.0, 551.5)]
fig, axs = plt.subplots(2, 2, figsize=(12.2, 4.75), sharex="row")
for r, (t0, t1) in enumerate(rows8):
    s_, lo_pk, ne_pk, thr, _T = old_new(K8, t0, t1)
    a, z = int(t0 * FS), int(t1 * FS)
    tt = np.arange(a, z) / FS
    ymin, ymax = s_[a:z].min(), s_[a:z].max()
    pad = 0.15 * (ymax - ymin)
    for c, (pks, name) in enumerate(((lo_pk, "OLD"), (ne_pk, "NEW"))):
        ax = axs[r, c]
        ax.plot(tt, s_[a:z], color=SIG, lw=1.6)
        if len(pks):
            ax.plot(pks / FS, s_[pks], "^", ms=9, color=BLUE, zorder=4)
        ax.set_ylim(ymin - pad, ymax + 2.2 * pad)
        ax.set_xlim(t0, t1)
        ax.text(0.99, 0.96, f"n = {len(pks)}", transform=ax.transAxes, ha="right", va="top", fontsize=18,
                fontweight="bold", color=BLUE if len(pks) else SIG)
        if c == 0:
            ax.set_ylabel("ADC units", fontsize=14)
            ax.text(0.01, 0.96, f"cutoff {thr:.0f} (off scale)", transform=ax.transAxes, fontsize=11,
                    color="#555555", va="top")
        if r == 1:
            ax.set_xlabel("time (s)", fontsize=14)
        tag(ax, f"BLA cohort 2 · 11_3 · {t0:.1f}-{t1:.1f} s")
    FACTS[f"s8_{t0}"] = dict(old=int(len(lo_pk)), new=int(len(ne_pk)), cutoff=float(thr),
                             ymin=float(ymin), ymax=float(ymax))
fig.tight_layout(h_pad=1.4, w_pad=2.5)
save(fig, "s8_old_vs_new", dataset="BLA cohort 2 (= overall cohort 3), cagemate, Section 1 windows 1.1 and 1.2",
     recording=K8, source=f"{wb.COHORT3_DIR}/{K8}.h5 via load_rank_resp_signal_interactions -> workbench_cache/cohort3_{K8}.npz",
     interval="300.0-301.5 s and 550.0-551.5 s (index time)", fs="400 Hz",
     stage="same 400 Hz input trace on both sides; OLD = bmObject(...,'rodentAirflow').estimateAllFeatures() inhalePeaks "
           "(cached workbench_cache/cohort3_*_legacyfit.npz); NEW = notebook detect_breaths final peaks; markers at input-signal values",
     algorithm="OLD legacy BreathMetrics port (post-2026-09-30 fixes); NEW " + DETECTOR_VERSION)

s_ = cohort3_signal(K8)
FACTS["s8_11_3_absmax"] = float(np.max(np.abs(s_)))

# supplementary Section 1 panels (external figures)
SUP = [("s8_supp_11_3_442", K8, 442.0, 443.5, "1.3 · 11_3 · small breaths on drift"),
       ("s8_supp_10_3_control", "10_3_s_CM_10_2_d_20260729_174121", 50.0, 51.5, "1.4 · 10_3 · control"),
       ("s8_supp_11_2_dropout", "11_2_s_CM_11_4_d_20260729_172306", 490.0, 491.5, "1.5 · 11_2 · signal loss"),
       ("s8_supp_11_2_after", "11_2_s_CM_11_4_d_20260729_172306", 521.0, 522.5, "1.6 · 11_2 · after dropout"),
       ("s8_supp_11_2_flaw", "11_2_s_CM_11_4_d_20260729_172306", 499.8, 501.3, "1.7 · 11_2 · ripples NEW counts")]
for name, key, t0, t1, lab in SUP:
    s2_, lo_pk, ne_pk, thr, T_ = old_new(key, t0, t1)
    a, z = int(t0 * FS), int(t1 * FS)
    tt = np.arange(a, z) / FS
    fig, axs = plt.subplots(1, 2, figsize=(12.2, 2.6), sharey=True)
    for ax, (pks, nm) in zip(axs, ((lo_pk, "OLD · BreathMetrics"), (ne_pk, "NEW · adaptive detector"))):
        inv = ~T_["valid"][a:z]
        if nm.startswith("NEW") and inv.any():
            ax.fill_between(tt, s2_[a:z].min(), s2_[a:z].max(), where=inv, color=RED, alpha=0.2, lw=0, step="mid")
        ax.plot(tt, s2_[a:z], color=SIG, lw=1.5)
        if len(pks):
            ax.plot(pks / FS, s2_[pks], "^", ms=8, color=BLUE)
        ax.set_title(f"{nm}   n = {len(pks)}", fontsize=14)
        ax.set_xlabel("time (s)", fontsize=13)
        ax.set_xlim(t0, t1)
    axs[0].set_ylabel("ADC units", fontsize=13)
    tag(axs[1], f"BLA cohort 2 · Section {lab}")
    fig.tight_layout()
    FACTS[f"supp_{name}"] = dict(old=int(len(lo_pk)), new=int(len(ne_pk)))
    save(fig, name, dataset="BLA cohort 2 (overall cohort 3) cagemate, Section 1 " + lab.split(" ")[0], recording=key,
         source=f"workbench_cache/cohort3_{key}.npz", interval=f"{t0}-{t1} s", fs="400 Hz",
         stage="same input on both sides; red = detector-invalid (NEW side)", algorithm="OLD legacy; NEW " + DETECTOR_VERSION)

# ================================================================== aggregate (slide 8 strip)
FACTS["aggregate_note"] = "recomputed in explore_examples.py; matches notebook cell 28 saved output"
with open(os.path.join(OUT, "facts.json"), "w") as f:
    json.dump(FACTS, f, indent=1, default=str)
with open(os.path.join(OUT, "manifest.txt"), "w", encoding="utf-8") as f:
    f.write("FIGURE PROVENANCE MANIFEST - generated by make_figures.py\n")
    f.write(f"Detector: {DETECTOR_VERSION}\nFrozen QC: workbench_support.bad_signal_mask {QC_VERSION}\n")
    f.write("400 Hz signals: resp_helper_functions loaders (20 kHz -> butter(4,200 Hz) filtfilt -> resample_poly(1,50) -> "
            "nk.signal_filter 0.1-20 Hz order 2); time axis = sample index / fs (no Trodes timestamps)\n\n")
    for m in MANIFEST:
        for k in ["file", "dataset", "recording", "source", "interval", "fs", "stage", "algorithm"]:
            if k in m:
                f.write(f"{k:10s}: {m[k]}\n")
        f.write("\n")
print(json.dumps(FACTS, indent=1, default=str))
