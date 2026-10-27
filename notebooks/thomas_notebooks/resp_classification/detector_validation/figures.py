"""Figures for the legacy vs adaptive comparison.   python figures.py <out_dir>
Reads <out_dir>/recordings/*.npz and the CSVs written by compare_real.py / sweep_dev.py."""
import glob
import os
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from real_common import FS, match_idx  # noqa: E402

OUT = sys.argv[1]
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)
TOL = int(0.025 * FS)
EXCLUDE = {"RI1_5_13", "RI2_5_13", "BLRI_5_13"}  # documented drops (docs/EXCLUSIONS_LOG.md)
REC = {os.path.basename(p)[:-4]: p for p in sorted(glob.glob(os.path.join(OUT, "recordings", "*.npz")))
       if os.path.basename(p)[:-4] not in EXCLUDE}
REASON_COL = {"flat_or_dropout": "#bbbbbb", "artifact": "#f4a6a6", "missing": "#d9c3f0", "short_segment": "#e6e6e6"}


def load(key):
    d = dict(np.load(REC[key], allow_pickle=False))
    d["la"] = match_idx(d["peaks_l"], d["peaks_a"], TOL)      # legacy peak also adaptive
    d["al"] = match_idx(d["peaks_a"], d["peaks_l"], TOL)
    return d


def panel(ax, key, d, s0, s1, title_extra=""):
    tt = np.arange(s0, s1) / FS
    y = d["smoothed"][s0:s1]
    ax.plot(tt, y, color="#167fa0", lw=0.9, label="signal (smoothed)")
    ax.plot(tt, d["baseline"][s0:s1], color="0.45", lw=0.8, ls="--", label="adaptive local baseline")
    # invalid shading
    reason = d["reason"][s0:s1]
    for r, colr in REASON_COL.items():
        m = reason == r
        if m.any():
            ax.fill_between(tt, np.nanmin(y) if np.isfinite(y).any() else 0, np.nanmax(y) if np.isfinite(y).any() else 1,
                            where=m, color=colr, alpha=0.5, step="mid", label=f"invalid: {r}")
    if not np.isfinite(y).any():   # invalid everywhere: show raw signal instead
        ax.plot(tt, d["sig"][s0:s1], color="0.5", lw=0.7)
        y = d["sig"][s0:s1]
    yv = lambda p: d["smoothed"][p] if np.isfinite(d["smoothed"][p]) else d["sig"][p]  # noqa: E731
    pa, pl = d["peaks_a"], d["peaks_l"]
    ina, inl = (pa >= s0) & (pa < s1), (pl >= s0) & (pl < s1)
    ax.plot(pl[inl] / FS, [yv(p) for p in pl[inl]], "o", mfc="none", mec="green", ms=10, mew=1.3, label="legacy peak")
    ax.plot(pa[ina] / FS, [yv(p) for p in pa[ina]], "^", color="#1f3fbf", ms=5, label="adaptive peak")
    tr = d["troughs_a"]
    tr = tr[np.isfinite(tr)].astype(int)
    tr = tr[(tr >= s0) & (tr < s1)]
    ax.plot(tr / FS, [yv(p) for p in tr], "v", color="#7a52c7", ms=4, label="adaptive trough")
    on = d["onsets_a"]
    on = on[np.isfinite(on)].astype(int)
    on = on[(on >= s0) & (on < s1)]
    ax.vlines(on / FS, *np.nanpercentile(y, [5, 95]) if np.isfinite(y).any() else (0, 1), color="#1f3fbf", lw=0.5,
              alpha=0.5, label="adaptive inhale onset")
    lo = pl[inl & ~d["la"]]
    ao = pa[ina & ~d["al"]]
    ax.plot(lo / FS, [yv(p) for p in lo], "o", mfc="none", mec="red", ms=16, mew=2, label="legacy only")
    ax.plot(ao / FS, [yv(p) for p in ao], "s", mfc="none", mec="darkorange", ms=14, mew=2, label="adaptive only")
    ax.set_title(f"{key}  {s0 / FS:.1f}-{s1 / FS:.1f} s  | legacy {inl.sum()}, adaptive {ina.sum()} peaks "
                 f"(legacy-only {len(lo)}, adaptive-only {len(ao)}){title_extra}", fontsize=8.5)
    ax.tick_params(labelsize=7)


def grid(items, fname, suptitle, ncols=2, h=2.6):
    nrows = int(np.ceil(len(items) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(8.5 * ncols, h * nrows), layout="constrained", squeeze=False)
    for ax, (key, s0, s1, extra) in zip(axes.ravel(), items):
        panel(ax, key, load(key), s0, s1, extra)
    for ax in axes.ravel()[len(items):]:
        ax.axis("off")
    h_, l_ = [], []
    for ax in axes.ravel():
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l_:
                h_.append(hh)
                l_.append(ll)
    fig.legend(h_, l_, loc="upper center", ncol=6, fontsize=8, bbox_to_anchor=(0.5, 1.0 + 0.6 / nrows / h * 2))
    fig.suptitle(suptitle, fontsize=10, y=1.0 + 1.2 / nrows / h * 2)
    fig.savefig(os.path.join(FIG, fname), dpi=80, bbox_inches="tight")
    plt.close(fig)


rng = np.random.default_rng(2026)
W = int(3 * FS)
random_items, hard_items, invalid_items = [], [], []
for key in REC:
    d = load(key)
    n = len(d["sig"])
    ok = np.where(d["evalmask"] & d["valid"])[0]
    starts = [s for s in range(0, n - W, W) if d["evalmask"][s:s + W].all() and d["valid"][s:s + W].all()]
    if key.startswith("CM_"):
        for s in rng.choice(starts, 2, replace=False):
            random_items.append((key, int(s), int(s) + W, ""))
    elif len(random_items) < 30 and rng.random() < 0.35:
        s = int(rng.choice(starts))
        random_items.append((key, s, s + W, ""))
    dis = np.r_[d["peaks_l"][~d["la"]], d["peaks_a"][~d["al"]]]
    dis = dis[d["evalmask"][dis]]
    if len(dis):
        c = np.bincount(dis // W, minlength=n // W + 1)
        b = int(np.argmax(c))
        hard_items.append((key, b * W, min(b * W + W, n), f"  [{c[b]} disagreements]", c[b]))
    inv = ~d["valid"]
    if inv.any() and len(invalid_items) < 8:
        runs = np.flatnonzero(np.diff(np.r_[0, inv.astype(int), 0]))
        a, z = runs[0], runs[1]
        mid = (a + z) // 2 if z - a < 6 * FS else a
        s0 = max(0, mid - int(2 * FS))
        invalid_items.append((key, s0, min(n, s0 + int(5 * FS)), f"  [{d['reason'][a]}]"))

grid(random_items, "R1_random_intervals.png",
     "RANDOM 3 s intervals (seeded; rank = evaluation minutes only; valence = held-out recordings)")
hard_items = sorted(hard_items, key=lambda x: -x[4])[:14]
grid([h[:4] for h in hard_items], "R2_hardest_intervals.png",
     "HARDEST 3 s interval per recording (most legacy/adaptive disagreements; worst case, not typical)")
if invalid_items:
    grid(invalid_items, "R3_invalid_intervals.png", "Intervals marked invalid by the adaptive detector (shaded)")

# amplitude strata
amp = pd.read_csv(os.path.join(OUT, "detection_by_amplitude.csv"))
amp = amp[~amp.key.isin(EXCLUDE)]
amp["cohort"] = np.where(amp.key.str.startswith("CM_"), "rank (eval minutes)", "valence (held out)")
g = amp.groupby(["cohort", "stratum"])[["legacy", "adaptive", "legacy_only", "adaptive_only"]].sum()
order = ["<0.25x", "0.25-0.5x", "0.5-1x", ">1x"]
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), layout="constrained")
for ax, coh in zip(axes, g.index.levels[0]):
    gg = g.loc[coh].reindex(order)
    xx = np.arange(len(order))
    ax.bar(xx - 0.2, gg.legacy, 0.4, label="legacy", color="green", alpha=.7)
    ax.bar(xx + 0.2, gg.adaptive, 0.4, label="adaptive", color="#1f3fbf", alpha=.7)
    for i, (lg, ad) in enumerate(zip(gg.legacy, gg.adaptive)):
        ax.text(i, max(lg, ad) * 1.02, f"{100 * (ad - lg) / max(lg, 1):+.0f}%", ha="center", fontsize=8)
    ax.set_xticks(xx, order)
    ax.set(xlabel="local amplitude at the peak (relative to recording median)", ylabel="peaks detected", title=coh)
    ax.legend()
fig.savefig(os.path.join(FIG, "R4_detections_by_amplitude.png"), dpi=90)
plt.close(fig)

# ACF agreement
acf = pd.read_csv(os.path.join(OUT, "acf_windows.csv"))
acf = acf[~acf.key.isin(EXCLUDE)]
acf = acf[acf.eval_window & (acf.qc_invalid_frac == 0) & acf.acf_rate.notna()]
fig, axes = plt.subplots(1, 2, figsize=(11, 5), layout="constrained", sharey=True)
for ax, col, nm in [(axes[0], "rate_legacy", "legacy"), (axes[1], "rate_adaptive", "adaptive")]:
    e = (acf[col] - acf.acf_rate) / acf.acf_rate
    ax.scatter(acf[col], acf.acf_rate, s=5, alpha=.3)
    ax.plot([0, 16], [0, 16], "k--", lw=1)
    ax.set(xlim=(0, 16), ylim=(0, 16), xlabel=f"{nm} window rate (Hz)", ylabel="autocorrelation rate (Hz)",
           title=f"{nm}: median error {100 * e.median():+.1f}%, within 15%: {100 * (e.abs() < .15).mean():.0f}%")
fig.suptitle("Window breathing rate vs an independent, peak-free estimate (evaluation windows, no invalid samples)")
fig.savefig(os.path.join(FIG, "R5_acf_rate_agreement.png"), dpi=90)
plt.close(fig)

# feature changes
fw = pd.read_csv(os.path.join(OUT, "features_windows.csv"))
fw = fw[~fw.key.isin(EXCLUDE)]
fw = fw[fw.eval_window]
key_feats = ["breathing_rate_hz", "cv_ibi", "mean_inhale_dur_sec", "mean_exhale_dur_sec", "mean_peak_insp_flow",
             "mean_inhale_vol", "pct_breaths_with_exhale_pause", "mean_exhale_pause_dur_sec"]
fig, axes = plt.subplots(2, 4, figsize=(17, 8.5), layout="constrained")
for ax, f in zip(axes.ravel(), key_feats):
    a, b = fw[f"{f}__legacy"], fw[f"{f}__adaptive"]
    ok = a.notna() & b.notna()
    ax.scatter(a[ok], b[ok], s=4, alpha=.25)
    lim = np.nanpercentile(np.r_[a[ok], b[ok]], [1, 99]) if ok.any() else (0, 1)
    ax.plot(lim, lim, "k--", lw=1)
    ax.set(xlim=lim, ylim=lim, xlabel="legacy", ylabel="adaptive",
           title=f"{f}\nSpearman {a[ok].corr(b[ok], method='spearman'):.2f}, n={ok.sum()}")
fig.suptitle("Per-window features, legacy vs adaptive (same windows; evaluation minutes + valence)")
fig.savefig(os.path.join(FIG, "R6_feature_changes.png"), dpi=85)
plt.close(fig)

# sweep sensitivity
sw = pd.read_csv(os.path.join(OUT, "sweep_dev_summary.csv"))
core = sw.iloc[:20]
fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")
for ax, m in zip(axes, ["amp_recall", "doubling", "halving"]):
    pv = core.pivot(index="prominence_frac", columns="min_peak_distance_s", values=m)
    im = ax.imshow(pv.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(pv.shape[1]), pv.columns)
    ax.set_yticks(range(pv.shape[0]), pv.index)
    for i in range(pv.shape[0]):
        for j in range(pv.shape[1]):
            ax.text(j, i, f"{pv.values[i, j]:.3f}", ha="center", va="center", color="w", fontsize=7)
    ax.set(xlabel="min peak distance (s)", ylabel="prominence fraction", title=f"dev data: {m}")
fig.savefig(os.path.join(FIG, "R7_parameter_sensitivity.png"), dpi=90)
plt.close(fig)
print("figures written to", FIG)
