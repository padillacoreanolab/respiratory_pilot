"""Reference-deck figures: annotated full-session traces + 4-panel good/bad teaching figure."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
FS = 400
INK, MUTED, LINE, FLAG = "#1F2937", "#6B7280", "#1D4E89", "#F2C9A0"
plt.rcParams.update({"font.family": "Calibri", "font.size": 11, "axes.edgecolor": "#9CA3AF",
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})

sig = np.load(os.path.join(HERE, "processed_400hz.npz"))
win = pd.read_csv(os.path.join(HERE, "qc_windows_labeled.csv"))
ses = pd.read_csv(os.path.join(HERE, "qc_sessions.csv")).set_index("session")
LABEL = {"Good": "Good", "Somewhat bad": "Somewhat bad (review)", "Bad": "Bad"}
order = {"Good": 0, "Somewhat bad": 1, "Bad": 2}
names = sorted(ses.index, key=lambda n: (order[ses.loc[n, "category"]], n))
NOTE = {
    "10_2_d": "loses amplitude after ~5 min",
    "11_3_s": "small breaths + slow drift + big artifacts",
    "14_1_d": "weak through most of the session",
    "11_2_s": "mainly artifacts",
    "14_3_s": "passes, but amplitude jumps at ~6.7 min: check",
}


def mouse(n):
    return n.split("_CM")[0]


def short(n):
    hhmm = n.rsplit("_", 1)[1]
    return f"{n.rsplit('_', 2)[0]} ({hhmm[:2]}:{hhmm[2:4]})"


# ---------- annotated full-session traces ----------
ref = np.median([np.percentile(np.abs(sig[n]), 99.5) for n in names])
fig, axes = plt.subplots(6, 2, figsize=(13.0, 6.4), dpi=200, sharex=True)
for c, col in enumerate([names[:6], names[6:]]):
    for r in range(6):
        ax = axes[r, c]
        if r >= len(col):
            ax.axis("off"); continue
        n = col[r]; x = sig[n]; t = np.arange(x.size) / FS / 60
        for _, w in win[(win.session == n) & (~win.good)].iterrows():
            ax.axvspan(w.t0 / 60, (w.t0 + 10) / 60, color=FLAG, lw=0, zorder=0)
        ax.plot(t[::2], x[::2], color=LINE, lw=0.3, zorder=1)
        ax.set_ylim(-1.3 * ref, 1.3 * ref); ax.set_yticks([])
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        cat = ses.loc[n, "category"]
        note = NOTE.get(mouse(n), "")
        ax.set_title(f"{short(n)}   {LABEL[cat]}, {ses.loc[n, 'pct_good']:.0f}%" + (f"  ·  {note}" if note else ""),
                     loc="left", fontsize=11, color=INK, fontweight="bold", pad=2)
for ax in axes.flat:
    ax.set_xlim(0, 12)
for ax in (axes[5, 0], axes[4, 1]):
    ax.tick_params(labelbottom=True); ax.set_xlabel("Time from respiration start (min)", fontsize=11)
fig.legend(handles=[Patch(color=FLAG, label="Flagged 10-s window")], loc="lower right",
           bbox_to_anchor=(0.97, 0.06), frameon=False, fontsize=11)
fig.supylabel("Respiration amplitude (a.u., same scale all panels)", fontsize=11, color=INK)
fig.tight_layout(h_pad=0.5, w_pad=2.0)
fig.savefig(os.path.join(HERE, "traces_qc_annot.png"), facecolor="white")
plt.close(fig)

# ---------- 4-panel teaching figure ----------
ex = [("Good", "10_3_s_CM_10_2_d_20260729_174121", 300, 3, "#2E7D4F"),
      ("Bad: very weak, almost flat (10_2_d)", "10_2_d_CM_10_1_s_20260729_114455", 400, 3, "#B42318"),
      ("Bad: breaths buried in slow drift (11_3_s)", "11_3_s_CM_11_4_d_20260729_122040", 120, 3, "#B42318"),
      ("Bad: large movement/contact artifact (13_1_s)", "13_1_s_CM_13_2_d_20260729_170124", 505, 12, "#B42318")]
fig, axes = plt.subplots(4, 1, figsize=(7.0, 5.8), dpi=200)
ylim = None
for i, (ax, (lab, n, t0, dur, col)) in enumerate(zip(axes, ex)):
    x = sig[n][int(t0 * FS):int((t0 + dur) * FS)]
    ax.plot(np.arange(x.size) / FS, x, color=LINE, lw=1.0 if dur <= 3 else 0.6)
    ax.set_title(lab, loc="left", fontsize=13, fontweight="bold", color=col, pad=2)
    if i == 0:
        ylim = (x.min() * 1.15, x.max() * 1.15)
    if i < 3:
        ax.set_ylim(*ylim)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_yticks([])
    ax.set_xlabel(f"{dur} s excerpt" + ("  (y-axis zoomed out ~15x)" if i == 3 else ""), fontsize=10, color=MUTED, labelpad=1)
fig.tight_layout(h_pad=0.4)
fig.savefig(os.path.join(HERE, "qc_teach.png"), facecolor="white")
plt.close(fig)
print("ok")
