import os, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
INK, MUTED, OLD, NEW = "#1F2937", "#6B7280", "#B45A1C", "#1D4E89"
plt.rcParams.update({"font.family": "Calibri", "font.size": 12, "axes.edgecolor": "#9CA3AF",
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})
ev = pd.read_csv(os.path.join(HERE, "misalign_events.csv"))
BAD = "CM_s4_7_d4_8_20250623_193718"

# ---------- 1. per-event error ----------
fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=200)
ax.axhspan(-0.05, 0.05, color="#DDEFE3", zorder=0)
ax.axhline(0, color="#2E7D4F", lw=1.2, zorder=1)
ends = []
for s, g in ev[ev.session != BAD].dropna(subset=["err"]).groupby("session"):
    g = g.sort_values("t_true")
    ax.plot(g.t_true / 60, g.err, "-o", color=OLD, alpha=0.75, ms=3.5, lw=1)
    ends.append((g.err.iloc[-1], g.t_true.iloc[-1] / 60, s.split("_2025")[0]))
ends.sort()
prev = -9
for k, (y, x, lab) in enumerate(ends):
    yl = max(y, prev + 0.11); prev = yl
    ax.annotate(lab, (x, y), xytext=(10.05, yl), fontsize=9, color=MUTED, va="center", arrowprops=dict(arrowstyle="-", color="#D1D5DB", lw=0.6))
ax.set_xlim(0, 12.2); ax.set_ylim(-1.5, 0.7)
ax.set_xlabel("Time of BORIS event in session (min)")
ax.set_ylabel("Old-pipeline error (s)")
ax.text(0.15, 0.55, "respiration read too LATE (start offset)", fontsize=11, color=INK, fontweight="bold")
ax.text(0.15, -1.42, "respiration read too EARLY (copied samples)", fontsize=11, color=INK, fontweight="bold")
ax.text(6.4, 0.12, "corrected pipeline: 0", fontsize=10, color="#2E7D4F")
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
bad = ev[ev.session == BAD]
ax.set_title(f"{int(ev[ev.session != BAD].err.notna().sum())} subject sniff onsets, 6 Aim1 baseline cagemate sessions (CM_s4_7 off the chart: 45–125 s)", loc="left", fontsize=11, color=MUTED)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "misalign_events.png"), facecolor="white"); plt.close(fig)

# ---------- 2. sniff-onset breathing rate ----------
d = np.load(os.path.join(HERE, "misalign_eta.npz"))
lag = d["lag"]
fig, ax = plt.subplots(figsize=(6.2, 4.4), dpi=200)
for key, col, lab in (("new", NEW, "Corrected (frame → Trodes tick)"), ("old", OLD, "Old (BORIS s = sample ÷ 20,000)")):
    m = d[key][d[key + "_sess"] != BAD]
    mu = np.nanmean(m, 0); se = np.nanstd(m, 0) / np.sqrt(np.sum(~np.isnan(m), 0))
    ax.fill_between(lag, mu - se, mu + se, color=col, alpha=0.18, lw=0)
    ax.plot(lag, mu, color=col, lw=2.2, label=f"{lab}, n={m.shape[0]}")
ax.axvline(0, color=INK, lw=1, ls="--")
ax.text(0.1, ax.get_ylim()[1], "BORIS sniff onset", fontsize=10, color=INK, va="top")
ax.set_title("Subject sniff onsets, 6 Aim1 sessions (CM_s4_7, 23 fps mp4, excluded) · mean ± SEM", loc="left", fontsize=11, color=MUTED)
ax.set_xlabel("Time from BORIS sniff onset (s)"); ax.set_ylabel("Breathing rate (Hz)")
ax.legend(frameon=False, fontsize=10, loc="lower right")
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "misalign_eta.png"), facecolor="white"); plt.close(fig)
print(bad[["t_video", "t_true", "err"]].describe().round(1))
EOF_MARKER = None
