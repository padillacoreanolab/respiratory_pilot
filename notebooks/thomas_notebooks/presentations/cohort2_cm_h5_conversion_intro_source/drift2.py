import os, glob, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
INK, MUTED, OLD = "#1F2937", "#6B7280", "#B45A1C"
plt.rcParams.update({"font.family": "Calibri", "font.size": 12, "axes.edgecolor": "#9CA3AF", "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})
def load(p, dt):
    raw = open(p, "rb").read(); i = raw.find(b"<End settings>") + len(b"<End settings>")
    while raw[i:i + 1] in (b"\n", b"\r"): i += 1
    return np.frombuffer(raw[i:], dt)
S = r"C:\Users\thoma\UFL Dropbox\Thomas Heeps\Padilla-Coreano Lab\2025\ECG_cohort1\Aim1\AIM1\Day1_new\CM interactions\Baseline cagemate interactions\CM_s1_1_d1_2_20250623_111352.rec"
ts = load(glob.glob(os.path.join(S, "*merged.analog", "*timestamps.dat"))[0], np.uint32).astype(np.int64)
label_t = np.arange(ts.size) / 20000.0           # old: time label = sample number / 20000
true_t = (ts - ts[0]) / 20000.0                   # true Trodes time of that sample
err = true_t - label_t                            # respiration read at label t comes from true time t+err
fig, ax = plt.subplots(figsize=(6.2, 3.6), dpi=200)
ax.plot(label_t[::2000] / 60, err[::2000], color=OLD, lw=2.2)
ax.axhline(0, color="#2E7D4F", lw=1)
ax.set_xlabel("Time in session (min)", fontsize=12); ax.set_ylabel("Old error (s)", fontsize=12)
ax.set_xlim(0, label_t[-1] / 60); ax.set_ylim(-1.5, 0.1)
ax.grid(axis="y", color="#E5E7EB", lw=0.8); ax.set_axisbelow(True)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
ax.annotate(f"respiration read {abs(err[-1]):.2f} s too early\nby the end", xy=(label_t[-1] / 60, err[-1]), xytext=(1.0, -1.25),
            fontsize=12, fontweight="bold", color=INK, arrowprops=dict(arrowstyle="->", color=MUTED))
ax.set_title("Aim1 CM_s1_1 (no start offset, so this is drift alone)", loc="left", fontsize=11, color=MUTED)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "drift2.png"), facecolor="white")
print("end err", err[-1])
