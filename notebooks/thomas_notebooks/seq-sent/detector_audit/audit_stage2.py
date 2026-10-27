"""Stage 2 of the breath-detector audit: aggregate stage-1 outputs, make figures,
and compare the rank classifier on original vs corrected features.

  python audit_stage2.py <rank_features_orig.csv> <results_dir>
"""
import os
import sys
import warnings

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import balanced_accuracy_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from audit_stage1 import FEATURE_COLS, FS, MAIN_FRAC  # noqa: E402

warnings.filterwarnings("ignore")
orig_csv, R = sys.argv[1], sys.argv[2]
FIG = os.path.join(R, "figures")
os.makedirs(FIG, exist_ok=True)
orig = pd.read_csv(orig_csv)
t = {k: pd.read_csv(os.path.join(R, f"{k}.csv")) for k in
     ["repro", "sens", "gap", "peaks", "win", "corr", "semi", "meta"]}
subj = orig.groupby("Trial")[["Subject", "RankLabel", "PairID"]].first()
out = []


def say(s=""):
    print(s)
    out.append(s)


# ----------------------------------------------------------------------------- 1. reproduction
rp = t["repro"].merge(orig[["Trial", "WindowID", "n_breaths", "breathing_rate_hz"]],
                      left_on=["trial", "WindowID"], right_on=["Trial", "WindowID"])
same = (rp.n_breaths_repro == rp.n_breaths) & np.isclose(rp.rate_repro, rp.breathing_rate_hz)
say(f"## 1. Reproduction of your saved rank features\n{same.sum()}/{len(rp)} windows reproduced exactly "
    f"(n_breaths and breathing_rate_hz identical).")

# ----------------------------------------------------------------------------- 2. how many missed
s = t["sens"]
say("\n## 2. Reference peaks (live signal) not found by BreathMetrics, by reference prominence setting")
agg = s.groupby("frac")[["n_ref_live", "n_missed_live", "n_bm_live", "n_bm_only_live"]].sum()
agg["missed_%"] = 100 * agg.n_missed_live / agg.n_ref_live
agg["bm_only_%"] = 100 * agg.n_bm_only_live / agg.n_bm_live
say(agg.round(1).to_string())
main = s[s.frac == MAIN_FRAC].set_index("trial")
main["missed_%"] = 100 * main.n_missed_live / main.n_ref_live
main = main.join(subj)
say(f"\nPer recording at frac={MAIN_FRAC}:\n" + main[["Subject", "RankLabel", "n_ref_live", "n_missed_live", "missed_%"]]
    .round(1).to_string())

# ----------------------------------------------------------------------------- 3. validation tests
p = t["peaks"]
lp = p[p.live]
mi, fo = lp[~lp.found], lp[lp.found]
say("\n## 3. Are the missed peaks real breaths?")
say(f"Missed peaks below the session threshold (mean+std/2): {100 * (~mi.above_thr).mean():.0f}%  "
    f"(found peaks below it: {100 * (~fo.above_thr).mean():.0f}%)")
say(f"Prominence / noise floor: missed median {mi.snr.median():.0f}x (IQR {mi.snr.quantile(.25):.0f}-"
    f"{mi.snr.quantile(.75):.0f}), found median {fo.snr.median():.0f}x")
say(f"Half-prominence width: missed median {1000 * mi.width_s.median():.0f} ms, found median "
    f"{1000 * fo.width_s.median():.0f} ms")

g = t["gap"]
g0, g1, gk = g[g.k == 0], g[g.k == 1], g[g.k >= 1]
split = gk.R / (gk.k + 1)
say(f"\nGap test ({len(g)} BreathMetrics intervals with clean neighbours):")
say(f"  intervals with NO missed peak inside (k=0): n={len(g0)}, length / neighbouring IBI median "
    f"{g0.R.median():.2f} (IQR {g0.R.quantile(.25):.2f}-{g0.R.quantile(.75):.2f})")
say(f"  intervals WITH k>=1 missed peaks: n={len(gk)}, raw ratio median {gk.R.median():.2f}; "
    f"ratio/(k+1) median {split.median():.2f} (IQR {split.quantile(.25):.2f}-{split.quantile(.75):.2f}); "
    f"{100 * split.between(.75, 1.33).mean():.0f}% within 0.75-1.33")
say(f"  k=1 intervals: position of missed peak in interval median {g1.f.median():.2f} "
    f"(IQR {g1.f.quantile(.25):.2f}-{g1.f.quantile(.75):.2f}); 0.5 = exact midpoint")

w = t["win"].merge(orig[["Trial", "WindowID", "breathing_rate_hz"]], left_on=["trial", "WindowID"],
                   right_on=["Trial", "WindowID"])
w = w.merge(t["corr"][["trial", "WindowID", "breathing_rate_hz"]].rename(columns={"breathing_rate_hz": "rate_corr"}),
            on=["trial", "WindowID"])
wa = w[(w.live_frac == 1) & w.acf_rate.notna()]


def agree(col):
    e = (wa[col] - wa.acf_rate) / wa.acf_rate
    return f"median error {100 * e.median():+.0f}%, within 15% of ACF in {100 * (e.abs() < .15).mean():.0f}% of windows"


say(f"\nAutocorrelation rate check ({len(wa)} fully-live windows with a periodic ACF):")
say(f"  your BreathMetrics rate: {agree('breathing_rate_hz')}")
say(f"  corrected-detector rate: {agree('rate_corr')}")

sm = t["semi"]
say("\nSemi-synthetic test (real 60 s stretch scaled down, whole session refit; truth = peaks at full scale):")
say(sm.round(3).to_string(index=False))

say("\n## 4. Per-window refit (BreathMetrics fit on each 3 s window alone)")
wr = t["win"]
say(f"Recall of reference peaks: session fit {100 * wr.ref_recall_session.mean():.0f}%, "
    f"window refit {100 * wr.ref_recall_window_refit.mean():.0f}%")
if "window_refit_error" in wr:
    say(f"Window refits that crashed: {wr.window_refit_error.notna().sum()}/{len(wr)}")
say(f"Window refit peaks with no reference peak (extra): mean {wr.window_refit_extra.mean():.2f} per window")

# ----------------------------------------------------------------------------- 5. impact on features
c = t["corr"].rename(columns={"trial": "Trial"}).merge(orig, on=["Trial", "WindowID"], suffixes=("_c", ""))
say("\n## 5. Impact on classifier features (same 689 windows)")
rows = []
for f in FEATURE_COLS:
    a, b = c[f], c[f + "_c"]
    ok = a.notna() & b.notna()
    mo = c.groupby("Subject").agg(o=(f, "mean"), n=(f + "_c", "mean"), r=("RankLabel", "first"))
    def d(col):
        dom, sub = mo[mo.r == "Dominant"][col], mo[mo.r == "Subordinate"][col]
        sd = np.sqrt((dom.var() + sub.var()) / 2)
        return (dom.mean() - sub.mean()) / sd if sd > 0 else np.nan
    rows.append(dict(feature=f, spearman=a[ok].corr(b[ok], method="spearman"),
                     median_change_pct=100 * np.nanmedian((b[ok] - a[ok]) / a[ok].abs().replace(0, np.nan)),
                     dom_minus_sub_d_orig=d("o"), dom_minus_sub_d_corr=d("n")))
imp = pd.DataFrame(rows).set_index("feature")
say(imp.round(2).to_string())

say("\nPer mouse (mean over windows):")
pm = c.groupby(["Subject", "RankLabel"]).agg(rate_orig=("breathing_rate_hz", "mean"), rate_corr=("breathing_rate_hz_c", "mean"),
                                             n_orig=("n_breaths", "mean"), n_corr=("n_breaths_c", "mean"))
pm["rate_underestimate_%"] = 100 * (pm.rate_corr - pm.rate_orig) / pm.rate_corr
pm = pm.join(main.set_index("Subject")[["missed_%"]], on="Subject")
say(pm.round(2).to_string())

# ----------------------------------------------------------------------------- 6. classifier A/B
say("\n## 6. Rank classifier, original vs corrected features (identical model and folds)")
say("Leave-one-cage-out (PairID). Pooled out-of-fold balanced accuracy; chance = 0.50.")
y = (c.RankLabel == "Dominant").astype(int).values
cages = c.PairID.values
models = {
    "LogReg": lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                    LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)),
    "RandomForest": lambda: make_pipeline(SimpleImputer(strategy="median"),
                                          RandomForestClassifier(300, class_weight="balanced", random_state=0, n_jobs=-1)),
}
TIMING = ["n_breaths", "breathing_rate_hz", "mean_ibi_sec", "cv_ibi"]
from itertools import combinations
mice = c.groupby("Subject").RankLabel.first()


def loco(X, yy, mk):
    pred = np.zeros_like(yy)
    for cg in np.unique(cages):
        te = cages == cg
        pred[te] = mk().fit(X[~te], yy[~te]).predict(X[te])
    return pred


for fset, base in [("all 27 features", FEATURE_COLS), ("timing only (validated)", TIMING)]:
    say(f"\n  Feature set: {fset}")
    for label, cols in [("original", base), ("corrected", [f + "_c" for f in base])]:
        X = c[cols].replace([np.inf, -np.inf], np.nan).values
        for name, mk in models.items():
            pred = loco(X, y, mk)
            rec = pd.DataFrame(dict(s=c.Subject, y=y, p=pred)).groupby("s").agg(y=("y", "first"), p=("p", "mean"))
            ba = balanced_accuracy_score(y, pred)
            extra = ""
            if name == "LogReg":
                # exact permutation test: every way of labelling 3 of the 7 mice Dominant (35 relabelings)
                null = []
                for dom in combinations(mice.index, 3):
                    yy = c.Subject.isin(dom).astype(int).values
                    null.append(balanced_accuracy_score(yy, loco(X, yy, mk)))
                extra = f"   permutation p={np.mean(np.array(null) >= ba - 1e-12):.2f} (35 mouse relabelings)"
            say(f"  {label:9s} {name:12s} window bal_acc={ba:.3f}   "
                f"mice correct by majority vote={int(((rec.p > .5).astype(int) == rec.y).sum())}/{len(rec)}{extra}")

with open(os.path.join(R, "summary.md"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(out))

# ----------------------------------------------------------------------------- figures
npz = {tr: np.load(os.path.join(R, f"{tr}.npz")) for tr in subj.index}

# F1: randomly sampled (not cherry-picked) missed peaks, all mice pooled
rng = np.random.default_rng(7)
pool = mi[(mi.idx > FS) & (mi.idx < mi.idx.max() - FS)].sample(12, random_state=7)
fig, axes = plt.subplots(6, 2, figsize=(16, 17), layout="constrained")
for ax, (_, r) in zip(axes.ravel(), pool.iterrows()):
    d = npz[r.trial]
    x, bm_pk, ref, found = d["x"], d["bm_pk"], d["ref"], d["found"]
    s0, s1 = int(r.idx - 0.75 * FS), int(r.idx + 0.75 * FS)
    tt = np.arange(s0, s1) / FS
    ax.plot(tt, x[s0:s1], color="#167fa0", lw=1)
    ax.axhline(float(d["thr"]), color="red", ls="--", lw=.8)
    b = bm_pk[(bm_pk >= s0) & (bm_pk < s1)]
    m = ref[~found]
    m = m[(m >= s0) & (m < s1)]
    ax.plot(b / FS, x[b], "o", mfc="none", mec="green", ms=9, mew=1.5)
    ax.plot(m / FS, x[m], "x", color="darkorange", ms=10, mew=2.5)
    ax.plot(r.idx / FS, x[int(r.idx)], "s", mfc="none", mec="black", ms=16)
    allp = np.sort(np.r_[b, m])
    for a0, a1 in zip(allp[:-1], allp[1:]):
        ax.annotate(f"{1000 * (a1 - a0) / FS:.0f}", ((a0 + a1) / 2 / FS, ax.get_ylim()[0]), fontsize=7,
                    ha="center", va="bottom", color="gray")
    ax.set_title(f"{subj.loc[r.trial, 'Subject']} ({subj.loc[r.trial, 'RankLabel'][:3]})  t={r.idx / FS:.2f}s  "
                 f"prominence={r.prom:.0f} ADC = {r.snr:.0f}x noise", fontsize=9)
fig.suptitle("12 RANDOMLY sampled peaks that BreathMetrics missed (black square).  green o = BreathMetrics peak, "
             "orange x = missed peak, red dashed = session threshold mean+std/2, gray numbers = inter-peak interval (ms)",
             fontsize=10)
fig.savefig(os.path.join(FIG, "F1_random_missed_examples.png"), dpi=90)

# F2: gap test
fig, ax = plt.subplots(1, 3, figsize=(16, 4.2), layout="constrained")
bins = np.linspace(0, 4, 81)
ax[0].hist(g0.R, bins, alpha=.6, density=True, label="intervals with no missed peak")
ax[0].hist(gk.R, bins, alpha=.6, density=True, label="intervals containing missed peak(s)")
ax[0].set(xlabel="BreathMetrics interval / neighbouring intervals", title="Intervals with missed peaks are ~2x or ~3x long")
ax[0].legend(fontsize=8)
ax[1].hist(split, bins, alpha=.8, color="darkorange", density=True)
ax[1].axvline(1, color="k", ls="--")
ax[1].set(xlabel="interval / neighbours / (k+1)", title="After inserting the k missed peaks: back to ~1")
ax[2].hist(g1.f, np.linspace(0, 1, 41), color="purple", alpha=.8)
ax[2].set(xlabel="position of missed peak inside the long interval (0-1)", title="Single missed peak sits near the midpoint")
fig.savefig(os.path.join(FIG, "F2_gap_test.png"), dpi=90)

# F3: ACF rate check
fig, ax = plt.subplots(1, 2, figsize=(12, 5.5), layout="constrained", sharey=True)
for a, col, name in [(ax[0], "breathing_rate_hz", "Your BreathMetrics rate"), (ax[1], "rate_corr", "Corrected rate")]:
    a.scatter(wa[col], wa.acf_rate, s=8, alpha=.4)
    a.plot([0, 16], [0, 16], "k--", lw=1)
    a.set(xlabel=f"{name} (Hz)", ylabel="Autocorrelation rate (Hz)", xlim=(0, 16), ylim=(0, 16), title=name)
fig.suptitle("Rate from an independent method (autocorrelation) vs each detector, per 3 s window")
fig.savefig(os.path.join(FIG, "F3_acf_rate_check.png"), dpi=90)

# F4: semi-synthetic
if "bm_recall_of_bm_truth" in sm:
    fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
    for tr, gg in sm.groupby("trial"):
        ax.plot(gg.scale, 100 * gg.bm_recall_of_bm_truth, "o-", label=f"BreathMetrics, {subj.loc[tr, 'Subject']}")
        ax.plot(gg.scale, 100 * gg.ref_recall_of_ref_truth, "s--", label=f"prominence, {subj.loc[tr, 'Subject']}")
    ax.set(xlabel="scale applied to a real 60 s stretch", ylabel="% of full-scale breaths still detected",
           title="Same real breaths, just quieter")
    ax.invert_xaxis()
    ax.legend(fontsize=8)
    fig.savefig(os.path.join(FIG, "F4_semi_synthetic.png"), dpi=90)

# F5: one long, continuous stretch per mouse with both detectors
fig, axes = plt.subplots(len(subj), 1, figsize=(16, 2.6 * len(subj)), layout="constrained")
for ax, tr in zip(axes, subj.index):
    d = npz[tr]
    x, bm_pk, ref, found, live = d["x"], d["bm_pk"], d["ref"], d["found"], d["live"]
    miss = ref[~found & live[ref]]
    # 4 s stretch with the most misses
    counts = np.bincount(miss // (4 * FS))
    s0 = int(np.argmax(counts)) * 4 * FS
    s1 = s0 + 4 * FS
    ax.plot(np.arange(s0, s1) / FS, x[s0:s1], color="#167fa0", lw=.9)
    ax.axhline(float(d["thr"]), color="red", ls="--", lw=.8)
    b = bm_pk[(bm_pk >= s0) & (bm_pk < s1)]
    m = miss[(miss >= s0) & (miss < s1)]
    ax.plot(b / FS, x[b], "o", mfc="none", mec="green", ms=8)
    ax.plot(m / FS, x[m], "x", color="darkorange", ms=9, mew=2)
    ax.set_title(f"{subj.loc[tr, 'Subject']} ({subj.loc[tr, 'RankLabel']}): worst 4 s  |  BM {len(b)} peaks, "
                 f"+{len(m)} missed", fontsize=9)
fig.suptitle("Worst 4 s stretch per mouse (selected for most misses, so this is the worst case, not typical)")
fig.savefig(os.path.join(FIG, "F5_worst_stretch_per_mouse.png"), dpi=80)
print("figures written to", FIG)
