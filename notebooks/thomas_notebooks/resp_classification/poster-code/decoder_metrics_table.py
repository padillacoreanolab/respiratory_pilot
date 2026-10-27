"""
Reproducible rank + valence decoder metrics and poster panels.

Runs ONE uniform, reproducible protocol over all four task x window-type feature matrices,
emits a combined metrics TABLE, and renders the best-performing variant of each task as a
Model-Performance + Top-5-Features panel (styled to match the poster mock).

Tasks / matrices (all read from data/, no notebook state needed):
  Valence (Positive=RI1 vs Negative=RI2), CV = LOSO (leave-one-MOUSE-out; valence is a
      within-mouse contrast, so a mouse is the group and there are no cages):
        session      -> raw-breathmetrics-features-session-3sec-subjectnames
        interaction  -> Raw-Breathmatrics-Features
  Rank (Dominant vs Subordinate), Aim1 cagemate cohort, CV = LOCO (leave-one-CAGE-out; both
      cagemates held out together):
        session      -> raw-breathmetrics-features-session-3sec-rank
        interaction  -> Raw-Breathmatrics-Features-Ran-Interactions

Protocol (identical across all datasets so the numbers are comparable):
  * Features   : raw BreathMetrics columns, numeric, <=40% missing (metadata/engineered dropped).
  * Preprocess : median impute (+ standardize for LogReg/SVM). class_weight="balanced".
  * Models     : LogisticRegression (L2), SVM (rbf), RandomForest.
  * CV         : leave-one-GROUP-out (group = mouse for valence, cage for rank). There is NO
                 separate held-out test set — every window is predicted exactly once, out-of-fold,
                 while its whole group is absent from training. "Validation" here = the held-out
                 group in each CV fold; the reported score is the cross-validated generalization to
                 an unseen mouse/cage. Hyper-parameters are fixed (not tuned), so no inner loop.
  * Metrics    : balanced accuracy (macro recall) and macro F1, per fold -> mean +/- std across
                 folds, plus pooled out-of-fold. Chance = 0.5.

Best variant per task is chosen by the highest single-model mean balanced accuracy; only that
variant is drawn as a panel (the table still lists all four).

Outputs:
  data/decoder_metrics_summary.csv        (task x window-type x model)
  data/decoder_metrics_per_fold.csv       (+ held-out group detail)
  poster-plots/decoder-metrics/decoder_performance.{svg,png}  (perf bars: Valence | Rank side by side)
  poster-plots/decoder-metrics/decoder_features.{svg,png}     (top-5 features: Valence | Rank, wide)
  poster-plots/decoder-metrics/NOTES.md    (methodology / train-val-eval notes)

Run:  conda run -n ephys_env python decoder_metrics_table.py   (numpy>=2 needed to read the pickles)
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score, f1_score

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
RC = os.path.abspath(os.path.join(HERE, ".."))
DATA_DIR = os.path.join(RC, "data")
PLOT_DIR = os.path.join(RC, "poster-plots", "decoder-metrics")
os.makedirs(PLOT_DIR, exist_ok=True)

CHANCE = 0.5
MAX_MISSING_FRAC = 0.40
GRAY = "#6B6B6B"
# x/y TICK label font sizes, scaled 1.65x from their bases for poster readability.
PERF_MODEL_TICK_FONTSIZE = 9 * 1.65   # perf-axis x ticks (model names, base 9 pt)
FEAT_TICK_FONTSIZE = 8 * 1.65         # feature-axis y ticks (feature names, base 8 pt)
DEFAULT_TICK_LABELSIZE = 10 * 1.65    # default-sized ticks (perf y, feat x) -> 16.5 pt
# x/y AXIS TITLE font sizes, also scaled 1.65x from their bases.
PERF_YLABEL_FONTSIZE = 10 * 1.65      # "Balanced Accuracy" / "Macro F1" (base 10 pt)
FEAT_XLABEL_FONTSIZE = 9 * 1.65       # "Coefficient" (base 9 pt)

META_BLOCK = {
    "WindowID", "BoutID", "Start", "Stop", "Duration", "PreBoutRespRate", "FeatureWindow",
    "Trial", "Recording", "StandardizedRecording", "Subject", "Cagemate", "social_agent",
    "Initiator", "PairID", "SubjectRankCode", "CagemateRankCode", "SubjectRankName",
    "CagemateRankName", "Condition", "Rank", "RankLabel", "Type", "Cohort", "SourceCohort",
    "Label", "h5_path", "h5_filename", "boris_path", "boris_filename", "Behavior",
    "TrueMouseID", "session_duration_sec", "n_breaths_total", "WindowSec", "TargetSRate",
    "MinBreathsPerWindow", "resp_rate_delta", "breathing_rate_z",
}

FEATURE_NAME_MAP = {
    "n_breaths": "Breath Count", "breathing_rate_hz": "Breathing Rate",
    "mean_ibi_sec": "Time Between Breaths", "cv_ibi": "Breath Interval Variability",
    "cv_breathing_rate": "Breathing Rate Variability", "mean_inhale_dur_sec": "Mean Inhale Duration",
    "cv_inhale_dur": "Inhale Duration Variability", "mean_exhale_dur_sec": "Mean Exhale Duration",
    "cv_exhale_dur": "Exhale Duration Variability", "ie_ratio": "Inhale/Exhale Ratio",
    "mean_peak_insp_flow": "Peak Inhale Flow", "mean_peak_exp_flow": "Peak Exhale Flow",
    "cv_peak_insp_flow": "Inhale Flow Variability", "mean_inhale_vol": "Mean Inhale Volume",
    "mean_exhale_vol": "Mean Exhale Volume", "mean_tidal_vol_proxy": "Mean Tidal Volume",
    "cv_inhale_vol": "Cv Inhale Vol", "ventilation_proxy": "Ventilation Proxy",
    "pct_breaths_with_inhale_pause": "Inhale Pauses (%)", "mean_inhale_pause_dur_sec": "Mean Inhale Pause",
    "cv_inhale_pause_dur": "Inhale Pause Variability", "inhale_pause_duty_cycle": "Inhale Pause Duty Cycle",
    "pct_breaths_with_exhale_pause": "Exhale Pauses (%)", "mean_exhale_pause_dur_sec": "Mean Exhale Pause",
    "cv_exhale_pause_dur": "Exhale Pause Variability", "exhale_pause_duty_cycle": "Exhale Pause Duty Cycle",
    "inhale_duty_cycle": "Inhale Duty Cycle", "exhale_duty_cycle": "Exhale Duty Cycle",
}


def pretty(name):
    return FEATURE_NAME_MAP.get(name, name.replace("_", " ").title())


TASKS = {
    "Valence": {
        "cv": "LOSO (leave-one-mouse-out)",
        "class_map": {"RI1": "Positive", "RI2": "Negative"},
        "class_order": ["Positive", "Negative"],   # panel row order (top, bottom)
        "palette": {"Positive": "#53A1A9", "Negative": "#A95376"},
        "perf_colors": ["#53A1A9", "#A95376", GRAY],  # best, 2nd, 3rd model bars
        "group": "mouse",
        "datasets": {
            "session": {"file": "raw-breathmetrics-features-session-3sec-subjectnames",
                        "target": "Condition", "type_filter": False},
            "interaction": {"file": "Raw-Breathmatrics-Features",
                            "target": "Condition", "type_filter": True},
        },
    },
    "Rank": {
        "cv": "LOCO (leave-one-cage-out)",
        "class_map": {"Dominant": "Dominant", "Subordinate": "Subordinate"},
        "class_order": ["Dominant", "Subordinate"],
        "palette": {"Dominant": "#1C304F", "Subordinate": "#D5B032"},
        "perf_colors": ["#D5B032", "#1C304F", GRAY],
        "group": "cage",
        "datasets": {
            "session": {"file": "raw-breathmetrics-features-session-3sec-rank",
                        "target": "RankLabel", "type_filter": False},
            "interaction": {"file": "Raw-Breathmatrics-Features-Ran-Interactions",
                            "target": "Rank", "type_filter": True},
        },
    },
}

MODEL_ORDER = ["LogReg", "SVM", "RandomForest"]


def make_models():
    num = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    imp = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    return {
        "LogReg": Pipeline([("prep", num), ("model", LogisticRegression(
            penalty="l2", C=0.1, solver="liblinear", class_weight="balanced",
            max_iter=2000, random_state=42))]),
        "SVM": Pipeline([("prep", num), ("model", SVC(
            kernel="rbf", C=1.0, gamma="scale", class_weight="balanced", random_state=42))]),
        "RandomForest": Pipeline([("prep", imp), ("model", RandomForestClassifier(
            n_estimators=300, max_depth=5, min_samples_leaf=5, class_weight="balanced",
            random_state=42, n_jobs=-1))]),
    }


def group_series(df, kind):
    if kind == "cage":
        if "PairID" in df.columns:
            return df["PairID"].astype(str)
        return df["Subject"].astype(str).str.split("_").str[0]
    return df["Subject"].astype(str)  # mouse


def _is_behavior_col(c):
    """BORIS-derived behavior features (e.g. n_bouts_*_sniffing, total_dur_*_sniffing) leak the
    label into a 'respiration' decoder — exclude them so features are respiration-only."""
    cl = c.lower()
    return "sniff" in cl or "bout" in cl


def feature_cols(df):
    cols = [c for c in df.columns
            if c not in META_BLOCK
            and not _is_behavior_col(c)
            and not (c.endswith("_z_subject") or c.endswith("_z_trial")
                     or c in {"vol_asymmetry", "peak_flow_ratio"})
            and pd.api.types.is_numeric_dtype(df[c])]
    miss = df[cols].isna().mean(axis=0)
    return miss[miss <= MAX_MISSING_FRAC].index.tolist()


def prepare(task, ds):
    cfg = TASKS[task]["datasets"][ds]
    df = pd.read_pickle(os.path.join(DATA_DIR, cfg["file"]))
    if cfg["type_filter"] and "Type" in df.columns and (df["Type"] == "Interaction").any():
        df = df[df["Type"] == "Interaction"].copy()
    tgt = cfg["target"]
    keep = list(TASKS[task]["class_map"].keys())
    df = df[df[tgt].isin(keep)].copy()
    df["_y"] = df[tgt].map(TASKS[task]["class_map"])
    df["_grp"] = group_series(df, TASKS[task]["group"]).values
    return df.reset_index(drop=True)


def eval_dataset(task, ds):
    df = prepare(task, ds)
    feats = feature_cols(df)
    X, y, grp = df[feats], df["_y"].astype(str), df["_grp"].astype(str)
    classes = np.sort(y.unique())
    folds = sorted(grp.unique(), key=lambda g: (len(g), g))
    rows, fold_rows = [], []
    for mname in MODEL_ORDER:
        model = make_models()[mname]
        bal, f1s, yt_all, yp_all = [], [], [], []
        for g in folds:
            te = (grp == g).to_numpy(); tr = ~te
            if te.sum() == 0 or len(np.unique(y[tr])) < 2:
                continue
            m = clone(model).fit(X[tr], y[tr])
            yp = m.predict(X[te]); yt = y[te].to_numpy()
            bal.append(balanced_accuracy_score(yt, yp))
            f1s.append(f1_score(yt, yp, labels=classes, average="macro", zero_division=0))
            yt_all.extend(yt); yp_all.extend(yp)
            fold_rows.append({"task": task, "window_type": ds, "model": mname, "held_out_group": g,
                              "n_test": int(te.sum()), "single_class_fold": len(np.unique(yt)) < 2,
                              "balanced_accuracy": round(bal[-1], 4), "macro_f1": round(f1s[-1], 4)})
        rows.append({
            "task": task, "window_type": ds, "model": mname,
            "n_windows": len(df), "n_features": len(feats), "n_groups": len(folds), "folds": len(bal),
            "balacc_mean": round(float(np.mean(bal)), 4), "balacc_std": round(float(np.std(bal)), 4),
            "balacc_sem": round(float(np.std(bal) / np.sqrt(len(bal))), 4) if bal else np.nan,
            "macro_f1_mean": round(float(np.mean(f1s)), 4), "macro_f1_std": round(float(np.std(f1s)), 4),
            "macro_f1_sem": round(float(np.std(f1s) / np.sqrt(len(f1s))), 4) if f1s else np.nan,
            "pooled_balacc": round(float(balanced_accuracy_score(yt_all, yp_all)), 4),
            "chance": CHANCE,
        })
    return df, feats, rows, fold_rows


def top_features(df, feats, task):
    """Fit standardized L2 LogReg on the full dataset; split signed coefs into the two classes."""
    X, y = df[feats], df["_y"].astype(str)
    pipe = Pipeline([("prep", Pipeline([("imputer", SimpleImputer(strategy="median")),
                                        ("scaler", StandardScaler())])),
                     ("model", LogisticRegression(max_iter=5000, class_weight="balanced",
                                                  solver="lbfgs", C=1.0, random_state=42))])
    pipe.fit(X, y)
    lr = pipe.named_steps["model"]
    coef = pd.Series(lr.coef_[0], index=feats)
    pos_class = lr.classes_[1]  # +coef points to classes_[1]
    out = {}
    for cls in TASKS[task]["class_order"]:
        s = coef if cls == pos_class else -coef       # orient so larger = more "this class"
        top = s.sort_values(ascending=False).head(5)
        out[cls] = [(pretty(f), abs(v)) for f, v in top.items()]
    return out


# ---------------------------------------------------------------------------
def _perf_axis(ax, sub, metric_mean, metric_std, colors, ylabel):
    sub = sub.sort_values("balacc_mean", ascending=False).reset_index(drop=True)
    x = np.arange(len(sub))
    bars = ax.bar(x, sub[metric_mean], yerr=sub[metric_std], capsize=5,
                  color=colors[:len(sub)], edgecolor="white", linewidth=0.8,
                  error_kw=dict(elinewidth=1.2, ecolor=GRAY))
    ax.axhline(CHANCE, ls="--", lw=1.0, color="black")
    # rotate model names so the 1.65x-larger labels don't overlap in the narrow perf subplot
    ax.set_xticks(x)
    ax.set_xticklabels(sub["model"], fontsize=PERF_MODEL_TICK_FONTSIZE, rotation=20, ha="right")
    ax.set_ylabel(ylabel, fontsize=PERF_YLABEL_FONTSIZE); ax.set_ylim(0, 1.05)
    ax.tick_params(axis="y", labelsize=DEFAULT_TICK_LABELSIZE)
    ax.spines[["top", "right"]].set_visible(False)
    off = float(sub[metric_std].max() or 0) + 0.03
    for b, v in zip(bars, sub[metric_mean]):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + off, f"{v:.4f}",
                ha="center", va="bottom", fontsize=8)
    return sub


def _feat_diverging_axis(ax, tops, cfg):
    """One diverging horizontal bar chart per task on a single shared y-axis:
    class_order[0] (Positive / Dominant) features extend RIGHT (positive coefficient), stacked on
    top; class_order[1] (Negative / Subordinate) extend LEFT (negative), stacked below. Bars
    diverge from a central x=0 line; all feature names sit on the left y-axis."""
    c0, c1 = cfg["class_order"]
    pos, neg = tops[c0], tops[c1]          # each [(name, magnitude), ...] strongest first
    n = len(pos)
    ys, names, vals, colors = [], [], [], []
    for i, (name, mag) in enumerate(pos):  # positive class on top: y = 2n-1 .. n
        ys.append(2 * n - 1 - i); names.append(name); vals.append(mag); colors.append(cfg["palette"][c0])
    for i, (name, mag) in enumerate(neg):  # negative class below: y = n-1 .. 0, values negated
        ys.append(n - 1 - i); names.append(name); vals.append(-mag); colors.append(cfg["palette"][c1])
    ax.barh(ys, vals, color=colors, edgecolor="white", linewidth=0.6)
    ax.axvline(0, color="0.4", lw=1.0)
    ax.set_yticks(ys); ax.set_yticklabels(names, fontsize=FEAT_TICK_FONTSIZE)
    ax.set_xlabel("Coefficient", fontsize=FEAT_XLABEL_FONTSIZE)
    ax.tick_params(axis="x", labelsize=DEFAULT_TICK_LABELSIZE)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(handles=[Patch(facecolor=cfg["palette"][c0], label=c0),
                       Patch(facecolor=cfg["palette"][c1], label=c1)],
              frameon=False, fontsize=12, loc="lower right")


def _save_fig(fig, stem):
    for ext in ("svg", "png"):
        fig.savefig(os.path.join(PLOT_DIR, f"{stem}.{ext}"), transparent=True,
                    bbox_inches="tight", dpi=300)
    plt.close(fig)


def _task_header(fig, col, task, wt, cfg):
    """Bold per-task column header (Valence left, Rank right). Kept short so the two headers
    don't collide in the mid-width performance figure."""
    fig.text(0.27 if col == 0 else 0.77, 0.99,
             f"{task}  ({wt}, {cfg['cv'].split()[0]})",
             ha="center", va="top", fontsize=14, fontweight="bold")


def render_performance(best):
    """Figure 1: Model-Performance bar charts, Valence (left) vs Rank (right), side by side.
    Rows = Balanced Accuracy (top) / Macro F1 (bottom). -> decoder_performance.{svg,png}"""
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.patch.set_alpha(0)
    for col, task in enumerate(("Valence", "Rank")):
        wt, df, feats, sub = best[task]
        cfg = TASKS[task]
        _perf_axis(axes[0, col], sub, "balacc_mean", "balacc_sem", cfg["perf_colors"], "Balanced Accuracy")
        _perf_axis(axes[1, col], sub, "macro_f1_mean", "macro_f1_sem", cfg["perf_colors"], "Macro F1")
        axes[0, col].set_title("Model Performance", fontsize=13, pad=22)
        _task_header(fig, col, task, wt, cfg)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save_fig(fig, "decoder_performance")


def render_features(best):
    """Figure 2: Top-5-Feature diverging bar charts, one per task on a single shared y-axis:
    Valence (left) and Rank (right), side by side in one figure. Positive/Dominant extend right,
    Negative/Subordinate extend left. -> decoder_features.{svg,png}
    Wide aspect ratio (stretched on x) for a horizontally long poster panel."""
    fig, axes = plt.subplots(1, 2, figsize=(20, 6.5))
    fig.patch.set_alpha(0)
    for col, task in enumerate(("Valence", "Rank")):
        wt, df, feats, sub = best[task]
        cfg = TASKS[task]
        tops = top_features(df, feats, task)
        _feat_diverging_axis(axes[col], tops, cfg)
        axes[col].set_title("Top 5 Features", fontsize=13, pad=18)
        _task_header(fig, col, task, wt, cfg)
    fig.subplots_adjust(wspace=0.55)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    _save_fig(fig, "decoder_features")


NOTES = """# Decoder metrics — methodology notes

**What is being decoded.** Two binary classifications from BreathMetrics respiration features:
- **Valence**: Positive (RI1) vs Negative (RI2) social interaction.
- **Rank**: Dominant vs Subordinate cagemate (Aim1 baseline cagemate cohort, 7 mice / 4 cages).

Each is run on two window types: **session** (3 s windows sampled across the recording) and
**interaction** (sniff-bout windows, both actors).

**Train / validation / evaluation.** There is no separate hold-out test split. Performance is
estimated by **leave-one-group-out cross-validation**, where the group is the biological unit:
- Valence → **leave-one-mouse-out (LOSO)**: valence is a within-mouse contrast (same mouse does
  RI1 and RI2), so an entire mouse is held out and predicted from models trained on the other mice.
- Rank → **leave-one-cage-out (LOCO)**: both cagemates of a pair are held out together, so no
  mouse from the test cage (in either social role) is seen during training.

In every fold the held-out group is the **validation set** and its windows are predicted
out-of-fold; concatenating all folds gives one out-of-fold prediction per window. Model
hyper-parameters are **fixed** (not tuned), so there is no inner tuning loop and no optimism from
peeking at the test groups. `class_weight="balanced"` corrects class imbalance inside training.

**Models & kernels.** Three fixed (untuned) estimators, all `class_weight="balanced"`:
- **LogisticRegression** — LINEAR (L2, liblinear).
- **SVM** — **NON-LINEAR: RBF (Gaussian) kernel** (`kernel="rbf"`, C=1, gamma="scale").
- **RandomForest** — NON-LINEAR (300 trees, depth 5). So two of the three models are non-linear;
  a non-linear kernel (RBF-SVM) WAS used and is reported alongside the linear LogReg and RF.

**Metrics.** Balanced accuracy (mean of per-class recall) and macro F1, computed per fold then
averaged (mean ± std across folds); chance = 0.5. The bar panels show per-fold **mean ± SEM**
(std/√folds); the CSV keeps both std and SEM. Features are respiration-only — BORIS behavior
columns (n_bouts_*/total_dur_*_sniffing) are excluded so behavior can't leak the label.

**Caveats.**
- Small cohorts. Rank LOCO has only 4 folds; **cage 3 = singleton {3_5}** (Subordinate only) makes
  one fold single-class and degenerate, inflating the std.
- Valence **interaction** windows are sparse for the negative class, so that variant is noisy;
  valence **session** is the reliable valence estimate.
- Feature panels: signed standardized **LINEAR L2-LogReg** coefficients fit on the full best variant
  (bars show |coefficient|); positive-coef features define one class, negative the other. These are
  from the LINEAR model even when the best-performing bar is the non-linear SVM/RF, and are
  **multivariate partial effects**, NOT group means.
- **Do NOT read direction off the RANK feature panel.** (a) The respiration features are highly
  collinear (breathing rate, breath count, IBI all measure frequency), so LogReg splits the weight
  and coefficient SIGNS flip — e.g. "Breathing Rate" can land on the Dominant side while "Breath
  Count" (the same thing) lands on Subordinate. (b) The rank model is at CHANCE, so its coefficients
  fit noise. Trust the univariate whole-session resp-rate plot (Dominants breathe slower) for
  direction, not these coefficients. Valence (~0.81) coefficients are meaningful; rank's are not.
"""


def main():
    all_sum, all_fold = [], []
    best = {}
    print("=" * 84)
    print("Decoder metrics — uniform leave-one-group-out CV (valence=mouse, rank=cage)")
    print("=" * 84)
    for task in TASKS:
        variants = {}
        for ds in TASKS[task]["datasets"]:
            df, feats, rows, frows = eval_dataset(task, ds)
            all_sum += rows; all_fold += frows
            variants[ds] = (df, feats, pd.DataFrame(rows))
            counts = df["_y"].value_counts().to_dict()
            best_ba = max(r["balacc_mean"] for r in rows)
            print(f"[{task:8s} / {ds:11s}] windows={len(df):4d} groups={df['_grp'].nunique()} "
                  f"classes={counts} bestBalAcc={best_ba:.3f}")
        # best variant = highest single-model mean balanced accuracy
        pick = max(variants, key=lambda d: variants[d][2]["balacc_mean"].max())
        df, feats, sub = variants[pick]
        best[task] = (pick, df, feats, sub)
        print(f"   -> best {task} variant: {pick}")

    summary = pd.DataFrame(all_sum)
    folds = pd.DataFrame(all_fold)
    summary.to_csv(os.path.join(DATA_DIR, "decoder_metrics_summary.csv"), index=False)
    folds.to_csv(os.path.join(DATA_DIR, "decoder_metrics_per_fold.csv"), index=False)
    with open(os.path.join(PLOT_DIR, "NOTES.md"), "w", encoding="utf-8") as fh:
        fh.write(NOTES)

    pd.set_option("display.width", 220, "display.max_columns", 40)
    print("\n" + "=" * 84)
    print("SUMMARY TABLE (all tasks x window types x models)")
    print("=" * 84)
    print(summary.to_string(index=False))

    render_performance(best)
    render_features(best)
    print(f"\nSaved: {os.path.join(DATA_DIR, 'decoder_metrics_summary.csv')}")
    print(f"Saved: {os.path.join(DATA_DIR, 'decoder_metrics_per_fold.csv')}")
    print(f"Saved: {os.path.join(PLOT_DIR, 'decoder_performance.svg')} (+ .png)")
    print(f"Saved: {os.path.join(PLOT_DIR, 'decoder_features.svg')} (+ .png)")
    print(f"Saved: {os.path.join(PLOT_DIR, 'NOTES.md')}")
    print("\nBest variant shown per task: "
          + ", ".join(f"{t}={best[t][0]}" for t in best))


if __name__ == "__main__":
    main()
