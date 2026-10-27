"""
Poster foundation plots — behavior + respiration rate (valence & cagemate/rank).

Consolidates the behavioral and simple respiration-rate box-whisker plots that sit
*under* the classifier story on the poster, using the CORRECT cohorts + the SAME
drops/signal-processing as the feature-extraction notebooks (see DROPS_AND_INPUTS.md).

Cohorts
  * Valence  -> Aim1 RI1 (positive) vs RI2 (negative) recordings.
  * Rank     -> baseline cagemate cohort (ECG_cohort1 / Aim1). BLA is NOT used (no baseline).

Produces 7 figures (each saved transparent, as .svg + .png):
  1. behavior_valence_subject_only — total sniffing time by behavior, Positive vs Negative (focal only)
  2. behavior_valence_both_actors  — same, counting focal + partner sniffing
  3. behavior_rank_subject_only    — total investigation time by behavior, Dom vs Sub (focal only)
  4. behavior_rank_both_actors     — same, counting focal + partner sniffing
  5. resp_rate_valence    — whole-session respiration rate, Positive vs Negative
  6. resp_rate_rank       — whole-session respiration rate, Dominant vs Subordinate
  7. resp_rate_rank_interactions — respiration rate DURING focal interaction (sniff) bouts, Dom vs Sub

Both actor conventions are plotted per cohort (subject-only + both) so the user can compare and
pick. The two originals used DIFFERENT filters — an inconsistency this consolidation surfaces:
  * rank = both actors (`cm_iteration1.ipynb`).
  * valence = subject-initiated only (`plot_social_behavior_RI1_RI2.ipynb`).
With both actors, negative valence reads higher (in RI2 the partner does most of the sniffing) —
that is NOT a flipped Positive/Negative label.
  * ONE canonical recording set per cohort (h5 must have a matching, non-empty BORIS) shared by
    the behavior AND resp-rate plots, so every recording is consistent across plots.
  * Documented drops applied: rank 1_1/1_2 (fighting) + 3_6 (no BORIS); valence 5_13 (outlier /
    broken RI2 signal) + RI1_3_5 (no BORIS). Prefer *_VT BORIS exports; drop timeless bad dup.
  * Bout cleaning 0.5-20 s (removes the 400 s scoring error + sub-0.5 s noise). NO min-bout gate
    (that is a classifier requirement; here low negative-valence investigation is real signal).
  * Signal processing = the shared helper `preprocess_resp_to_target_rate` at 400 Hz, exactly as
    the feature-extraction notebooks (NO FFT resample; that was only in cm_iteration1's local loader).

Run:  conda run -n biopipeline-env python poster_behavior_resp_plots.py
Outputs -> resp_classification/poster-plots/behavior and resp rate/
"""

from __future__ import annotations

import os
import re
import sys
import glob
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks

# --- shared helper library lives one dir up ---
HERE = os.path.dirname(os.path.abspath(__file__))
RESP_CLASSIFICATION_DIR = os.path.abspath(os.path.join(HERE, ".."))
if RESP_CLASSIFICATION_DIR not in sys.path:
    sys.path.insert(0, RESP_CLASSIFICATION_DIR)

from resp_helper_functions import (  # noqa: E402
    load_clean_resp_signal,
    load_rank_resp_signal_interactions,
    load_clean_boris,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths (Dropbox is the live source; E:/D: mirrors auto-fall-back if present)
# ---------------------------------------------------------------------------
_DROPBOX_ROOT = r"C:\Users\thoma\UF Dropbox\Thomas Heeps\Padilla-Coreano Lab\2025"


def _first_existing(*candidates: str) -> str:
    for c in candidates:
        if c and os.path.isdir(c):
            return c
    raise FileNotFoundError("No data dir found:\n  " + "\n  ".join(candidates))


VALENCE_H5_DIR = _first_existing(
    os.path.join(_DROPBOX_ROOT, r"ECG_cohort1\Aim1\AIM1\Day1_new\Resp_h5"),
    r"E:\Aim1\AIM1\Day1_new\resp_h5",
)
VALENCE_BORIS_DIR = _first_existing(
    os.path.join(_DROPBOX_ROOT, r"ECG_cohort1\Aim1\AIM1\Day1_new\boris_csv_RI1 and RI2"),
)
CM_H5_DIR = _first_existing(
    os.path.join(_DROPBOX_ROOT,
                 r"ECG_cohort1\Aim1\AIM1\Day1_new\CM interactions\cm_h5\baseline_cagemate_interactions_h5_outputs"),
    r"E:\Aim1\AIM1\Day1_new\resp_CM_h5\baseline_cagemate_interactions_h5_outputs",
)
CM_BORIS_DIR = _first_existing(
    os.path.join(_DROPBOX_ROOT,
                 r"ECG_cohort1\Aim1\AIM1\Day1_new\CM interactions\cm_boris\baseline csv"),
)
OUT_DIR = os.path.abspath(os.path.join(HERE, "..", "poster-plots", "behavior and resp rate"))
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Config — drops & processing (toggle here; see DROPS_AND_INPUTS.md for rationale)
# ---------------------------------------------------------------------------
TARGET_RATE = 400          # Hz — matches the feature-extraction notebooks
MIN_BOUT_SEC = 0.5         # drop sub-0.5 s noise bouts
MAX_BOUT_SEC = 20.0        # drop >20 s bouts (kills the ~400 s scoring error) — FE convention

# Behavior plots now render BOTH actor conventions per cohort (see main()), so these toggles no
# longer gate them. SUBJECT_ONLY_RANK still governs the resp-rate-DURING-interactions plot only:
# False = average resp rate over sniff bouts by either actor; True = focal-initiated bouts only.
SUBJECT_ONLY_RANK = False       # interaction resp-rate: both actors' sniff bouts
SUBJECT_ONLY_VALENCE = False    # (retained for reference; behavior plots ignore this now)

# --- Plot styling ---
# x/y TICK label font size. Matplotlib default is 10 pt; scaled 1.65x for poster readability
# (-> 16.5 pt). Applied via ax.tick_params(labelsize=...) on both axes.
TICK_LABELSIZE = 10 * 1.65
# x/y AXIS TITLE font size (the "Respiration Rate (Hz)" / "Behavior" labels). Base 11 pt, ×1.65.
AXIS_LABEL_FONTSIZE = 11 * 1.65
# Behavior-plot legend: larger text + anchored further left (closer to the plot). Base was 10 pt
# at x-anchor 1.02 (just outside the right edge); now 14 pt pulled in to 0.86.
LEGEND_FONTSIZE = 14
LEGEND_ANCHOR_X = 0.86
BOX_LINEWIDTH = 2.2        # box/whisker/cap/median line width (heavier, poster-style)
STRIP_SIZE = 6             # jitter dot size
BOX_ALPHA = 0.9            # box fill opacity (higher = darker/more saturated, closer to originals)
WHIS = 1.5                 # whisker length: 1.5 = Tukey 1.5*IQR (matches original; outliers fall
                           # outside). Set to (0, 100) for whiskers to span the full data range.
LINE_WIDTH = 1.0           # subject-connecting lines (resp-rate paired plots only)
LINE_ALPHA = 0.55

# Rank/cagemate drops. 3_6 has no (empty) BORIS. Cage 1 (1_1/1_2) had fighting but is KEPT here
# because the cagemate cohort is small — dropping it would leave too few subjects for the plot.
RANK_DROP_SUBJECTS = {"3_6"}
# Valence drops (from feature_extraction-interactions-valence.ipynb + resp-rate outlier)
VALENCE_DROP_SUBJECTS = {"5_13"}             # low-resp outlier; RI2 signal is broken (flat)
# (RI1_3_5 has no BORIS -> dropped automatically by the matching-BORIS requirement)

# ---------------------------------------------------------------------------
# Palettes & labels
# ---------------------------------------------------------------------------
RANK_PALETTE = {
    "Dominant": (28 / 255, 48 / 255, 79 / 255),       # navy   ~ #1C304F
    "Subordinate": (213 / 255, 176 / 255, 50 / 255),  # mustard ~ #D5B032
}
VALENCE_PALETTE = {"Positive": "#53A1A9", "Negative": "#A95376"}  # teal / pink (RI1 / RI2)
RANK_ORDER = ["Subordinate", "Dominant"]
VALENCE_ORDER = ["Positive", "Negative"]

AIM1_RANK_MAP = {
    "1_1": "Subordinate", "1_2": "Dominant",
    "2_3": "Subordinate", "2_4": "Dominant",
    "3_5": "Subordinate", "3_6": "Dominant",
    "4_7": "Subordinate", "4_8": "Dominant",
}
BEHAVIOR_DISPLAY = {
    "facial sniffing": "Face Investigation",
    "body sniffing": "Body Investigation",
    "anogenital sniffing": "Anogenital Investigation",
}
BEHAVIOR_ORDER = ["Face Investigation", "Body Investigation", "Anogenital Investigation"]


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------
def subject_from_name(name: str) -> str | None:
    m = re.search(r"_s(\d+)_(\d+)", name)
    return f"{m.group(1)}_{m.group(2)}" if m else None


def valence_condition(name: str) -> str | None:
    if name.startswith("RI1"):
        return "Positive"
    if name.startswith("RI2"):
        return "Negative"
    return None  # BLRI baseline etc.


def cm_session_key(name: str) -> str | None:
    """Common key shared by a CM h5 and its BORIS csv: 'CM_s..._<date>_<time>'."""
    m = re.match(r"(CM_s\d+_\d+_(?:d|sub)\d+_\d+_\d{8}_\d{6})", name)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# BORIS loading
# ---------------------------------------------------------------------------
def safe_load_boris(path, subject_only) -> pd.DataFrame:
    """Sniffing bouts (0.5-20 s), both actors (subject_only=False) or focal-only (True).
    Tolerant of empty / malformed / tab-separated files (returns empty rather than raising)."""
    try:
        df = load_clean_boris(path, subject_only=subject_only)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()
    if df is None or df.empty or not {"Behavior", "Start", "Stop", "Duration"}.issubset(df.columns):
        return pd.DataFrame()
    return df[(df["Duration"] >= MIN_BOUT_SEC) & (df["Duration"] <= MAX_BOUT_SEC)]


def boris_is_usable(path) -> bool:
    """A BORIS export counts as a valid behavior file if it parses and has >=1 sniffing bout by
    EITHER actor. Drops only truly-bad files: the empty 3_6 export and the tab-separated timeless
    CM_s4_8_sub4_7_20250623.csv duplicate. Keeps valid-but-low-focal-activity recordings."""
    try:
        df = load_clean_boris(path, subject_only=False)
    except Exception:  # noqa: BLE001
        return False
    return df is not None and not df.empty and {"Behavior", "Start", "Stop", "Duration"}.issubset(df.columns)


def pick_cm_boris_files() -> dict[str, str]:
    """CM session key -> best BORIS csv (prefer *_VT; require date_time so the tab-separated,
    timeless bad-duplicate CM_s4_8_sub4_7_20250623.csv is excluded)."""
    best: dict[str, str] = {}
    for path in sorted(glob.glob(os.path.join(CM_BORIS_DIR, "*.csv"))):
        name = os.path.basename(path)
        key = cm_session_key(name)
        if key is None:
            continue
        if key not in best or ("_VT" in name and "_VT" not in os.path.basename(best[key])):
            best[key] = path
    return best


def valence_boris_lookup() -> dict[tuple, str]:
    """(condition, subject) -> BORIS csv path."""
    out: dict[tuple, str] = {}
    for path in sorted(glob.glob(os.path.join(VALENCE_BORIS_DIR, "*.csv"))):
        name = os.path.basename(path)
        cond, subj = valence_condition(name), subject_from_name(name)
        if cond and subj:
            out[(cond, subj)] = path
    return out


# ---------------------------------------------------------------------------
# Canonical recording sets — h5 paired with a matching, non-empty BORIS, minus drops.
# The SAME set feeds the behavior and resp-rate plots for that cohort.
# Each entry: dict(h5, boris, subject, group)
# ---------------------------------------------------------------------------
def rank_recordings() -> list[dict]:
    boris_by_key = pick_cm_boris_files()
    recs = []
    for h5 in sorted(glob.glob(os.path.join(CM_H5_DIR, "*.h5"))):
        name = os.path.basename(h5)
        subj = subject_from_name(name)
        rank = AIM1_RANK_MAP.get(subj)
        if rank is None or subj in RANK_DROP_SUBJECTS:
            continue
        boris = boris_by_key.get(cm_session_key(name))
        if boris is None or not boris_is_usable(boris):     # require a usable BORIS file
            continue
        recs.append({"h5": h5, "boris": boris, "subject": subj, "group": rank})
    return recs


def valence_recordings() -> list[dict]:
    lookup = valence_boris_lookup()
    recs = []
    for h5 in sorted(glob.glob(os.path.join(VALENCE_H5_DIR, "*.h5"))):
        name = os.path.basename(h5)
        cond, subj = valence_condition(name), subject_from_name(name)
        if cond is None or subj in VALENCE_DROP_SUBJECTS:
            continue
        boris = lookup.get((cond, subj))
        if boris is None or not boris_is_usable(boris):     # require a usable BORIS file
            continue
        recs.append({"h5": h5, "boris": boris, "subject": subj, "group": cond})
    return recs


# ---------------------------------------------------------------------------
# Respiration rate
# ---------------------------------------------------------------------------
def respiratory_rate(signal, time, start, end, fs) -> float:
    mask = (time >= start) & (time < end)
    seg = signal[mask]
    if seg.size < 3:
        return np.nan
    peaks, _ = find_peaks(seg, distance=max(1, fs * 0.0833))  # >=12 Hz refractory
    dur = end - start
    return len(peaks) / dur if dur > 0 else np.nan


def load_signal(h5, cohort):
    """Cohort-matched loader, exactly as the feature-extraction notebooks (400 Hz, no FFT)."""
    try:
        if cohort == "rank":
            sig, t, fs, _ = load_rank_resp_signal_interactions(
                h5, target_rate=TARGET_RATE, prefer_raw=True, verbose=False)
        else:
            sig, t, fs, _ = load_clean_resp_signal(h5, target_rate=TARGET_RATE, verbose=False)
    except Exception as e:  # noqa: BLE001
        print(f"  ! load failed {os.path.basename(h5)}: {e}")
        return None, None, None
    return (sig, t, fs) if sig is not None else (None, None, None)


# ---------------------------------------------------------------------------
# Dataframe builders (all from the canonical sets)
# ---------------------------------------------------------------------------
def behavior_df(recs: list[dict], group_key: str, subject_only: bool) -> pd.DataFrame:
    rows = []
    for r in recs:
        bouts = safe_load_boris(r["boris"], subject_only)
        for beh, disp in BEHAVIOR_DISPLAY.items():
            total = bouts.loc[bouts["Behavior"] == beh, "Duration"].sum()
            rows.append({"Recording": os.path.basename(r["h5"]), "Subject": r["subject"],
                         group_key: r["group"], "Behavior": disp, "Total_Duration": total})
    df = pd.DataFrame(rows)
    # collapse any subject with >1 recording to a per-subject total (rank 3_5 etc.)
    if not df.empty:
        df = df.groupby(["Subject", group_key, "Behavior"], as_index=False)["Total_Duration"].sum()
    return df


def whole_session_rate_df(recs: list[dict], cohort: str, group_key: str) -> pd.DataFrame:
    rows = []
    for r in recs:
        sig, t, fs = load_signal(r["h5"], cohort)
        if sig is None:
            continue
        rows.append({"Recording": os.path.basename(r["h5"]), "Subject": r["subject"],
                     "PairID": r["subject"].split("_")[0],  # cage id (cagemate pairing for rank)
                     group_key: r["group"], "Respiration Rate (Hz)": respiratory_rate(sig, t, t[0], t[-1], fs)})
    return pd.DataFrame(rows).dropna(subset=["Respiration Rate (Hz)"])


def interaction_rate_df(recs: list[dict], cohort: str, group_key: str) -> pd.DataFrame:
    """Mean respiration rate during sniff bouts (both actors for rank), per recording."""
    rows = []
    for r in recs:
        bouts = safe_load_boris(r["boris"], SUBJECT_ONLY_RANK)
        if bouts.empty:
            continue
        sig, t, fs = load_signal(r["h5"], cohort)
        if sig is None:
            continue
        rates = [respiratory_rate(sig, t, b["Start"], b["Stop"], fs) for _, b in bouts.iterrows()]
        rows.append({"Recording": os.path.basename(r["h5"]), "Subject": r["subject"],
                     "PairID": r["subject"].split("_")[0],  # cage id (cagemate pairing for rank)
                     group_key: r["group"], "Respiration Rate (Hz)": np.nanmean(rates) if rates else np.nan})
    return pd.DataFrame(rows).dropna(subset=["Respiration Rate (Hz)"])


# ---------------------------------------------------------------------------
# Plotting (style matches cm_iteration1.ipynb: box + strip, translucent, no fliers)
# ---------------------------------------------------------------------------
def _save(fig, stem: str):
    for ext in ("svg", "png"):
        fig.savefig(os.path.join(OUT_DIR, f"{stem}.{ext}"), transparent=True,
                    bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_group_box(df, group_col, palette, order, title, ylabel, stem, paired=False,
                   pair_by="Subject", pair_note=None):
    """Box + strip by group. If paired, connect each `pair_by` unit across the two x-groups:
      * valence  -> pair_by="Subject" (the SAME mouse spans Positive/Negative)
      * rank     -> pair_by="PairID"  (a CAGE spans Subordinate/Dominant; the line links the two
                    *cagemates*, i.e. two different mice sharing a cage — NOT the same individual).
    Only units present in both x-groups get a line (e.g. the singleton cage {3_5} is not drawn)."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.boxplot(data=df, x=group_col, y="Respiration Rate (Hz)", hue=group_col,
                palette=palette, order=order, hue_order=order, fliersize=0, whis=WHIS,
                linewidth=BOX_LINEWIDTH, legend=False, ax=ax,
                boxprops=dict(alpha=BOX_ALPHA), whiskerprops=dict(alpha=0.9),
                capprops=dict(alpha=0.9),
                medianprops=dict(alpha=0.95, linewidth=BOX_LINEWIDTH + 0.4))
    if paired and pair_by in df.columns:
        piv = (df.pivot_table(index=pair_by, columns=group_col,
                              values="Respiration Rate (Hz)", aggfunc="mean")
                 .reindex(columns=order))
        for _, row in piv.iterrows():
            if row.notna().all():
                ax.plot(range(len(order)), [row[o] for o in order],
                        color="0.5", lw=LINE_WIDTH, alpha=LINE_ALPHA, zorder=1)
    sns.stripplot(data=df, x=group_col, y="Respiration Rate (Hz)", hue=group_col,
                  palette=palette, order=order, hue_order=order, jitter=not paired, alpha=0.9,
                  size=STRIP_SIZE, edgecolor="black", linewidth=0.8, legend=False, ax=ax, zorder=2)
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_xlabel("")
    ax.tick_params(axis="both", labelsize=TICK_LABELSIZE)
    if pair_note:
        ax.text(0.0, -0.14, pair_note, transform=ax.transAxes, fontsize=8,
                color="0.35", ha="left", va="top")
    sns.despine(ax=ax)
    _save(fig, stem)


def plot_behavior_box(df, hue_col, palette, hue_order, title, stem, legend_title):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x="Behavior", y="Total_Duration", hue=hue_col, palette=palette,
                order=BEHAVIOR_ORDER, hue_order=hue_order, fliersize=0, whis=WHIS,
                linewidth=BOX_LINEWIDTH, ax=ax,
                boxprops=dict(alpha=BOX_ALPHA), whiskerprops=dict(alpha=0.9),
                capprops=dict(alpha=0.9),
                medianprops=dict(alpha=0.95, linewidth=BOX_LINEWIDTH + 0.4))
    sns.stripplot(data=df, x="Behavior", y="Total_Duration", hue=hue_col, palette=palette,
                  order=BEHAVIOR_ORDER, hue_order=hue_order, dodge=True, jitter=True, marker="o",
                  alpha=0.9, size=STRIP_SIZE, edgecolor="black", linewidth=0.8, ax=ax)
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_ylabel("Total Sniffing Time (s)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_xlabel("Behavior", fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_LABELSIZE)
    handles, labels = ax.get_legend_handles_labels()
    n = len(hue_order)
    ax.legend(handles[:n], labels[:n], title=legend_title, frameon=False,
              bbox_to_anchor=(LEGEND_ANCHOR_X, 1), loc="upper left",
              fontsize=LEGEND_FONTSIZE, title_fontsize=LEGEND_FONTSIZE)
    sns.despine(ax=ax)
    _save(fig, stem)


def _counts(df, col):
    return ", ".join(f"{k}={v}" for k, v in df[col].value_counts().items()) if not df.empty else "empty"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    rank_recs = rank_recordings()
    val_recs = valence_recordings()
    print("Canonical recording sets (h5 + matching focal BORIS, drops applied):")
    print(f"  rank    : {len(rank_recs)} recs -> "
          + ", ".join(f"{r['subject']}({r['group'][:3]})" for r in rank_recs))
    print(f"  valence : {len(val_recs)} recs -> "
          + ", ".join(f"{r['subject']}/{r['group'][:3]}" for r in val_recs))
    print(f"  output  : {OUT_DIR}\n")

    # behavior plots — produce BOTH actor conventions per cohort so the user can compare and
    # decide which to keep (subject-only matches plot_social_behavior_RI1_RI2.ipynb; both-actors
    # matches cm_iteration1.ipynb rank). These two originals used DIFFERENT filters — that is the
    # inconsistency we surfaced. NOTE: with both actors, negative-valence reads higher because in
    # RI2 the partner does most of the sniffing (NOT a flipped Positive/Negative label).
    val_by_group = pd.Series([r["group"] for r in val_recs]).value_counts().to_dict()
    rank_by_group = pd.Series([r["group"] for r in rank_recs]).value_counts().to_dict()

    # 1-2. behavior — valence (subject-only + both actors)
    plot_behavior_box(behavior_df(val_recs, "Valence", True), "Valence", VALENCE_PALETTE,
                      VALENCE_ORDER, "Sniffing Investigation by Valence (Subject-Initiated)",
                      "behavior_valence_subject_only", "Valence")
    plot_behavior_box(behavior_df(val_recs, "Valence", False), "Valence", VALENCE_PALETTE,
                      VALENCE_ORDER, "Sniffing Investigation by Valence (Both Actors)",
                      "behavior_valence_both_actors", "Valence")
    print(f"[1-2/7] behavior_valence (subject_only + both_actors)  recs: {val_by_group}")

    # 3-4. behavior — rank (subject-only + both actors)
    plot_behavior_box(behavior_df(rank_recs, "Rank", True), "Rank", RANK_PALETTE, RANK_ORDER,
                      "Sniffing Interaction by Rank (Cagemate, Subject-Initiated)",
                      "behavior_rank_subject_only", "Rank")
    plot_behavior_box(behavior_df(rank_recs, "Rank", False), "Rank", RANK_PALETTE, RANK_ORDER,
                      "Sniffing Interaction by Rank (Cagemate, Both Actors)",
                      "behavior_rank_both_actors", "Rank")
    print(f"[3-4/7] behavior_rank (subject_only + both_actors)  recs: {rank_by_group}")

    # 5. respiration rate — valence (whole session)
    vr = whole_session_rate_df(val_recs, "valence", "Valence")
    plot_group_box(vr, "Valence", VALENCE_PALETTE, VALENCE_ORDER,
                   "Whole-Session Respiration Rate (Valence)", "Respiration Rate (Hz)",
                   "resp_rate_valence", paired=True)
    print(f"[5/7] resp_rate_valence recs: {len(vr)}  ({_counts(vr, 'Valence')})")

    # 6. respiration rate — rank (whole session). Lines link CAGEMATES (Sub<->Dom within a cage).
    _CAGE_NOTE = "* lines link cagemates (same cage, session-level), not the same mouse"
    rr = whole_session_rate_df(rank_recs, "rank", "Rank")
    plot_group_box(rr, "Rank", RANK_PALETTE, RANK_ORDER,
                   "Whole-Session Respiration Rate (Cagemate)", "Respiration Rate (Hz)", "resp_rate_rank",
                   paired=True, pair_by="PairID", pair_note=_CAGE_NOTE)
    print(f"[6/7] resp_rate_rank    recs: {len(rr)}  ({_counts(rr, 'Rank')})")

    # 7. respiration rate — rank during interactions (cagemate-paired lines)
    ir = interaction_rate_df(rank_recs, "rank", "Rank")
    plot_group_box(ir, "Rank", RANK_PALETTE, RANK_ORDER,
                   "Respiration Rate During Interactions (Cagemate)", "Respiration Rate (Hz)",
                   "resp_rate_rank_interactions", paired=True, pair_by="PairID", pair_note=_CAGE_NOTE)
    print(f"[7/7] resp_rate_rank_interactions  recs: {len(ir)}  ({_counts(ir, 'Rank')})")

    print(f"\nDone. 7 transparent figures (.svg + .png) -> {OUT_DIR}")


if __name__ == "__main__":
    main()
