# Respiration → Rank & Valence — Poster Consolidation Map

> **Purpose of this doc:** a single orientation map for the poster ("decoding rank and
> valence from respiration"). It says *which cohort to use, which notebooks made what, where
> the data files and palettes live, and what actually worked*. It is a **map, not gospel** —
> notebooks drift; verify a path/cell before trusting it.
>
> **Status (2026-07-13):** the consolidation is **DONE** — the plotting code now lives in
> [`poster-code/poster_behavior_resp_plots.py`](../poster-code/poster_behavior_resp_plots.py)
> (foundation behavior + resp-rate plots) and [`poster-code/decoder_metrics_table.py`](../poster-code/decoder_metrics_table.py)
> (uniform LOCO/LOSO decoder metrics + poster panels). This doc is now the **orientation map** for
> that code. This folder (`docs/`) is git-ignored.
>
> **Root of everything:** `notebooks/thomas_notebooks/resp_classification/` (called `resp_classification/` below).
> `data/` = `resp_classification/data/`.

---

## 0. The headline result (what the poster says)

- **Valence decodes cleanly** (~0.80 window-level, **1.00 recording-level** majority vote,
  shuffle control ~0.52). Driven mostly by **breathing rate** differing between conditions.
- **Rank does NOT decode** (all models ≈ chance under strict LOSO, indistinguishable from
  label-shuffle). Documented reason: rank is **pair-dependent, not mouse-intrinsic** (in a
  triad the same mouse is dominant to one cagemate and subordinate to another), so focal
  respiration alone is underdetermined — and rank shows little respiration-rate difference.
- The poster needs, as a **foundation** under the classifier story: behavioral plots + simple
  per-session respiration-rate plots (§4).

---

## 1. ⚠️ Cohort decision — USE / AVOID

There are **two rank/cagemate cohorts**. They differ a lot because **one has a baseline
recording and one doesn't**. Use only the baseline cohort.

| | ✅ **USE — Baseline cagemate (Aim1 / ECG_cohort1)** | ❌ **AVOID — BLA** |
|---|---|---|
| Why | Has baseline recordings; the correct/trusted set | **No baseline** → large cohort differences |
| CM H5 (Dropbox) | `…\ECG_cohort1\Aim1\AIM1\Day1_new\CM interactions\cm_h5\baseline_cagemate_interactions_h5_outputs` | `…\BLA_ChR_resp_pilot1 (SS)\BLA_resp_ChR_cm_hc\h5_outputs` |
| CM BORIS (Dropbox) | `…\CM interactions\cm_boris\baseline csv` | `…\BLA_ChR_resp_pilot1 (SS)\BLA_resp_ChR_cm_boris` |
| Local mirror | `E:\Aim1\AIM1\Day1_new\resp_CM_h5\baseline_cagemate_interactions_h5_outputs` (+ baselines `E:\Aim1\AIM1\Cagemate_bl_no_ISO\h5_outputs`) | `D:\BLA_ChR_resp\BLA_resp_ChR_cm_hc\h5_outputs` |

**Valence cohort** (uncomplicated, single cohort — Aim1 RI1/RI2):
- H5 (Dropbox): `…\ECG_cohort1\Aim1\AIM1\Day1_new\Resp_h5` (RI1/RI2 interactions + `BLRI` baselines)
- BORIS (Dropbox): `…\ECG_cohort1\Aim1\AIM1\Day1_new\boris_csv_RI1 and RI2`
- Local mirror: `E:\Aim1\AIM1\Day1_new\resp_h5`

> **All Dropbox roots** start with `C:\Users\thoma\UF Dropbox\Thomas Heeps\Padilla-Coreano Lab\2025\`.
> Several notebooks read from the `E:`/`D:` local mirrors instead of Dropbox — same data.
>
> **Rank cohort caveat:** the rank pipelines currently ingest **both** cohorts and tag a
> `Cohort` column; the interactions classifier has a commented toggle to drop a cohort. For
> the poster, restrict to the baseline (Aim1) cohort. See
> [rank preprocessing notes](../rank-classification/preprocessing_docs/PREPROCESSING_NOTES.md).

---

## 2. Color palettes (canonical for the poster)

```python
# RANK / CAGEMATE  (yellow + blue)
rank_palette = {
    "Dominant":    (28/255,  48/255,  79/255),   # navy   ≈ #1C304F
    "Subordinate": (213/255, 176/255, 50/255),   # mustard ≈ #D5B032
}
GRAY = "#6B6B6B"

# VALENCE  (light blue + pink)
valence_palette = {   # code keys are the condition labels RI1 / RI2
    "RI1": "#53A1A9",   # teal / light blue   (BLUE)
    "RI2": "#A95376",   # pink / purple       (PURPLE_PINK)
}
# Poster shows these as Positive vs Negative — teal≈Positive, pink≈Negative
# (⚠️ confirm which RI = positive vs negative before final captions).
```

Secondary palettes seen in the notebooks (for reference, not the main scheme):
- **Sniffing behavior type:** facial `#4C72B0`, body `#DD8452`, anogenital `#55A868`.
- **Breath annotation accents:** inhale `#4C78A8`, exhale `#F58518`, pause `#B279A2` (arrows `#7A4F84`).
- **Classifier-performance ("lab") palette** (r34logistic notebooks): valence `#15616F`/`#2A7F7F` teal, rank `#9C4221` rust / `#6E6E6E` gray; learning curves train `#3266ad`, val `#d85a30`, chance line at 0.5.

> **Palette gotcha:** [`cm_iteration1.ipynb`](../../../cm_iteration1.ipynb) (the behavior/resp-rate
> notebook, §4) colors Dominant/Subordinate with the **teal/pink** valence colors, not
> navy/mustard. Re-color those to `rank_palette` for consistency on the poster.

---

## 3. Data files in `data/` — which pipeline each belongs to

| File | Task | Level | Produced by | Consumed by | Status |
|---|---|---|---|---|---|
| `raw-breathmetrics-features-session-3sec-subjectnames` | Valence | session 3 s windows | `feature-extraction-session-3sec.ipynb` | `valence-classification-session-3sec.ipynb` | ✅ **poster** |
| `raw-breathmetrics-features-session-2sec-subjectnames` | Valence | session 2 s windows | `feature-extraction-session.ipynb` | `valence-classification-session.ipynb` | ✅ alt |
| `Raw-Breathmatrics-Features`, `…-More` | Valence | interaction bout | `feature_extraction-interactions-valence.ipynb` | `valence-classification-interactions.ipynb` | ⚠️ weak result |
| `raw-breathmetrics-features-session-3sec-rank` | Rank | 3 s windows | `rank-featureextraction-3sec.ipynb` | `rank_classification_session.ipynb` | ✅ **poster panels** |
| `Raw-Breathmatrics-Features-Ran-Interactions`² | Rank | interaction bout | `feature_extraction-interactions-rank.ipynb` | `rank-classification-interactions.ipynb` | ⚠️ chance result |
| `Raw-Breathmatrics-Features-Rank-balanced-Interactions` | Rank | interaction bout (balanced) | ″ | ″ | ⚠️ |
| `resp_paths_interactions_rank.pkl` | Rank | trial→h5 path lookup | ″ | ″ | helper |
| `raw-breathmetrics-features-session`, `…-session-2sec` | mixed | — | older attempts | — | 🗃️ superseded / stale (other stale files deleted 2026-10-27) |

² Typo "**Ran**" (not "Rank") is real in the save name — match it exactly.

---

## 4. Poster foundation plots

### ✅ Consolidated script (built) — `poster-code/poster_behavior_resp_plots.py`
Single self-contained script that regenerates all foundation plots from the **correct
cohorts** (valence RI1/RI2; baseline Aim1 cagemate — **BLA excluded**). Reads Dropbox paths
(auto-falls back to `E:` mirror), imports `resp_helper_functions`. Writes **7** transparent
figures (`.svg` + `.png`) to `poster-plots/behavior and resp rate/` — behavior plots emit **both
actor conventions** per cohort so the user can pick:

| File | Plot | Groups | Palette |
|---|---|---|---|
| `behavior_valence_subject_only` / `_both_actors` | total sniffing time by behavior | Positive vs Negative | teal/pink |
| `behavior_rank_subject_only` / `_both_actors` | total investigation time by behavior | Dominant vs Subordinate | navy/mustard |
| `resp_rate_valence` | whole-session respiration rate | Positive vs Negative | teal/pink |
| `resp_rate_rank` | whole-session respiration rate | Dominant vs Subordinate | navy/mustard |
| `resp_rate_rank_interactions` | resp rate **during** sniff bouts | Dominant vs Subordinate | navy/mustard |

Run: `conda run -n biopipeline-env python poster_behavior_resp_plots.py` (needs `neurokit2`;
`base` env lacks it — `biopipeline-env`/`ephys_env` have it). Latest run: rank = 7 mice / 4 cages
(Sub 4 / Dom 3, cage 1 kept), valence = Pos 7 / Neg 8. Results: valence resp rate Positive≫Negative
(clean); rank whole-session Dom breathes *slower* than Sub, and Dom≈Sub during interactions
(consistent with "rank doesn't decode" from the classifier).

> ⚠️ The consolidated plots differ slightly from the originals (actor filtering, recording set,
> signal cleaning, valence metric). These are catalogued in
> [PLOT_PROCESSING_DIFFERENCES.md](./PLOT_PROCESSING_DIFFERENCES.md), which also documents that
> **dead signal is not trimmed within a recording** (only whole-subject `5_13` is dropped).

### ✅ Decoder metrics — `poster-code/decoder_metrics_table.py`
Uniform, reproducible protocol over all four task × window-type feature matrices (valence/rank ×
session/interaction) → `data/decoder_metrics_summary.csv` + `_per_fold.csv`, and a Model-Performance
+ Top-5-Features poster panel per task → `poster-plots/decoder-metrics/decoder_panels.{svg,png}` +
`NOTES.md`. CV = leave-one-group-out (mouse for valence, cage/LOCO for rank). Run:
`conda run -n ephys_env python decoder_metrics_table.py` (needs numpy≥2 to read the pickles —
`biopipeline-env` cannot). Result: valence ~0.81, rank ~chance (0.46–0.52).

### Original sources (for reference)

### 4a. Behavioral plots (rank / cagemate)
**Source: [`cm_iteration1.ipynb`](../../../cm_iteration1.ipynb) at the REPO ROOT** (not inside
`resp_classification/`). Reads the baseline Aim1 cohort from the `E:` mirror (`resp_paths_bl` +
`resp_paths_cm`, cells 1–2).

- **"Resident Investigation by Rank (Pair-Housed)"** boxplots — Face / Body / Anogenital
  investigation time, Dominant vs Subordinate — cells ~12, 13, 17 (this is the attached poster plot).
- Related variants: cells 14, 16, 23, 26.

### 4b. Whole-session respiration-rate plots (rank / cagemate)
Also **`cm_iteration1.ipynb`**:
- **"Baseline Respiration Rate (Dominant vs Subordinate)"** — cells ~61, 62.
- **Whole-session respiration** Dom vs Sub — cells ~69, 79, 80 ("Whole-session window").
- Paired-by-subject respiration rate ("by Condition", "Habituation vs Cagemate") — cells ~64, 65.
  ↳ The attached **Positive-vs-Negative paired boxplot** (connected dots) most likely comes from
  one of these paired cells — **confirm exact cell** when consolidating.

### 4c. Valence behavior + respiration-rate plots
- **[`valence_sniffing_behavior_overview.ipynb`](../valence_sniffing_behavior_overview.ipynb)** —
  per-recording sniffing bout counts, total sniffing time, and ethograms (RI1 vs RI2). Pure BORIS,
  no respiration. Behavior colors `#4C72B0 / #DD8452 / #55A868`.
- **[`downsample_50hz_analysis.ipynb`](../valence-classification/downsample_50hz_analysis.ipynb)** —
  "**Session-average respiration rate per valence recording**" bar chart (RI1 tend higher than RI2).
  This is the whole-session respiration-rate figure for valence.

### 4d. Helper for per-window/bout respiration rate
`get_sniff_respiratory_rate(signal, time, start, end, fs)` in
[`resp_helper_functions.py`](../resp_helper_functions.py) computes mean instantaneous breathing rate
in a window (used to build all rate plots).

---

## 5. Classifier attempts inventory (the "what I tried" table)

Every decoding attempt, so nothing gets re-run blindly. **CV: LOSO/LOMO = leave-one-mouse-out;
strict LOSO = held-out mouse never appears as subject *or* cagemate in train.**

| # | Task | Granularity | Notebook | Models | CV | Result | Verdict |
|---|---|---|---|---|---|---|---|
| V1 | **Valence** | interaction bout (~233 samples) | `valence-classification/valence-classification-interactions.ipynb` | LogReg, SVM, RF | 5-fold | LR 0.59, SVM 0.63, RF 0.50 (bal-acc) | ❌ weak |
| V2 | **Valence** | session 2 s windows | `valence-classification/valence-classification-session.ipynb` | LogReg, SVM, LinSVM, RF | LOMO + StratGroupKFold | LR 0.80, SVM 0.80 (window) | ✅ strong |
| **V3** | **Valence** | **session 3 s windows** | `valence-classification/valence-classification-session-3sec.ipynb` | LogReg, SVM, RF, GB | nested LOMO | 0.79–0.81 window, **1.00 recording**, shuffle 0.52 | ✅✅ **poster** |
| R1 | **Rank** | interaction bout (~281) | `rank-classification/rank-classification-interactions.ipynb` | LogReg, SVM, RF, XGB | LOSO (inner GroupKFold) | LR ~0.46–0.53, SVM ~0.56, RF ~0.2, XGB ~0.35 | ❌ chance |
| **R2** | **Rank** | **3 s windows** | `rank-classification/rank_classification_session.ipynb` | LogReg, SVM, RF, XGB | nested **strict LOSO** (+relaxed, LOPO) | 0.42–0.50 window, shuffle 0.49 | ❌ chance (**poster panels**) |
| C1 | Valence + Rank + **transfer** | session windows, resp-only ("noBoris") | repo-root `r34logistic_post_sfn_5_noBoris.ipynb` | LogReg | GroupKFold / leave-one-pair-out | Valence AUC 0.62, Rank ~0.64, Valence→Rank 0.39 | mixed |
| C2 | same, extended | session windows, resp-only | *(local copy deleted 2026-10-27)* | LogReg | 10-fold StratKFold + LOPO | **Valence AUC 0.83**, Rank 0.63, transfer 0.37 | valence ✅ / rank ❌ |
| C3 | combined-cohort assembly | Aim1 + BLA merge + classify | *(local copy deleted 2026-10-27; original at repo root `r34logistic_combine_cm_data.ipynb`)* | LogReg | LOSO | valence teal / rank gray AUC bars | provenance/merge notebook |

**Feature-extraction notebooks (no classifier — upstream of the above):**
`rank-classification/feature_extraction-interactions-rank.ipynb`,
`rank-classification/rank-featureextraction-3sec.ipynb`,
`valence-classification/feature_extraction-interactions-valence.ipynb`,
`valence-classification/feature-extraction-session-3sec.ipynb`,
`feature-extraction-session.ipynb`,
`valence-classification/downsample_50hz_analysis.ipynb` (also the valence resp-rate bar chart, §4c).

---

## 6. Shared code

- **[`resp_helper_functions.py`](../resp_helper_functions.py)** (~2050 lines) — the shared library. Key groups:
  - *Signal:* `load_rank_resp_signal_interactions`, `load_clean_resp_signal`,
    `load_clean_resp_signal_cagemate_rank`, `preprocess_resp_to_target_rate` (→ **400 Hz**,
    lowpass→downsample→NeuroKit bandpass 0.1–20 Hz).
  - *Rate/behavior:* `get_sniff_respiratory_rate`, `load_clean_boris` (keeps facial/body/anogenital sniffing).
  - *Catalog & filename parsing:* `build_interaction_boris_catalog` (pairs Aim1+BLA H5↔BORIS,
    returns `bla_rank_map`), `normalize_subject_id`, `subject_id_from_trial_key`, `cohort_from_trial_key`.
  - *QC/exclusions:* `build_upstream_exclusion_set`, `filter_interaction_paths` (cohort-aware).
  - *Features:* `fit_bm_session`, `extract_bout_features`, `sample_random_nonoverlapping_windows`,
    `extract_features_from_windows`. `METADATA_COLS` lists non-feature columns.
- **[`breathmetrics.py`](../breathmetrics.py)** — self-contained Python port of BreathMetrics.
  `bmObject(resp, srate, dataType='rodentAirflow').estimateAllFeatures()` → per-breath
  inhale/exhale onsets, durations, volumes, peak flows, breathing rate. The engine every
  `extract_*` / `fit_bm_session` wraps.
- **Rank preprocessing deep-dives** (already written, authoritative for the rank *interactions*
  pipeline): [`PREPROCESSING_NOTES.md`](../rank-classification/preprocessing_docs/PREPROCESSING_NOTES.md)
  and [`RNN_DATA_GUIDE.md`](../rank-classification/preprocessing_docs/RNN_DATA_GUIDE.md).

---

## 7. Status of the old open items (mostly resolved 2026-07-13)

- [x] **Restrict rank to the baseline (Aim1) cohort** — DONE. BLA dropped from both rank pipelines
      (session 3-sec + interactions) and from the poster script; all rebuilt Aim1-only + LOCO.
- [x] Re-color rank plots teal/pink → **navy/mustard** `rank_palette` — DONE in the poster script.
- [x] **RI1 = Positive, RI2 = Negative** — confirmed and used throughout.
- [x] Foundation plots pulled out of the repo-root `cm_iteration1.ipynb` into the poster script.
- [ ] Output pickle name `Raw-Breathmatrics-Features-Ran-Interactions` still has the **"Ran" typo** —
      match it exactly when reading.
- [ ] Signal is re-derived to **400 Hz from raw**, not the stored 100 Hz `resp_clean` — by design.
- [ ] **Behavior actor convention** — script emits BOTH per cohort; user still to pick which to keep.
- [ ] **Dead signal** is not trimmed within a recording (see PLOT_PROCESSING_DIFFERENCES.md) — revisit
      only if a kept recording turns out to have a large flat stretch.
```
