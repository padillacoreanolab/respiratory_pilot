# Full-recording feature extraction — handover

Moves from 100 random windows per session to **every** window of every recording, across
both Aim1 cohorts. Window length is **2 s** (`WINDOW_SEC` in the script; output filenames
follow it). Already run locally (2026-08-14): **7,780 windows × 39 cols from 27 recordings
/ 7 mice**, ~2 min at `--jobs 6`. Re-run only if paths or data change.

Outputs live in **`../data/whole-session-final/`** — the final whole-session set. Anything
matching `../data/full-recording-*` at the top level is a superseded earlier run.

## Files

| File | What |
|---|---|
| `full_recording_feature_extraction.py` | The extraction. Catalog + full-recording tiling FE, parallel via `--jobs`. |
| `feature-exploration.ipynb` | Four checks (lengths/yield, distributions, redundancy, PCA) defined once and run **separately on valence and on rank**. Loads the outputs below. |
| `../data/whole-session-final/full-recording-features-2sec.parquet` (`.pkl` fallback) | Window-level feature matrix. |
| `../data/whole-session-final/full-recording-manifest.csv` | One row per recording: duration, breaths, windows possible/kept, yield. |
| `../data/whole-session-final/full-recording-pcs-{valence,rank}.pkl` | 9 PC scores (95% of variance, de-correlated features) + window metadata — the input for k-means / HMM. |

## Running

```bash
export RESP_DATA_ROOT=/blue/.../ECG_cohort1/Aim1/AIM1/Day1_new
python full_recording_feature_extraction.py --jobs 16 --out /path/to/data
```

Flags: `--dataset {valence,rank}`, `--limit N`, `--root`, `--out`, `--jobs`.

Needs `neurokit2`, `breathmetrics`, `h5py`, `scipy`, `sklearn`, numpy ≥ 2 (pickles are
numpy-2). Locally conda env **`ephys_env`**; `biopipeline-env` has numpy < 2 and cannot
unpickle these. `pyarrow` is absent locally so it falls back to `.pkl` — install it on HPC
for parquet.

Cost: ~29 s per recording single-threaded, so 27 recordings ≈ 13 min serial. Not the
bottleneck — the state modelling is.

## Cohort and exclusions

**35 H5 on disk → 27 extracted** (26 valence + 9 rank). BLA is excluded project-wide.
The exclusion set matches [`../docs/EXCLUSIONS_LOG.md`](../docs/EXCLUSIONS_LOG.md) exactly.
**Do not diverge from that list — ask the user first**, including for analyses that look
like they don't need BORIS.

Dropped (`DROP` in the script):

| Recording | Why |
|---|---|
| `BLRI_s5_13_…175809` | H5 has no `resp` dataset at all, only `ekg` (found 2026-08-07) |
| `RI1_s5_13_…181801` | whole-session resp rate outlier (~4.9 Hz vs ~7.5) |
| `RI2_s5_13_…184136` | resp trace flat/broken (std ≈ 20 vs ~700–1000) |
| `CM_s3_5_d3_6_…170708` | duplicate 3_5 cagemate session; `…160001` is canonical (has BORIS) |
| `RI1_s3_5_…105014` | no BORIS export |
| `CM_s3_6_sub3_5_…174348` | BORIS export is empty |
| `RI2_s3_5_…112618` | incomplete session (213 s vs ~605 s), unpaired once `RI1_3_5` went |
| `CM_s3_5_d3_6_…160001` | short (355 s) and, with 3_6's cagemate recording gone, the lone occupant of cage 3 |

**A recording without BORIS is dropped, full stop** — signal quality notwithstanding, a
recording with no behavior scoring does not enter any dataset.

**Subject 3_5 is now out entirely** (no valence recording survives, and its cagemate
recording was short and left cage 3 a singleton). So **valence = 7 mice** and
**rank = 6 mice / 3 complete cages** {1_1,1_2}, {2_3,2_4}, {4_7,4_8} — every leave-one-cage-out
fold now contains both ranks, which removes the degenerate single-class fold.

## Paired recordings

`feature-exploration.ipynb` builds the pairing dict and writes
`../data/whole-session-final/full-recording-pairs.json`. Three families, checked for equal length (±5%):
`valence_within_subject` (21), `cross_dataset_mouse` (18 — the same mice appear in both
datasets), `rank_reciprocal` (3 — one per cage).

**42 pairs, 0 unequal** — every surviving pair is within 5% on duration.

Mice missing an expected recording: **3_6** has no cagemate recording (dropped). 3_5 is gone
from both datasets. Everything else is complete.

## Things to know before analysing

- **Windows within a recording are not independent.** Any p-value over pooled windows is
  anticonservative. Group by `RecordingID` / `Subject` / `Cage` for anything inferential,
  and use grouped CV (LOCO on `Cage` for rank, leave-one-mouse-out for valence).
- **Nothing is Gaussian, and the notebook now says so numerically** (skew, excess kurtosis,
  Shapiro-Wilk W on a 5k subsample — p is ~0 at this n, so read W). Valence: |skew| ≥ 2 in
  14/25, excess kurtosis ≥ 7 in 17/25, **W < 0.95 in 22/25**. Rank: 12/25, 10/25, 18/25.
  Prefer rank/robust methods, or transform first.
- **Six features are heavily zero-inflated — this is real, not a plotting artifact.** The
  inhale-pause trio (`pct_breaths_with_inhale_pause`, `inhale_pause_duty_cycle`,
  `mean_inhale_pause_dur_sec`) is **91.4% exact zeros**; the exhale-pause trio is **56.2%**.
  Most 2 s windows simply contain no detected pause. The table reports `pct_zero` and the
  1st/99th percentiles alongside each histogram, and the histograms are clipped to the
  1st–99th percentile so heavy tails don't squash the bulk into one bar. **A zero-inflated
  spike is not something a Gaussian HMM or Euclidean k-means handles gracefully** — consider
  splitting these into a binary "pause present" indicator plus a magnitude, or dropping them.
- **Several features are algebraic twins** — `n_breaths` ≡ `breathing_rate_hz` ≡
  1/`mean_ibi_sec`; the pause duration / duty-cycle / pct-with-pause families likewise.
  At the **|ρ| > 0.85** cutoff: 9 pairs / 9 features in valence, 14 pairs / 11 in rank.
  Correlated twins split each other's feature importance. The notebook plots a heatmap of
  **only the features involved in an over-threshold pair** (a full 25×25 is unreadable) and
  returns an explicit drop list per dataset. **Redundant in BOTH — safe to drop everywhere:**
  `breathing_rate_hz`, `mean_ibi_sec`, `exhale_pause_duty_cycle`,
  `pct_breaths_with_exhale_pause`, `pct_breaths_with_inhale_pause`. Dataset-specific:
  `inhale_pause_duty_cycle` (valence), `mean_inhale_pause_dur_sec` and `n_breaths` (rank).
  Note rank drops **all three** rate twins at 0.85 — that cluster merges with
  `ventilation_proxy`, which becomes the surviving rate representative.
- **PCA runs on the de-correlated set** (`*_KEEP`: 19 features valence, 18 rank), not all 25
  — otherwise the rate twins get counted three times and dominate PC1. Not cosmetic: PC1
  stops being a rate axis and 95% of variance needs **9 PCs instead of 11** (valence).
- **Compression: 6 PCs → 80%, 8 → 90%, 9 → 95%** (valence); 5 / 7 / 9 (rank). Those 9 PCs are
  **saved** to `../data/whole-session-final/full-recording-pcs-{valence,rank}.pkl`
  (metadata columns + `PC1…PC9`) as the input for later k-means / HMM state work.
- **Only one PC is a single feature.** The save prints a per-PC clean/mixed verdict (top
  loading > 1.5× the next). **PC6 is clean in both datasets and is the pause axis** —
  `mean_inhale_pause_dur_sec` (+0.94) in valence, `inhale_pause_duty_cycle` (+0.96) in rank,
  the same thing under each dataset's chosen representative — but note that feature is 91%
  zeros, so PC6 is essentially a pause-present indicator. PC1–PC5 are genuine blends:
  volume/depth, inspiratory timing, peak flow, exhale duration, duty cycle.
- **Two pause-CV features are ~97% and ~73% NaN** and dropped by the notebook's NaN filter.
  A pause CV is undefined unless a window has ≥2 pauses of that type; median-imputing a
  97%-empty column fakes a well-behaved low-variance feature. Don't silently re-include.
- **`WindowIndex`** is the temporal slot within each recording (gaps = windows rejected by
  the dead-signal or min-4-breath gate), so recordings read back as ordered sequences —
  transition matrices and dwell times need no re-extraction.
- **`MIN_BREATHS_PER_WINDOW = 4`** (valence used 4, rank 5). `n_breaths` is per row, so a
  stricter gate is a filter, not a re-run. At 2 s and ~7 Hz breathing a window holds ~14
  breaths, so the gate is still slack — but it bites harder than at 3 s, which is where the
  lower window yields below come from.
- **Window yield < 0.90 in 5 recordings**, worst `RI2_s2_4_…125648` at 0.68, then
  `RI2_s3_6` 0.76, `CM_s1_2` 0.88, `CM_s1_1` 0.88, `RI2_s1_2` 0.88. Median 0.98.

## Open questions

- **The zero-inflated pause features**: keep as-is, split into indicator + magnitude, or
  drop? They are 91% (inhale) / 56% (exhale) exact zeros and they own PC6.
- **Rank n is now 6 mice / 3 cages** — 3 LOCO folds. Small, but every fold is two-class.
