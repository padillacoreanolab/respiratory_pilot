# Respiration → rank & valence: analysis guide

Can a mouse's **social valence** (positive vs negative interaction) or **social rank** (dominant vs
subordinate) be decoded from its respiration during social interaction / open-cage exploration?
This folder holds Thomas's analyses for that question. This guide is the single reference for
humans and agents: where things are, what data goes in, what is dropped, how signals are processed,
what was tried, and how each figure is made. It is a map: verify a cell before trusting a detail.

Shared code that the whole lab uses lives in the repo's `src/`. Raw data and the H5 converters are
documented in [`docs/DATA_README.md`](../../docs/DATA_README.md).

**Contents:** 1 Results · 2 Folder map · 3 Data · 4 Exclusions (binding) · 5 Signal processing &
features · 6 Rank dataset details · 7 Classifiers & data files · 8 Figures · 9 Environments · 10 Open issues

---

## 1. Headline results

- **Valence decodes.** About 0.80 balanced accuracy at window level and 1.00 at recording level
  (majority vote), with shuffle control around 0.52. Leave-one-mouse-out AUC is 0.85–0.89. The
  signal is mostly **breathing rate**: positive (RI1) > negative (RI2).
- **Rank does not decode.** All models are at chance under leave-one-cage-out (AUC 0.44–0.53), the
  same as a label shuffle. Rank is pair-dependent rather than mouse-intrinsic, and the rank
  difference in respiration rate is small: dominants breathe slightly slower over the whole
  session and the same as subordinates during interactions.
- An SRNN latent decoder (figures only, in `figures/srnn_latent_decoder/`) shows the same pattern:
  valence 0.69 and rank 0.50 balanced accuracy.

---

## 2. Folder map

| Folder | What is in it |
|---|---|
| `resp_lib/` | Shared library: `resp_helper_functions.py` (loaders, BORIS, catalog, feature extraction), `breathmetrics.py` (Python BreathMetrics port), `adaptive_breath_detection.py` (validated peak detector). **Staging area for a later move into `src/`.** |
| `valence_classification/` | `valence_feature_extraction_*` → `data/` pickles → `valence_classifier_*`. Also `valence_downsample_50hz_resp_rate.ipynb` and `valence_sniffing_behavior_overview.ipynb` (BORIS-only bout counts / ethograms). |
| `rank_classification/` | `rank_feature_extraction_*` → `data/` pickles → `rank_classifier_*`. |
| `rank_and_valence/` | Code that covers both tasks. `decoder_metrics_table.py` (balanced accuracy / F1 + panels) and `decoder_roc_curves.ipynb` (ROC / AUC) run uniform decoders on the BreathMetrics pickles and make the **poster decoder figures**. `behavior_resp_rate_plots.py` makes the behavior and resp-rate plots. `full_recording_feature_exploration.ipynb` covers distributions, redundancy and PCA of every 2 s window (`data/whole-session-final/`). |
| `keypoint_moseq/` | keypoint-MoSeq setup on SLEAP output (`kpms_project/` outputs are git-ignored). |
| `figures/` | All final plots, by category (§8). |
| `data/` | Feature pickles and decoder CSVs (git-ignored; §7). |
| `presentations/` | Slide decks and their build code, including the respiration methods deck (git-ignored). |
| `review&delete/` | Superseded docs, data and leftovers waiting for deletion. |
| `biopipeline-env.yml` | Conda environment (§9). |

Notebook naming: `<task>_feature_extraction_<window>` writes a pickle and `<task>_classifier_<window>`
reads it. Windows are `session_2sec` / `session_3sec` (random non-overlapping windows over the
whole session) or `interactions` (sniff-bout windows).

---

## 3. Data

**Cohorts** (see [`docs/DATA_README.md`](../../docs/DATA_README.md) for numbering and sensor failures):

| Cohort | Experiment | Recordings | Raw complete | h5 | BORIS | Type |
|---|---|---|---|---|---|---|
| **1 Aim1** (`2025\ECG_cohort1`) | Cagemate baseline | 9 | 9 | 9 | 9 | wireless, merged |
| | Cagemate 1-day / 10-day isolation | 8 / 8 | 8 / 8 | 8 / 8 | 0 | wireless, merged |
| | Valence BLRI (baseline) / RI1 (positive) / RI2 (negative) | 9 / 9 / 9 | 9 each (added 2026-11-10) | 9 each | 0 / 8 / 9 | wireless, merged |
| **2 BLA pilot 1** (`2025\BLA_ChR_resp_pilot1 (SS)`, Jan 2026) | Cagemate home cage | 24 | 24 | 24 | 24 | tethered |
| | Baseline `bl_cc` | 23 | 23 | **0** | 0 | tethered |
| | RI1 d1 / RI1 d2 / RI2 post d1 / RI2 post d2 | 24 / 24 / 23 / 23 | 24 / 23 / 23 / 23 | 24 / 24 / 23 / 23 | same as h5 | tethered |
| | OFT / `wrong_pulse` | 24 / 7 | 24 / 6 | 24 / **0** | 0 | tethered |
| **3 BLA cohort 2** (`2026\BLA_ChR_resp_cohort2`, Jul 2026) | Cagemate home cage / RI1 d1+2 / OFT | 11 / 44 / 22 | 11 / 42 / 22 | 11 / 44 / **0** | 11 / 44 / 0 | tethered |

"Raw complete" means the session has the `.rec`, an analog export with `ECU_Ain1` + `timestamps.dat`,
`.videoTimeStamps`, and the mp4. Per-recording detail: `review&delete/data_inventory/DATA_INVENTORY_recordings.csv`
(made by `build_data_inventory.py`, a read-only Dropbox scan, now in the same folder).

**Inputs the analyses here actually use.** All are cohort 1 (Aim1). Dropbox root:
`C:\Users\thoma\UF Dropbox\Thomas Heeps\Padilla-Coreano Lab\2025\ECG_cohort1\Aim1\AIM1\Day1_new\`.
The `E:\Aim1\AIM1\Day1_new\` mirror holds the same data.

| Dataset | Respiration H5 | BORIS |
|---|---|---|
| Valence (RI1 = positive, RI2 = negative, BLRI = baseline) | `Resp_h5\` (old format; `BLRI_s5_13` has no `resp`) | `boris_csv_RI1 and RI2\` |
| Rank (baseline cagemate) | `CM interactions\cm_h5\baseline_cagemate_interactions_h5_outputs\` | `CM interactions\cm_boris\baseline csv\` |

**Newer H5s that no analysis uses yet:** Aim1 valence rebuilt on 2026-11-10 (v2 format,
`Aim1\new_resp_h5_valence\{RI1,RI2,BLRI}`), and all cagemate cohorts 1–3 rebuilt on 2026-10-27 (v2 +
`behavior_aligned/`, repo `exports\cagemate_h5\`). Other H5 folders: Aim1 1d/10d in repo
`src\file_conversion\h5_outputs\`; pilot 1 in each experiment's `h5_outputs`; cohort 3 cagemate is v2.

---

## 4. Exclusions (binding)

This is the binding list of everything left out of each dataset. Update it in the same change as
any new drop, and **ask before diverging from it**.

> **Rule:** a recording with no usable BORIS export is dropped everywhere, including window-level
> pipelines that never read behavior.

**Cohorts.** Rank uses cohort 1 (Aim1) baseline cagemate only. Cohort 2 (BLA pilot) is excluded
because it has no baseline, and cohort 3 was recorded after these datasets were built. Valence uses
cohort 1 BLRI / RI1 / RI2.

**Rank: cohort 1 cagemate**

| Excluded | Pipelines | Why |
|---|---|---|
| Subject `3_6` | all | BORIS export is empty |
| `CM_s3_5_d3_6_…170708` | all | Duplicate 3_5 session with no BORIS (`…160001` is kept) |
| `CM_s3_5_d3_6_…160001` | full-recording only | Short (355 s vs about 605 s) and the only recording left in cage 3 |
| BORIS `CM_s4_8_sub4_7_20250623.csv` | all | Bad duplicate: tab-separated, no timestamp. Use `…182649_VT`. |
| *Kept:* `1_1`, `1_2` | — | Fighting occurred, but the cohort is too small to drop a cage |

**Valence: cohort 1 RI**

| Excluded | Pipelines | Why |
|---|---|---|
| Subject `5_13` (RI1, RI2, BLRI) | all | RI1 rate is an outlier (about 4.9 vs 7.5 Hz); RI2 is flat (std about 20 vs 700–1000); the old BLRI H5 has no `resp` dataset |
| `RI1_3_5` | all | No BORIS |
| `RI2_3_5` | full-recording only | Incomplete (213 s) and unpaired |

**Resulting sets**

| Pipeline | Rank | Valence |
|---|---|---|
| 3-sec windows and interactions | 7 mice / 4 cages: Sub {1_1, 2_3, 3_5, 4_7}, Dom {1_2, 2_4, 4_8} | Pos n = 7, Neg n = 8 |
| Full-recording (`data/whole-session-final/`) | 6 mice / 3 complete cages | 7 mice |

The full-recording set has 27 recordings, 7,780 windows and 7 mice.

**Processing drops**

| Stage | Rule | Applies to |
|---|---|---|
| Bouts | Keep facial / body / anogenital sniffing with 0.5 s ≤ duration ≤ 20 s (removes one ~400 s anogenital scoring error) | Behavior and bout analyses |
| Bouts | Min-sniff gate: ≥ 10 sniff bouts per subject, counting both actors (keeps `1_2`) | Rank interactions |
| Windows | Min-breaths gate on random windows | Random-window feature extraction |
| Windows | Every 2 s window kept; dead signal inside a recording is **not** trimmed | Full-recording |
| Features | `cv_inhale_pause_dur`, `cv_exhale_pause_dur` dropped (> 50 % NaN), leaving 25 | Full-recording |
| Features | Redundant pairs pruned at \|ρ\| > 0.85; PCA on the pruned set (9 PCs = 95 % variance) | Full-recording |

**Known caveats (kept in the data)**
- Actor convention differs by task: rank counts both actors, valence counts subject-initiated bouts. See §8.
- `RI1_s2_3_…104059_ss.csv` has only `subject` bouts scored, so for RI1 2_3 the both-actor count equals the subject-only count.
- `CM_s1_2_sub1_1_…133932.1.csv` (non-VT) has 7 stray `no focal subject` bouts. The pipelines use the `_VT` export.
- Valence BORIS was re-scored with longer bouts after the old plots were made. The source of truth is `Day1_new\boris_csv_RI1 and RI2`.
- Inhale-pause features are 91 % exact zeros and exhale-pause features 56 %. This matters for Gaussian or Euclidean models.
- In rank LOCO, cage 3 is a singleton {3_5}, so its fold is single-class.
- Dead signal is handled only at whole-subject level (`5_13`). A partial flat stretch in a kept recording lowers its rate.

---

## 5. Signal processing & features

**Loaders** (`resp_lib/resp_helper_functions.py`): valence uses `load_clean_resp_signal(h5, target_rate=400)`;
rank uses `load_rank_resp_signal_interactions(h5, target_rate=400, prefer_raw=True)`. Both read the raw
~20 kHz trace (`resp`, or `analog/*ECU_Ain1`) and call `preprocess_resp_to_target_rate`:
1. Butterworth low-pass, order 4, cutoff `(target/2)/nyquist`.
2. Downsample to **400 Hz** with `resample_poly` (polyphase).
3. NeuroKit band-pass Butterworth **0.1–20 Hz**, order 2. Do not filter again.

The stored 100 Hz `resp_clean` is used only when raw is missing. `time[0]` is not always 0, so mask by
time, not by sample index. The old FFT `resample` + 0.1–15 Hz loader (`cm_iteration1.ipynb`,
`r34_analysis5.ipynb` at repo root) is retired; it shifts rates by < ~0.2 Hz.

**Respiration rate** for the descriptive plots: inhalation peak count / window duration,
`find_peaks(distance = fs·0.0833)` (≥ 12 Hz refractory). `get_sniff_respiratory_rate` gives the mean
instantaneous rate within a window.

**BreathMetrics features.** `bmObject(resp, srate, dataType='rodentAirflow').estimateAllFeatures()` is fit
on the **full session** (`fit_bm_session`), then breaths are assigned to a window if their inhale onset
falls inside it. Per-window features (`extract_bout_features` / `extract_features_from_windows`):
`n_breaths, breathing_rate_hz, mean_ibi_sec, cv_ibi, mean/cv_inhale_dur, mean/cv_exhale_dur, ie_ratio,
mean_peak_insp_flow, mean_peak_exp_flow, cv_peak_insp_flow, mean_inhale_vol, mean_exhale_vol,
mean_tidal_vol_proxy, cv_inhale_vol, ventilation_proxy`, inhale/exhale pause percent, mean/cv duration
and duty cycle, and `inhale_duty_cycle, exhale_duty_cycle`. Interaction pipelines add `vol_asymmetry`,
`peak_flow_ratio`, per-subject and per-trial z-scores, `PreBoutRespRate`, `resp_rate_delta`.
`METADATA_COLS` lists the non-feature columns.

Interaction windows are 2 s, onset-anchored (`anchor="onset"`, `min_dur=0.5`, `pre_bout_window=2.0`,
`min_breaths_per_bout=2`).

**Breath detection.** The legacy BreathMetrics detector misses about 6–8 % of real breaths (session-wide
threshold). `adaptive_breath_detection.py` (local prominence, v1.0.0, validated 2026-10-09) is opt-in:
`rh.DEFAULT_DETECTOR = "adaptive_prominence"`. All existing pickles use the legacy detector. Method,
validation and usage: [`review&delete/breath_detection/README.md`](review&delete/breath_detection/README.md) (moved there 2026-11-24; not used by any analysis).

---

## 6. Rank dataset details

**Rank map (Aim1, hard-coded):** Sub = `1_1, 2_3, 3_5, 4_7`; Dom = `1_2, 2_4, 3_6, 4_8`. Cages are
pairs `{1_1,1_2}, {2_3,2_4}, {3_5,3_6}, {4_7,4_8}`. The first index of a subject id is the cage. Each pair
was recorded in both directions (each mouse as focal). **Split by cage (LOCO), never by bout.**

**Filenames**
- Aim1 H5: `CM_s{sub}_{rank}{partner}_{date}_{time}_merged.h5`, e.g. `CM_s1_1_sub1_2_20250623_111352_merged.h5`.
  The `d` / `sub` token is informational; the rank map above is authoritative.
- Aim1 BORIS: `CM_s…_{date}_{time}.{n}.csv` or `…_VT.csv`. Prefer `_VT` when a session has several.
- Trial keys: `AIM1cm_{sub}_{partner}_{date}_{time}`; subject = `"_".join(key.split("_")[1:3])`.

**BORIS gotcha:** the raw CSV column `Subject` is the **actor** (`subject` / `social_agent`), not the
mouse id. `load_clean_boris` renames it to `Initiator`. The respiration is always the focal mouse's,
even during `social_agent`-initiated bouts.

**BLA (excluded, but still in the helper code).** BLA filenames encode role `d` / `i` / `s` (an `i` role
has no rank). Cages are triads with per-cage numbering (`X_1..X_3`), and BLA ids `1_1, 1_2, 2_3` collide
with Aim1 ids. The rank map is keyed by subject id only, so if BLA is ever pooled, key it by
`(cohort, subject)`. BLA sessions have split `VT_SA` / `VT_CM` exports; keep `VT_SA`
(`dedupe_h5_boris_exports`).

**Sequence models (RNN / SRNN).** The feature pickles hold bout summaries. For sequences, reuse
`build_interaction_boris_catalog`, `load_rank_resp_signal_interactions` and `load_clean_boris`, and slice
the 400 Hz trace per bout (2 s onset-anchored ≈ 800 samples). Apply §4 and carry `Trial, Subject,
Cohort, Rank, Behavior, Initiator, Start, Stop` with each sequence.

---

## 7. Classifiers & data files

**Every decoding attempt** (balanced accuracy unless stated):

| Task | Windows | Notebook | CV | Result |
|---|---|---|---|---|
| Valence | interaction bouts | `valence_classifier_interactions` | 5-fold | 0.50–0.63, weak |
| Valence | session 2 s | `valence_classifier_session_2sec` | LOMO | ~0.80 |
| **Valence** | **session 3 s** | `valence_classifier_session_3sec` | nested LOMO | **0.79–0.81 window, 1.00 recording**, shuffle 0.52 |
| Rank | interaction bouts | `rank_classifier_interactions` | LOSO / LOCO | 0.2–0.56, chance |
| **Rank** | **3 s windows** | `rank_classifier_session_3sec` | nested strict LOSO / LOCO | 0.42–0.50, chance |
| Both | all four matrices, uniform protocol | `rank_and_valence/` | valence LOSO, rank LOCO | valence ~0.81, rank 0.46–0.52 |
| Both + transfer | resp-only session windows | repo-root `r34logistic_post_sfn_5_noBoris.ipynb` | GroupKFold / leave-one-pair-out | valence AUC 0.62–0.83, rank ~0.63, transfer ~0.38 |

The `rank_and_valence/` protocol: fixed (untuned) LogReg (L2), RBF-SVM and RandomForest (300 trees,
depth 5), all `class_weight="balanced"`; respiration features only (BORIS columns excluded). Full
methodology and caveats: [`figures/decoder_metrics/NOTES.md`](figures/decoder_metrics/NOTES.md) and
[`figures/decoder_roc/NOTES.md`](figures/decoder_roc/NOTES.md). Don't read direction off the rank
coefficients: the model is at chance, and collinear rate features flip sign.

**Files in `data/`** (git-ignored):

| File | Task / windows | Written by | Read by |
|---|---|---|---|
| `raw-breathmetrics-features-session-3sec-subjectnames` | Valence, 3 s | `valence_feature_extraction_session_3sec` | `valence_classifier_session_3sec`, decoders |
| `raw-breathmetrics-features-session-2sec-subjectnames` | Valence, 2 s | `valence_feature_extraction_session_2sec` | `valence_classifier_session_2sec` |
| `Raw-Breathmatrics-Features`, `…-More` | Valence, bouts | `valence_feature_extraction_interactions` | `valence_classifier_interactions`, decoders |
| `raw-breathmetrics-features-session-3sec-rank` | Rank, 3 s | `rank_feature_extraction_session_3sec` | `rank_classifier_session_3sec`, decoders |
| `Raw-Breathmatrics-Features-Ran-Interactions` (typo "Ran" is real), `…-Rank-balanced-Interactions`, `resp_paths_interactions_rank.pkl` | Rank, bouts | `rank_feature_extraction_interactions` | `rank_classifier_interactions`, decoders |
| `decoder_metrics_summary.csv`, `_per_fold.csv`, `decoder_roc_auc_summary.csv` | both | `rank_and_valence/` | — |
| `whole-session-final/` (features, manifest, pairs, PCs) | both, every 2 s window | `full_recording_feature_extraction.py` (now in `review&delete/`) | `rank_and_valence/full_recording_feature_exploration.ipynb` |

Pickles need numpy ≥ 2 to load (§9).

---

## 8. Figures

| `figures/…` | Contents | Made by |
|---|---|---|
| `behavior_and_resp_rate/` | `behavior_{rank,valence}_{subject_only,both_actors}`: investigation time by sniff type. `resp_rate_{rank,valence}`: whole-session rate. `resp_rate_rank_interactions`: rate during sniff bouts. | `rank_and_valence/behavior_resp_rate_plots.py` |
| `decoder_metrics/` | `decoder_performance`, `decoder_features` (top-5 LogReg coefficients), `decoder_panels`, `NOTES.md` | `rank_and_valence/decoder_metrics_table.py` |
| `decoder_roc/` | `<task>_<model>_roc`, `<task>_<model>_paired`, `decoder_roc_panels_<model>`, `NOTES.md` | `rank_and_valence/decoder_roc_curves.ipynb` |
| `srnn_latent_decoder/` | SRNN latent decode: balanced accuracy, ROC, paired probabilities | external SRNN run (code not in this folder) |
| `classifier_notebook_panels/` (git-ignored) | Draft panels (A–G) from the classifier notebooks | `*_classifier_session_3sec` |

Detector validation figures (R1–R7, F1–F5) are in `review&delete/breath_detection/`.

**Foundation plot choices** (`behavior_resp_rate_plots.py`). It uses the §4 sets and one canonical
recording set per cohort (H5 with a matching, parseable BORIS), and the same set feeds both behavior
and rate. Bouts are 0.5–20 s, with no min-bout gate: low investigation in negative valence is the
finding. Both actor conventions are emitted. With both actors, negative valence reads higher because
the partner does most of the sniffing (focal median ~3 s vs ~62 s), so use **subject-only** for
valence. Compared with the original `cm_iteration1.ipynb` / `plot_social_behavior_RI1_RI2.ipynb` (repo
root), it uses the 400 Hz shared loader instead of FFT, a 20 s bout cap instead of 30 s, the QC
recording set instead of a hard-coded list, and rank recolored navy / mustard.

**Palettes**
```python
rank_palette = {"Dominant": "#1C304F", "Subordinate": "#D5B032"}   # navy / mustard
valence_palette = {"RI1": "#53A1A9", "RI2": "#A95376"}             # teal = Positive, pink = Negative
GRAY = "#6B6B6B"
# sniff type: facial #4C72B0, body #DD8452, anogenital #55A868
# breath annotation: inhale #4C78A8, exhale #F58518, pause #B279A2
```

---

## 9. Environments & running

- `biopipeline-env` (`biopipeline-env.yml`): has `neurokit2`, but numpy < 2, so it **cannot unpickle**
  the feature files. Use it for loading signals, feature extraction, and `behavior_resp_rate_plots.py`.
- `ephys_env`: numpy ≥ 2. Use it for the classifier notebooks, `decoder_metrics_table.py`,
  `decoder_roc_curves.ipynb` and `full_recording_feature_exploration.ipynb`.
- Scripts find `resp_lib/` relative to their own folder. Notebooks find it via `Path.cwd().parent / "resp_lib"`,
  so launch them from their own folder (the VS Code default).

---

## 10. Open issues / decisions

- **Actor convention** for the final behavior plots (both are emitted; subject-only recommended for valence).
- **`5_13` drop:** the rebuilt `BLRI_s5_13` (2026-11-10) has 10.3 min of respiration. Re-check, or keep the drop on RI1 / RI2 grounds.
- Which BORIS file is final for valence `RI1_s2_3` (two exist).
- The Aim1 valence rebuild found: `RI1_s3_5` has corrupt `.videoTimeStamps`; gaps in `BLRI_s2_3` (0.46 s) and `BLRI_s3_5` (0.12 s); 297 camera stalls in `BLRI_s5_13`; 15 / 24 fps captures in several sessions; `RI2_s3_5` respiration stops at 3.6 min.
- Pilot 1: rank conflict for session `…20260107_152812`; `6_2_ri2_…190944` sits in the RI1 day 2 folder; missing analog exports for 4 sessions; no H5 yet for `bl_cc`, `wrong_pulse`, and cohort 3 OFT.
- Aim1 1-day / 10-day isolation cagemate sessions have no BORIS.
- `Raw-Breathmatrics-Features-Ran-Interactions` and `…-Rank-balanced-Interactions` are byte-identical (checked 2026-11-24), so the "balanced" variant is not actually different.
- No analysis uses the adaptive detector or the v2 H5s yet.
