# Poster foundation plots — inputs, drops, signal processing

> Reference for [`poster-code/poster_behavior_resp_plots.py`](../poster-code/poster_behavior_resp_plots.py):
> exactly what goes in, what's dropped and why, how the respiration signal is processed, and how
> this differs from the old `cm_iteration1.ipynb` originals. Drop rules trace to the
> feature-extraction notebooks (see [PREPROCESSING_NOTES.md](../rank-classification/preprocessing_docs/PREPROCESSING_NOTES.md)).

---

## 1. Inputs (Dropbox; `E:` mirror auto-used if mounted)

Root: `C:\Users\thoma\UF Dropbox\…\Padilla-Coreano Lab\2025\ECG_cohort1\Aim1\AIM1\Day1_new\`

| Cohort | Respiration H5 | BORIS |
|---|---|---|
| **Valence** (RI1=Positive, RI2=Negative) | `Resp_h5\` | `boris_csv_RI1 and RI2\` |
| **Rank / cagemate** (baseline Aim1) | `CM interactions\cm_h5\baseline_cagemate_interactions_h5_outputs\` | `CM interactions\cm_boris\baseline csv\` |

**BLA cagemate cohort is NOT used** (no baseline recording → cohort artifacts).

---

## 2. Signal processing (identical to the feature-extraction notebooks)

Loaders (cohort-matched, to mirror each FE notebook exactly):
- **Valence:** `load_clean_resp_signal(h5, target_rate=400)`
- **Rank:** `load_rank_resp_signal_interactions(h5, target_rate=400, prefer_raw=True)`

Both route through the shared helper **`preprocess_resp_to_target_rate`**:
1. Butterworth **low-pass**, order 4, cutoff = `(target/2)/nyquist`.
2. **Downsample** to 400 Hz via `resample_poly(down=int(fs//target))` (polyphase).
3. NeuroKit **band-pass** Butterworth **0.1–20 Hz**, order 2.

Respiration rate = **inhalation peak count / window duration**, `find_peaks(distance = fs·0.0833)`
(≥12 Hz refractory). Whole-session uses `[t0, t_end)`; the interactions plot averages the rate
over the focal sniff-bout windows.

> **Dead signal is NOT trimmed within a recording.** The whole-session window is the entire
> `[t0, t_end)` — there is no detector for flat/saturated/disconnected stretches, so a dead segment
> just yields few/no peaks and **lowers** that recording's rate. Dead signal is handled only at the
> whole-**subject** level, manually: valence `5_13` is dropped (RI2 trace flat/broken, std ≈ 20 vs
> ~700–1000). A kept recording with a *partial* dead stretch is used as-is — flag it upstream if that
> matters.

> **Why no FFT?** The FFT `scipy.signal.resample` (+ a 0.1–**15** Hz band-pass, `0.45*fs` cutoff)
> is a one-off local `load_clean_resp_signal`. It was never the project pipeline; we standardized
> on the shared helper above (`resample_poly`, 0.1–20 Hz), which is what the feature-extraction
> notebooks and the majority of the codebase use.

### FFT-method audit (all 157 notebooks outside `resp_classification`)

Classified by the actual resampling call, not the function name (`load_clean_resp_signal` is
redefined locally in ~60 notebooks):

- **Exact FFT resp loader (FFT resample + 0.1–15 Hz + `0.45*fs`)** — only **2**:
  `cm_iteration1.ipynb` and `r34_analysis5.ipynb`.
- **Contains an FFT `resample()` too** (usually alongside the newer `resample_poly`, i.e.
  transitional/compare notebooks): `r34_analysis3` (+ `AIM1/r34_analysis3`), `r34_analysis4`,
  `r34logistic.ipynb`, `Other_people/luke_play_ipynb`, `resp_pipline/Comparing_3versions…`,
  `resp_pipline/resp_pipline_iteration1`+`2`. Plus old **ECG/heart-rate** notebooks in
  `old_shit/` (`*ecg*`, `*hear_rate*`, `subj11_*`) that resample a *different* signal.
- **Canonical method (`resample_poly` + `highcut=20`, = the shared helper)** — the majority:
  ~50 recent notebooks (all `IBI_percent_*`, `cohort*_hzchange*`, `BLA_*`, `r34logistic_*`,
  `r34_analysis1`, `plot_social_behavior_RI1_RI2`, …).

**Bottom line:** the FFT method is *not* used broadly — it's essentially isolated to
`cm_iteration1.ipynb` + `r34_analysis5.ipynb` (with a few transitional notebooks carrying both).
The poster script matches the dominant/maintained pipeline.

---

## 3. What's dropped, and why

**Cohort-level (whole subject/recording):**

| Dropped | Cohort | Why |
|---|---|---|
| `1_1`, `1_2` | Rank | **Fighting** sessions (per rank FE notebook). |
| `3_6` | Rank | BORIS export is **empty** → no behavior to align. |
| `CM_s3_5_d3_6_…170708` (h5) | Rank | Duplicate 3_5 session with **no BORIS**; the `…160001` session (h5+BORIS) is kept. |
| `CM_s4_8_sub4_7_20250623.csv` | Rank | **Bad duplicate** BORIS — tab-separated, missing session time; the timestamped `…182649_VT` export is kept. |
| `5_13` | Valence | Respiration **outlier** (RI1 whole-session ≈ 4.9 Hz vs ~7.5) and the **RI2 signal is broken/flat** (std ≈ 20 vs ~700–1000). Also not one of the 8 rank-matched mice. |
| `RI1_3_5` | Valence | **No BORIS** for RI1 3_5 → dropped by the matching-behavior rule (per valence FE: "drop RI1_3_5"). |

**Bout-level cleaning (behavior + interaction-rate):**
- Actor convention is **per cohort**, matching each original analysis:
  - **Rank / cagemate** = **both actors** (`SUBJECT_ONLY_RANK=False`, as `cm_iteration1.ipynb`).
    The focal's respiration responds both when it sniffs and when it is being sniffed.
  - **Valence** = **subject-initiated only** (`SUBJECT_ONLY_VALENCE=True`, as
    `plot_social_behavior_RI1_RI2.ipynb`). Counting the partner would inflate *negative* valence
    (there the agent does almost all the sniffing: focal median ~3 s vs both-actor ~62 s) and
    erase the real effect (subjects investigate positive agents far more than negative).
- Keep bouts with **0.5 s ≤ Duration ≤ 20 s** → removes the single ~400 s anogenital scoring
  error and sub-0.5 s noise (the 20 s cap is the rank FE convention).
- **No minimum-bout gate.** The `min_sniffs=10` rule in the FE notebooks is a *classifier*
  requirement (enough windows to fit features). Here it would wrongly delete real
  low-investigation **negative-valence** recordings (RI2 focals barely investigate — that low
  investigation *is* the finding), so it is not applied to these descriptive plots.

**Consistency rule:** one **canonical recording set per cohort** — an H5 is included only if it
has a matching, parseable BORIS file (and survives the drops above) — and the **same set feeds
both the behavior and the respiration-rate plots**, so no recording appears in one plot but not
the other. Prefer `*_VT` BORIS exports when a session has more than one.

---

## 4. Final recordings actually plotted (verified 2026-07-13 run)

| Plot | Group counts | Subjects |
|---|---|---|
| `behavior_rank_*`, `resp_rate_rank`, `resp_rate_rank_interactions` | **Sub 4 / Dom 3** | Sub: 1_1, 2_3, 3_5, 4_7 · Dom: 1_2, 2_4, 4_8 |
| `behavior_valence_*`, `resp_rate_valence` | **Pos 7 / Neg 8** | Pos: 1_1,1_2,2_3,2_4,3_6,4_7,4_8 · Neg: + 3_5 |

> ⚠️ **Small rank n (4 Sub / 3 Dom = 7 mice / 4 cages).** `RANK_DROP_SUBJECTS = {"3_6"}` only —
> the fighting **cage 1 (1_1/1_2) is KEPT** (cohort too small to drop; user confirmed), so it is no
> longer the "Sub 3 / Dom 2" set an earlier version of this doc listed. Only `3_6` (empty BORIS) is
> dropped. To drop cage 1 too, add `"1_1", "1_2"` to `RANK_DROP_SUBJECTS`.

---

## 5. How this differs from the original `cm_iteration1.ipynb` plots

| Aspect | Original (cm_iteration1) | Here (consolidated) | Effect |
|---|---|---|---|
| Behavior actor | Rank: both actors; Valence: subject-initiated | **Per cohort** — rank both actors, valence subject-initiated | Matches each original's convention (the two originals differ). |
| Recording set | Globbed / hard-coded ~8–9 incl. fighting + no-BORIS + duplicate 3_5 | Canonical QC set (matching BORIS, drops applied) | Fewer, cleaner points; consistent across plots. |
| Signal cleaning | Local loader: FFT resample, band-pass 0.1–**15** Hz | Shared helper: `resample_poly`, 0.1–**20** Hz, 400 Hz | Small rate shifts (<~0.2 Hz); now identical to feature extraction. |
| Long/short bouts | Duration ≤ 30 s | 0.5 ≤ Duration ≤ 20 s | Removes the 400 s error + sub-0.5 s noise. |
| Valence resp metric | (no confirmed original) | Whole-session peak-count, RI1 vs RI2, `5_13` dropped | Clean Positive ≫ Negative; outlier removed. |

Full pre-fix catalogue: [PLOT_PROCESSING_DIFFERENCES.md](./PLOT_PROCESSING_DIFFERENCES.md).
