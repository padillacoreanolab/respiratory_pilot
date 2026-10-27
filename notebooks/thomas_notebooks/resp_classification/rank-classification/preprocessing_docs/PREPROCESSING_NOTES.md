# Rank Classification — Preprocessing Notes (for future-me)

> A self-contained reference so that if you come back in a week or two you don't have to
> re-read the notebooks to remember *what was included, what was dropped, and why*.
>
> Source of truth: [feature_extraction-interactions-rank.ipynb](../feature_extraction-interactions-rank.ipynb)
> and the shared helpers in [resp_helper_functions.py](../../resp_helper_functions.py).
> If those change, this doc can drift — treat it as a map, not gospel.

---

## 1. What this pipeline does (one paragraph)

Loads cagemate **social-interaction** recordings from two cohorts (Aim1 + BLA), pairs each
respiration `.h5` with its hand-scored **BORIS** behavior `.csv`, runs **BreathMetrics** on
the full respiration trace, slices out **sniffing bouts**, and produces a **window-level
feature matrix** (one row = one sniffing bout) labeled with the subject's social **Rank**
(Dominant / Subordinate). The end product is used for rank classification.

It is **not** raw time-series output — each row is a *summary* of the breaths inside a bout.
(See the companion RNN guide if you need sequences instead.)

---

## 2. The two cohorts

| Cohort | Label in code | Recording era | Resp signal in H5 | Cage size |
|--------|---------------|---------------|-------------------|-----------|
| Aim1   | `Aim1`        | June 2025     | top-level `resp` @ 20 kHz | pairs (2/cage) |
| BLA    | `BLA`         | Jan 2026      | raw `analog/...ECU_Ain1` @ ~20 kHz **and** cleaned `resp_clean` @ 100 Hz | triads (3/cage) |

Paths are hard-coded near the top of the notebook (UF Dropbox):
- Aim1 H5: `.../ECG_cohort1/Aim1/AIM1/Day1_new/CM interactions/cm_h5/baseline_cagemate_interactions_h5_outputs`
- Aim1 BORIS: `.../CM interactions/cm_boris/baseline csv`
- BLA H5: `.../BLA_ChR_resp_pilot1 (SS)/BLA_resp_ChR_cm_hc/h5_outputs`
- BLA BORIS: `.../BLA_ChR_resp_pilot1 (SS)/BLA_resp_ChR_cm_boris`

---

## 3. Filename conventions (the part that's easy to forget)

### Aim1 H5
```
CM_s{subA}_{subB}_{rank}{parA}_{parB}_{date}_{time}_merged.h5
```
- `rank` token is literally `d` (dominant) or `sub` (subordinate) — but for Aim1 the rank
  actually used comes from the **hard-coded map** (§4), not this letter.
- e.g. `CM_s1_1_sub1_2_20250623_111352_merged.h5`

### Aim1 BORIS
```
CM_s{...}_{date}_{time}.{n}.csv      or      ..._{date}_{time}.{n}_VT.csv
```
- When a session has **multiple** exports, the loader prefers the one with `_VT`.

### BLA H5
```
{subA}_{subB}_{subRole}_cm_{parA}.{parB}_{parRole}_{date}_{time}(.rec).h5
```
- `subRole` / `parRole` ∈ `{d, i, s}` → **d**ominant / **i**ndeterminate / **s**ubordinate.
- `cm` = cagemate. Note the partner numbers can be `.`-separated (`4.1`) not `_`.
- BLA rank is **derived from `subRole`** (d→Dominant, s→Subordinate, **i→no rank → dropped**).

### BLA BORIS
```
{subA}_{subB}_{subRole}_cm_{parA}_{parB}_{parRole}_{date}_{time}(.n)_{tag}.csv
```
- `tag` varies: `VT_SA`, `VT_CM`, `BORIS_SS`, `BORIS`, `boris_AJ_with_image_indices`, …
- `VT_SA` = subject scored; `VT_CM` = social-agent scored (see §6 outlier).

### Internal trial keys (what you see in `master_df["Trial"]`)
```
AIM1cm_{sub}_{par}_{date}_{time}
BLAcm_{sub}_{par}_{date}_{time}_{tag}
```
- Subject id = chars between cohort prefix and partner, e.g. `1_1`. Parsing in code:
  `subj = "_".join(trial.split("_")[1:3])`, `partner = parts[3:5]`.
- Date = `YYYYMMDD`. Aim1 = `20250623`; BLA = `20260106`–`20260108`.

> **Gotcha:** in the BORIS files the column called `Subject` is the **actor** label
> (`subject` vs `social_agent`), *not* the mouse ID. The loader renames it to `Initiator`
> to avoid confusion with the mouse `Subject` id added later.

---

## 4. Rank mapping

### Aim1 — hard-coded (`aim1_rank_map`)
| Subject | Rank        |
|---------|-------------|
| 1_1     | Subordinate |
| 1_2     | Dominant    |
| 2_3     | Subordinate |
| 2_4     | Dominant    |
| 3_5     | Subordinate |
| 3_6     | Dominant    |
| 4_7     | Subordinate |
| 4_8     | Dominant    |

### BLA — derived from the H5 filename role letter
`d` → Dominant, `s` → Subordinate, `i` → (no rank; those rows are dropped because
`master_df = master_df[master_df["Rank"].notna()]`).

> ⚠️ **Collision risk worth remembering:** the rank map is keyed by subject id **only,
> not by cohort**. Aim1 has `1_1`/`1_2`/`2_3` and BLA also has `1_1`/`1_2`/`2_3` — these
> are *different mice*. The merge loop (`for subj_id, rnk in bla_aim1_rank_map.items()`)
> overwrites the Aim1 entry with the BLA value on collision and prints a conflict warning.
> Everything downstream that *filters* subjects is cohort-aware (`exclude_subjects_by_cohort`),
> but the rank lookup itself is not. If you ever pool both cohorts, double-check rank labels.

---

## 5. Who was caged together

Animals only interact with **cagemates** (`cm` in every filename). The **first index** of a
subject id is the cage/cohort.

### Aim1 — 4 cages, 2 mice each, animal numbers are global (1–8)
| Cage | Mice        | Notes |
|------|-------------|-------|
| 1    | 1_1, 1_2    | **excluded** — fighting sessions |
| 2    | 2_3, 2_4    | |
| 3    | 3_5, 3_6    | 3_6 has no BORIS |
| 4    | 4_7, 4_8    | |

Each pair was recorded **both directions** (each animal as the focal subject).

### BLA — up to 8 cages, ~3 mice each, animal numbers restart per cage (`X_1, X_2, X_3`)
| Cage | Mice observed         |
|------|-----------------------|
| 1    | 1_1, 1_2, 1_3         |
| 2    | 2_1, 2_2, 2_3         |
| 3    | 3_1, 3_2, 3_3         |
| 4    | 4_1, 4_2, 4_3         |
| 5    | 5_1, 5_2, 5_3         |
| 6    | 6_2, 6_3 (no 6_1 seen)|
| 7    | 7_1, 7_2              |
| 8    | 8_1, 8_2              |

BLA cagemates interact **round-robin** (e.g. 1_1↔1_2, 1_1↔1_3, 1_2↔1_3). So a recording's
subject and partner are *always from the same cage*. Cross-cage interactions do **not** exist
in this dataset.

> The numbering schemes differ between cohorts (Aim1 global 1–8 in pairs; BLA per-cage 1–3).
> That difference, plus §4's collision note, is the main thing to keep straight.

---

## 6. Known data issues & explicit drops (the "why is X missing" list)

**Hard exclusions baked into the catalog build:**
- `CM_s3_6_sub3_5_20250623_174348` (Aim1) — empty/missing BORIS → skipped (`aim1_skip_boris_stems`).
- `CM_s4_8_sub4_7_20250623.csv` (Aim1) — bad duplicate, missing session time → skipped (`aim1_exclude_boris_names`).
- BLA sessions split into **`VT_SA` (subject)** + **`VT_CM` (social agent)** exports of the
  same H5. Both are kept through the catalog; downstream `dedupe_h5_boris_exports(prefer_tag="VT_SA")`
  keeps the **VT_SA** trial per H5.

**Behavior-level filters (in order):**
1. Keep only sniffing: `facial sniffing`, `body sniffing`, `anogenital sniffing`.
2. Drop bouts with `Duration < 0.5 s`.
3. Drop the one ~400 s `anogenital sniffing` bout in `AIM1cm_2_3_2_4_20250623_151153`.
4. Drop trials with `< 10` subject-initiated sniff bouts (`min_sniffs_per_trial=10`).
5. Drop subjects whose bout counts are too **imbalanced** across their recordings
   (`imbalance_ratio` 0.4 upstream, 0.25 in the balance audit), cohort-aware.

**Feature-matrix-level drops:**
- Bouts with `< 2` breaths in the feature window → dropped (and diagnostically plotted).
- `master_df = master_df[master_df["Trial"] != "RI1_3_5"]` (explicit one-off).
- `Rank.isna()` rows dropped (covers BLA `i`-role recordings and unparseable names).
- Balance audit additionally drops Aim1 `1_1` & `1_2` (`drop_aim1_subjects=["1_1","1_2"]`)
  and any subject with `< 10` windows (`min_windows=10`).

> Two `master_df` variants exist: the lightly-filtered `master_df` and the
> `master_df_balanced` (deduped + balanced). Know which one you're loading.

---

## 7. Signal preprocessing (`load_rank_resp_signal_interactions`)

1. **Prefer raw** high-rate trace:
   - BLA → `analog/{stem}.analog_ECU_Ain1`, read the `voltage` field, fs from `clockrate` attr (~20 kHz).
   - Aim1 → top-level `resp` dataset, fs from `resp_metadata` (default 20 kHz).
2. **`preprocess_resp_to_target_rate`** (target = **400 Hz** here):
   - 4th-order **Butterworth low-pass** at `(target/2)/nyquist`.
   - **Downsample** by `int(fs // target)` via `resample_poly`.
   - **NeuroKit band-pass** 0.1–20 Hz, Butterworth order 2.
   - Returns cleaned signal + time vector at 400 Hz.
3. **Fallback** to `resp_clean/signal` (+ `resp_clean/time`, ~100 Hz) only when raw is
   missing or `prefer_raw=False`.

> The stored `resp_clean` is 100 Hz; this pipeline deliberately re-derives **400 Hz** from
> raw so Aim1 and BLA are processed identically.

---

## 8. Feature extraction (BreathMetrics)

- BreathMetrics is fit on the **full session** (`data_type="rodentAirflow"`, srate 400),
  *then* breaths are selected per bout window. This is intentional — full-session context
  gives more reliable onset/offset/volume estimates than fitting tiny clips.
- Bout windows: `target_dur=2.0 s`, `anchor="onset"` (window starts at behavior onset),
  `min_dur=0.5`, `pre_bout_window=2.0 s`, `min_breaths_per_bout=2`.
- A breath belongs to a window if its **inhale onset** falls inside it (non-strict).

**Feature columns produced per bout** (from `extract_bout_features`):
`n_breaths`, `breathing_rate_hz`, `mean_ibi_sec`, `cv_ibi`, `mean_inhale_dur_sec`,
`cv_inhale_dur`, `mean_exhale_dur_sec`, `cv_exhale_dur`, `ie_ratio`, `mean_peak_insp_flow`,
`mean_peak_exp_flow`, `cv_peak_insp_flow`, `mean_inhale_vol`, `mean_exhale_vol`,
`mean_tidal_vol_proxy`, `cv_inhale_vol`, `ventilation_proxy`, `pct_breaths_with_inhale_pause`,
`mean_inhale_pause_dur_sec`, `cv_inhale_pause_dur`, `inhale_pause_duty_cycle`,
`pct_breaths_with_exhale_pause`, `mean_exhale_pause_dur_sec`, `cv_exhale_pause_dur`,
`exhale_pause_duty_cycle`, `inhale_duty_cycle`, `exhale_duty_cycle`.

**Engineered on top** (`add_engineered_features`): `vol_asymmetry`, `peak_flow_ratio`,
plus per-subject (`*_z_subject`) and per-trial (`*_z_trial`) z-scores of every numeric
feature; also `breathing_rate_z`, `PreBoutRespRate`, `resp_rate_delta`.

**Metadata columns:** `Behavior, Initiator, Start, Stop, Duration, PreBoutRespRate, Trial,
Subject, social_agent, Cohort, Rank, Type, h5_path, h5_filename, boris_path, boris_filename,
FeatureWindow`.

---

## 9. Outputs written

`DATA_DIR = notebooks/thomas_notebooks/resp_classification/data`

| File | Contents |
|------|----------|
| `Raw-Breathmatrics-Features-Ran-Interactions` | `master_df` (note the **`Ran`** typo — not `Rank`) |
| `Raw-Breathmatrics-Features-Rank-balanced-Interactions` | `master_df_balanced` |
| `resp_paths_interactions_rank.pkl` | dict of trial_key → h5 path |

> ⚠️ Filename typo `Ran` is real in the current notebook — match it exactly when loading,
> or fix it at the save site.

---

## 10. Quick gotcha checklist

- [ ] BORIS `Subject` column = actor, renamed to `Initiator`; mouse id is added separately.
- [ ] Rank map is **not** cohort-aware → Aim1/BLA id collisions (1_1, 1_2, 2_3).
- [ ] BLA `i`-role recordings have no rank and get dropped.
- [ ] `master_df` vs `master_df_balanced` — pick the right one.
- [ ] Output pickle name has the `Ran` typo.
- [ ] Aim1 1_1 & 1_2 are excluded (fighting); 3_6 has no BORIS.
- [ ] Signal is re-derived to **400 Hz** from raw, not the stored 100 Hz `resp_clean`.
- [ ] Each row is a **bout summary**, not a time series.
