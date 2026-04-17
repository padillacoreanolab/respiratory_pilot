# Data provenance, anti-leakage design, and next-notebook plan

## What data this notebook uses

This notebook builds respiration-based decoding datasets from multiple `.h5` sources and merges them into one processing stream.

### 1) AIM1 valence sessions
- `RI1_*` and `RI2_*` recordings (valence conditions)
- `BLRI_*` baseline recordings
- Loaded via the `resp_paths` dictionary

### 2) AIM1 cagemate sessions
- `CM_*` interaction recordings
- `BL_*` cagemate baseline recordings
- Loaded via `resp_paths_cm` and `resp_paths_bl`

### 3) Additional BLA cohort cagemate recordings
- Pulled from:
  - `D:\BLA_ChR_resp\BLA_resp_ChR_bl_cc\h5_outputs`
  - `D:\BLA_ChR_resp\BLA_resp_ChR_cm_hc\h5_outputs`
- Filenames are renamed with `rename_map` so subject IDs are unique when combining cohorts

### 4) Labeling and metadata
- `rank_map` maps subject IDs (example: `1_1`, `1_2`) to `Subordinate` / `Dominant`
- Trial metadata is parsed from trial names:
  - `Condition` from prefix (`RI1`, `RI2`, `CM`, `BL`, `BLRI`)
  - `Subject` from ID tokens in trial key
  - `Rank` from `rank_map` (or file suffixes for some renamed BLA files)

### 5) Final sample table built in this notebook
- Main table: `resp_sample_df`
- Columns used in modeling sections:
  - `Trial`
  - `Subject`
  - `Condition`
  - `Rank`
  - `RespRate`
- Feature currently used for classifiers: **only** `RespRate`

---

## How this notebook organizes data to reduce leakage

### Current good practice already present
For several analyses, this notebook uses:
- `groups = df_val["Subject"]`
- `GroupKFold(...)`

That is the right idea because each mouse appears in many windows/trials. Grouping by `Subject` prevents the same mouse from being in both train and test folds in those runs.

### Why leakage is a real risk here
The pipeline creates many random windows (`n_samples=100`) per recording. Those windows from one mouse are highly correlated. If you split at window level (instead of subject/group level), the model can memorize mouse-specific patterns (including dominant/subordinate identity proxies) instead of learning generalizable respiration biology.

### Leakage risk still present in this notebook
There are also sections using `StratifiedKFold` on window rows (no grouping), which can mix windows from the same subject into train and test. Those results are likely optimistic for repeated-mouse data.

---

## Critical grouping principle for repeated dominant/subordinate mice

Because mice are repeated and rank is stable within each mouse, **evaluation must split by mouse identity** (or stricter).

Use this hierarchy:

1. **Best minimum**: split by `Subject` (GroupKFold / GroupShuffleSplit)
2. **Stricter for dyad confounds**: split by cage pair/dyad (e.g., `1_1-1_2`) so neither mouse from a pair appears in both train and test
3. **Strictest for cohort/site confounds**: nested/grouped splits that also isolate cohort (`AIM1` vs BLA) in outer evaluation

If the goal is rank decoding generalization to unseen mice, never report window-level stratified CV alone.

---

## Recommended design for the next notebook (more features + preprocessing)

## A) Data table structure to build first
Create one master dataframe where every row is one fixed window from one trial:

Required ID columns:
- `cohort` (AIM1 or BLA)
- `trial_id`
- `subject_id`
- `partner_id` (if present)
- `dyad_id` (canonical sorted pair key)
- `condition` (`RI1`, `RI2`, `CM`, `BL`, `BLRI`, etc.)
- `rank_label` (`Dominant`/`Subordinate`)
- `session_date` or recording block
- `window_id`

Feature columns:
- `resp_rate` and future extracted features (time-domain, frequency, variability, waveform-shape)

This ID-first table is the anti-leakage backbone.

## B) Preprocessing rules (fit only on train folds)
For each CV fold:
1. Split by groups first (subject/dyad)
2. Fit preprocessing on train only:
   - scaling
   - imputation
   - optional PCA / feature selection
3. Apply fitted transformers to test

Put all preprocessing inside an sklearn `Pipeline` so leakage through preprocessing is impossible.

## C) Windowing strategy improvements
- Keep windows non-overlapping within a trial (already partly done)
- Track `window_start`, `window_end`
- Use consistent window duration per experiment (or include as metadata)
- For robustness: evaluate multiple random seeds for window sampling and aggregate metrics

## D) Evaluation protocol to report
Use two complementary evaluations:

1. **Primary (required):** Grouped CV by `subject_id` (or `dyad_id` for stricter test)
2. **Secondary:** Cohort transfer
   - Train on AIM1, test on BLA
   - Train on BLA, test on AIM1

Report:
- ROC AUC, PR AUC, balanced accuracy, calibration (if probabilistic outputs)
- mean ± SEM or 95% CI over folds
- per-group performance (dominant vs subordinate, each cohort)

## E) Aggregation checks to avoid pseudo-replication
Also create subject-level summaries:
- average features per subject per condition/session
- train/test on these summaries as a sensitivity analysis

If window-level and subject-level results disagree strongly, treat window-level gains cautiously.

## F) Confound checks for rank task
Because rank and mouse identity are tightly linked:
- verify no subject overlap between train/test in every split
- test whether adding explicit identity proxies changes performance drastically
- check dyad leakage explicitly (`dyad_id` overlap)
- run label permutation within training folds as a sanity check

---

## Practical split templates for this project

### Rank decoding (CM only)
- `X = feature_matrix`
- `y = rank_label`
- `groups = subject_id` (minimum) or `dyad_id` (stricter)
- CV: `GroupKFold` or repeated `GroupShuffleSplit`

### Valence decoding (RI1 vs RI2)
- `X = feature_matrix`
- `y = condition_binary`
- `groups = subject_id`
- Optional: stratify at group level by dominant/subordinate balance if feasible

### Combined-cohort modeling
- include `cohort` column
- evaluate both pooled-group CV and leave-one-cohort-out transfer

---

## What else can be done next (high-value additions)

1. Add richer respiration features beyond rate:
   - inter-breath interval variability
   - spectral band power and peak frequency
   - inhale/exhale duration ratio
   - amplitude and shape descriptors

2. Add quality-control flags per window:
   - signal dropout/artifact ratio
   - confidence score for breath detection
   - exclude low-quality windows before modeling

3. Use nested grouped CV for tuning:
   - outer grouped split for honest test
   - inner grouped split for hyperparameter tuning

4. Compare model families under same grouped protocol:
   - logistic regression baseline
   - regularized linear models
   - tree-based models (with careful overfitting control)

5. Perform ablations:
   - RespRate-only vs full feature set
   - AIM1-only vs BLA-only vs pooled
   - subject-grouped vs dyad-grouped split sensitivity

---

## Bottom line
For this repeated-mouse dominant/subordinate design, leakage control is mainly a **split-definition problem**. The next notebook should treat `subject_id`/`dyad_id` as first-class grouping keys, keep preprocessing strictly fold-local, and report grouped generalization as the primary result. This will make future feature-extraction improvements scientifically interpretable and trustworthy.