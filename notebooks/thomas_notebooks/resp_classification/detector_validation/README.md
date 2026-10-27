# Adaptive breath detection (`adaptive_prominence`) — method, validation, usage

Detector version 1.0.0 (`adaptive_breath_detection.py`), validated 2026-10-09.
Everything here is reproducible from the scripts in this folder; outputs are in `outputs_v1/`.
No existing feature file, pickle or H5 was modified.

---

## 1. Why the legacy detector misses breaths

The Python BreathMetrics port (`breathmetrics.py`) follows the original MATLAB
(`zelanolab/breathmetrics`: `breathmetrics.m`, `findRespiratoryExtrema.m`):

| Step | Genuine BreathMetrics design | Effect on rodent data |
|---|---|---|
| Extrema | A sample counts as a peak only if it is above `mean + std/2` of the **whole session**, and a trough only if it is below `mean − std/2` | Breaths that are small relative to the loudest breathing, or that ride on sub‑second drift, never qualify |
| Search | Rodent windows 5/10/20/50 ms (from `breathmetrics.m`), 3 offsets, consensus "knee" threshold | — |
| Alternation | Peaks and troughs must alternate. Two peaks in a row → one is deleted | A missed shallow trough deletes a **real** breath |
| Onsets | Inhale onset = crossing of the **session mean** between trough and peak | Breaths riding on drift may never cross it |

The Casali et al. (2026) mouse adaptation (`RouxLaboratory/Casali_2026`) keeps the same
extrema step and global threshold; it changes the onset crossing to a local level and uses
Savitzky‑Golay smoothing. That supports the same diagnosis: the global reference level is
the weak point.

### Translation differences between the Python port and MATLAB

| # | Location (`breathmetrics.py`) | Port behaviour | Status |
|---|---|---|---|
| 1 | extrema index `windowIter + maxInd - 1` | 1 sample early | **fixed** (2026-09-30) |
| 2 | first-peak loop and `pki`/`tri` started at 1 | dropped the first peak and trough | **fixed** |
| 3 | `nxtPk == 2` / `nxtTr == 2` (`argmax` returns 0 or 1) | always kept the first of two peaks | **fixed** |
| 4 | window slice `[w : w+sw-1]` | skipped the last sample of each window | **fixed** |
| 5 | `running_mean` zero-filled edges (MATLAB uses circular `fftSmooth`) | first/last 30 s not baseline-corrected | **fixed** |
| 6 | consensus `swPeakVect > threshold` (MATLAB `>=`) | needs one more vote than MATLAB | documented, **not changed** (legacy must reproduce existing files) |
| 7 | inhale onset = *first* sample above the mean (MATLAB: *last* sample at or below it) | differs when the trace wiggles around the mean | documented, not changed |
| 8 | last exhale offset: `x < lower and x >= upper` (impossible) | last exhale offset always NaN (1 breath per session) | documented, not changed |
| 9 | rodent window sizes floor to 0 samples below 200 Hz | crashed with unrelated errors | now a clear `ValueError` |

The legacy path (`detector="legacy"`, the default) reproduces the post‑fix (#1–5) outputs
exactly: `seq-sent/detector_audit/compare_fixed.py` gives an identical per-window
comparison before and after this work.

---

## 2. The adaptive method

Implemented in `../adaptive_breath_detection.py`. All parameters live in
`AdaptiveDetectorConfig`. Indices always refer to the input array, and `time_s` is
looked up, never shifted.

1. **Validity mask.** Uses 1 s blocks with a 0.25 s hop.
   - **Invalid:**
     - NaN samples;
     - recording-time gaps, where the step exceeds 1.5× the median step;
     - **dropouts**: block range below 3% of the session median **and** no periodicity (ACF peak in 60–500 ms below 0.5);
     - **artifacts**: block range above 8× the session median, padded by 0.25 s.
   - **Valid but flagged** `low_amplitude`: range below 15% of the session median, or below 3% but periodic.
   - Detection runs separately in each valid segment of at least 1 s. Nothing is linked across segments.
2. **Provisional cycles.** A copy of the signal is high-passed at 1 Hz and used *only* to find cycles and to measure local scale, so drift cannot inflate the threshold.
3. **Local baseline.** The cycle-mean baseline: the mean of the unfiltered signal over each peak-to-peak cycle, interpolated. For airflow, the flow integrated over one cycle is zero, so the cycle mean estimates zero flow locally. This is an assumption; see the limitations.
4. **Peaks.** `scipy.signal.find_peaks` on the baseline-corrected signal (10 ms running mean, odd window). A peak is kept if:
   - its prominence (bases searched within ±0.5 s, `wlen` = 1 s) is at least `max(0.20 × local 1 s p90−p10 range, 2 × noise floor)`;
   - its half-prominence width is at least 15 ms (spike rejection);
   - it is at least 60 ms from the next peak (the taller one wins).
   Every rejected candidate is recorded with a reason: `low_prominence`, `spike_width`, `too_close` or `shallow_trough`.
5. **Cycles.** The trough is the minimum between consecutive accepted peaks. Because both peaks passed the prominence test, the trough is at least that deep below the lower peak. This is enforced explicitly; violations drop the lower peak (`shallow_trough`). So "two peaks in a row" cannot delete a real breath, and a dip without evidence is never promoted to a cycle.
6. **Onsets, offsets and pauses**, per cycle against its own local baseline:
   - The inhale onset is the first sample after the *last* at-or-below-baseline sample before the peak (MATLAB semantics). The exhale onset is symmetric.
   - A pause is a run within ±(5% of *that cycle's* peak-to-trough amplitude, at least 1.5× noise) of baseline that contains the crossing and lasts at least 25 ms. A change in amplitude therefore cannot create a pause.
   - Offsets, durations, pause durations and volume proxies use BreathMetrics' definitions and scaling.
7. **Outputs.**
   - `bmObject` attributes have the same shapes as legacy, so `extract_bout_features` works unchanged.
   - In adaptive mode, window IBIs are computed only between consecutive breaths in the same segment.
   - The full per-breath table is `bm.breathTable`, and the rejected candidates are `bm.adaptiveResult.rejected`.

### Parameter choices
Parameters were chosen on **development data** (odd minutes of the 7 rank recordings) by
detector-intrinsic criteria, never by classifier accuracy; see `outputs_v1/selection_note.md`
and figure `R7`.
- **Prominence fraction:** 0.10–0.30 changed every metric by less than 5%; the prototype's 0.20 was kept.
- **Minimum spacing:** at 80 ms, intervals piled up at the limit (1.6%); at 40–60 ms there was no pile-up, so 60 ms was kept. Real intervals shorter than 80 ms exist, and no evidence required going below 60 ms.
- **Noise multiplier:** 3 → **2**, which raised worst-recording recall under 25% amplitude scaling from 0.957 to 0.987, with no cost elsewhere.
- **High-pass:** removing the 1 Hz scale high-pass dropped worst-case drift stability to 0.88.
- **Minimum width:** 15 ms. Real breaths are at least 30 ms (0.1st percentile of 26,000 peaks); a 14 Hz sinusoid is about 24 ms.
- **Disclosed post-hoc change:** the pre-specified spike criterion failed for *every* configuration, because it counted real breaths whose peak was displaced by a nearby spike. It was dropped from selection after seeing results.

---

## 3. Validation results

### Synthetic ground truth (`test_adaptive_breath_detection.py`: 25/25 pass; `outputs_v1/synthetic_benchmark.csv`)

| scenario | legacy recall / rate error | adaptive recall / rate error |
|---|---|---|
| clean 6 Hz | 0.998 / −0.01% | 1.000 / 0.00% |
| drift + DC offset | 0.072 / −92% | 1.000 / −0.05% |
| 10–40% amplitude stretch | 0.725 / −27% | 1.000 / 0.00% |
| genuine pauses (exhale-pause recall, false) | 0.77, 5 | 0.995, 0 |
| irregular 2–14 Hz | 0.928 / −7.1% | 1.000 / 0.00% |
| pauses + 20% amplitude + drift | 0.145 / −85% | 1.000 / +0.3% (pause recall 0.86) |

Adaptive peak, onset and trough timing is within 1 sample at 50–1000 Hz. The tests also
cover:
- spikes, which create no extra breaths;
- movement artifacts and dropouts, which are masked while low-amplitude breathing is kept and flagged;
- NaN and time gaps, which are never bridged, with recording time preserved;
- partial breaths at boundaries;
- the polarity flag;
- unsupported rates (below 40 Hz raises an error);
- consistency between the whole-session breath table and window summaries;
- the default-detector switch;
- regressions for each legacy fix.

### Real recordings (`outputs_v1/summary.md`; rank = held-out even minutes, valence = never used for tuning; 5_13 excluded per `docs/EXCLUSIONS_LOG.md`)

| | rank (7 rec, 34 min) | valence (23 rec, 224 min) |
|---|---|---|
| breaths legacy → adaptive | 12,788 → 13,686 (+7.0%) | 83,849 → 94,964 (+13.3%) |
| found only by legacy | 3 | 21 |
| intervals ~2× neighbours (miss signature) | 4.0% → 3.3% | 4.4% → 3.3% |
| intervals <0.6× neighbours (split signature) | 2.7% → 1.7% | 3.1% → 1.7% |
| NeuroKit2 peaks also found | 95.8% → 99.9% | 91.8% → 99.8% |
| detector peaks also found by NeuroKit2 | 99.9% → 97.4% | 99.9% → 95.9% |
| window rate within 15% of autocorrelation rate | 49% → 66% (both cohorts) | |

- The added breaths are concentrated in low local amplitude (`R4`): below 0.5× the recording median, 884 → 3,866 across both cohorts.
- About 3–4% of adaptive peaks are not confirmed by NeuroKit2. These are candidate false positives, *or* NeuroKit2 misses; only labels can say which.
- **The legacy undercount differs by condition and animal.** Rate changes:
  - valence: BLRI +11.7%, RI1 +13.8%, RI2 +15.6% (RI2_3_6 +39%, RI2_2_4 +35%);
  - rank: Dominant +4.1 to +12.0%, Subordinate +3.7 to +9.1%.
  Switching detectors can therefore change between-condition contrasts, not just shift all values.

### Downstream features (same windows; `summary_feature_changes.csv`, figure `R6`)
- **Timing** (n_breaths, rate, IBI): Spearman about 0.90 between detectors, median change about 5%. CV of IBI is lower in adaptive (fewer doubled intervals).
- **Exhale duration, peak flows, ventilation proxy:** Spearman 0.89–0.96.
- **Inhale duration, I:E ratio, duty cycles, inhale-volume proxy:** Spearman 0.43–0.73. These depend on the zero-flow estimate.
- **Pauses:** Spearman 0.34–0.62. Adaptive finds almost no inhale pauses (about 0.5% of breaths, against legacy's larger fraction), and 40–60% fewer exhale pauses.
  - Adaptive pause detection is correct on synthetic signals, but which detector is right on real pressure signals is **unknown**.
- **Feature status** (in `FEATURE_STATUS`, printable with `feature_status_table()`):
  - `supported`: n_breaths, breathing_rate_hz, mean_ibi_sec, cv_ibi;
  - `provisional`: durations, flows, pauses;
  - `proxy_provisional`: volumes (integrated uncalibrated pressure, not air volume).

### Polarity
The H5 metadata records only `analog_ECU_Ain1`, in volts; the sensor's physical polarity is
not recorded. Weak evidence supports positive = inspiration:
- the negative lobe is longer than the positive one in 100% of recordings (58 vs 50 ms);
- pauses follow the negative lobe in 4.8% of breaths, against 0.5% after the positive lobe.
Both match mouse physiology (longer expiration, end-expiratory pauses). If inspiration is
actually negative, set `polarity=-1`.

---

## 4. Limitations / what is NOT established

- **No human-labeled ground truth exists.** All real-data numbers are agreement with other methods and internal consistency, not accuracy. Use the labeling workflow (section 6) before calling the detector validated.
- **Whole-session failures pass the within-session rules.** Every rule is relative to the session's own amplitude, so a recording that is broken from start to finish looks normal to them. Valence 5_13 (median cycle amplitude 4–5 ADC, against ≥1,292 elsewhere) produced thousands of "breaths". Check `session_quality()` across sessions before analysis.
- **Spikes on a breath crest move that breath's peak.** A spike as tall as a breath, within about 40 ms of the crest, does this. Net extra breaths were 4–11 per 100 injected full-height spikes (development data). Artifacts wider than 15 ms are not rejected by width.
- **Low-amplitude rhythmic oscillations are counted as breaths.** These are 5–25% of normal amplitude at 7–12 Hz (figure `R2`). They look respiratory, but partial sensor decoupling would look similar. They are flagged `low_amplitude`, and `window_qc()` reports them per window.
- **The cycle-mean baseline assumes zero net flow per cycle.** If the sensor measures nasal pressure (roughly flow²) or the bandpass distorts slow components, inhale/exhale boundaries, pauses and volumes are biased. Hence `provisional`.
- **Parameter selection used the same 7 animals' other minutes**, not independent animals. Valence is the only fully independent check.

---

## 5. Using it in the notebooks

The detector is opt-in; nothing changes unless you ask for it.

**Option A — one line per notebook** (every `fit_bm_session` call in that kernel). Put it after the `resp_helper_functions` import:
```python
import resp_helper_functions as rh
rh.DEFAULT_DETECTOR = "adaptive_prominence"
```

**Option B — explicit per call:**
```python
from adaptive_breath_detection import AdaptiveDetectorConfig
bm = fit_bm_session(signal, fs, data_type="rodentAirflow",
                    detector="adaptive_prominence",
                    detector_config=AdaptiveDetectorConfig(),   # defaults = selected config
                    time_s=time)                                # optional; splits at time gaps
```

Where the calls are:
- **Rank:** `rank-featureextraction-3sec.ipynb` cell 20 (`build_cagemate_rank_feature_matrix_from_rank_map`); `feature_extraction-interactions-rank.ipynb` cells 53 and 78.
- **Valence:** `feature-extraction-session-3sec.ipynb` cells 4 and 12; `feature_extraction-interactions-valence.ipynb` cells 11, 12 and 16.

Both cohorts already use 400 Hz from the same Aim1 hardware. The 50 Hz notebook never fits BreathMetrics.

**Before running:**
1. **Change the output file names** (for example append `-adaptive-v1`) so the legacy pickles in `data/` are not overwritten.
2. **Restrict primary analyses to supported features:**
   ```python
   from adaptive_breath_detection import features_with_status
   primary = features_with_status(("supported",))   # timing features
   ```
   Report `provisional` features as exploratory only.
3. **Optional window QC.** QC columns are deliberately *not* added to the feature dict, because the classification notebooks treat every numeric column as a feature:
   ```python
   qc = pd.DataFrame([rh.window_qc(bm, s, e) for s, e in windows])   # join, then filter, then DROP
   ```
4. **Check sessions** with `session_quality(bm.adaptiveResult)` and compare `median_cycle_amplitude` across sessions.
5. **Reproduce the old numbers** with `detector="legacy"` (the default).

---

## 6. Reproducing and extending the validation (`biopipeline-env`)

```
python test_adaptive_breath_detection.py          # 25 tests (no pytest needed)
python synthetic_benchmark.py outputs_v1
python sweep_dev.py outputs_v1                     # dev-minute parameter sweep
python compare_real.py outputs_v1 default          # held-out comparison (+ per-recording npz)
python figures.py outputs_v1                       # R1-R7
python summarize.py outputs_v1                     # summary.md
python make_labeling_set.py outputs_v1 60          # blind labeling set
python score_labels.py outputs_v1                  # after labeling
```
Signals are cached in the session scratchpad. Set `BREATH_VALIDATION_CACHE` to move the
cache.

**Blind labeling.**
1. Open `outputs_v1/labeling/label_breaths.html`. It has 204 shuffled items:
   - 60 adaptive-only peaks;
   - all 24 legacy-only peaks;
   - 60 agreed peaks (controls);
   - 60 adaptive-rejected near-misses.
   The detector identity is hidden.
2. Label each item breath / not_breath / unsure. Progress is kept in the browser.
3. Click "Download CSV" and save it as `outputs_v1/labeling/labels.csv`.
4. Run `score_labels.py`. It gives per-class breath rates with 95% CIs, and precision and union-recall for each detector, stratified by class counts.
5. Do **not** open `answer_key.csv` before labeling. Ideally two labelers work independently.

---

## Files
- `../adaptive_breath_detection.py` — detector, config, feature status, `session_quality`, `polarity_diagnostic`
- `../breathmetrics.py` — `estimateAllFeatures(detector=...)`; legacy path unchanged; clear error for unsupported rates
- `../resp_helper_functions.py` — `fit_bm_session(detector=..., detector_config=..., time_s=...)`, `DEFAULT_DETECTOR`, segment-aware window IBIs (adaptive only), `window_qc`
- `synthetic.py`, `test_adaptive_breath_detection.py`, `synthetic_benchmark.py`, `real_common.py`, `sweep_dev.py`, `compare_real.py`, `figures.py`, `summarize.py`, `make_labeling_set.py`, `score_labels.py`
- `outputs_v1/` — CSVs, `summary.md`, `selection_note.md`, `figures/R1–R7`, `labeling/`, `recordings/*.npz`
