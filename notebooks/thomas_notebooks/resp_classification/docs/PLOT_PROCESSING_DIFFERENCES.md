# Consolidated poster plots vs. the originals — processing differences

> How the current [`poster-code/poster_behavior_resp_plots.py`](../poster-code/poster_behavior_resp_plots.py)
> differs from the repo-root original notebooks it consolidates. This is the "why don't the new
> plots match the old ones exactly" reference. Every item here is a **deliberate, already-made
> decision** baked into the script — not an open TODO.
>
> Originals (repo root, READ-ONLY): [`cm_iteration1.ipynb`](../../../cm_iteration1.ipynb) (rank /
> cagemate behavior + resp rate), [`plot_social_behavior_RI1_RI2.ipynb`](../../../plot_social_behavior_RI1_RI2.ipynb)
> (valence behavior). Inputs/drops rationale: [DROPS_AND_INPUTS.md](./DROPS_AND_INPUTS.md).
>
> ⚠️ **This doc was rewritten 2026-07-13** to match the current script. The earlier version
> described a superseded 100 Hz / focal-only build with function names (`build_rank_behavior_df`,
> `build_valence_resp_df`, …) that no longer exist — ignore any stale copy you may have cached.

---

## What the current script actually does (baseline for the diffs below)

- **Signal:** shared helper `preprocess_resp_to_target_rate` at **400 Hz** (Butterworth low-pass
  order 4 → `resample_poly` → NeuroKit band-pass **0.1–20 Hz**, order 2). Cohort-matched loaders:
  `load_clean_resp_signal` (valence), `load_rank_resp_signal_interactions(..., prefer_raw=True)` (rank).
- **Resp rate:** local `respiratory_rate()` = `find_peaks(distance=fs*0.0833)` (≥12 Hz refractory),
  **peak count / window duration**. Whole-session window = the full `[t[0], t[-1])`.
- **Behavior:** both actor conventions plotted **per cohort** (`subject_only` + `both_actors`),
  bouts cleaned to `0.5 ≤ Duration ≤ 20 s`, no min-bout gate.
- **Recording set:** one canonical QC set per cohort (an H5 is in only if it has a matching,
  parseable BORIS and survives the documented drops); the SAME set feeds behavior + resp-rate.

---

## 1. Behavior — actor convention (biggest visible difference)

The two originals used **different** filters, and the script now surfaces this by plotting **both**:

| | Original | Script emits |
|---|---|---|
| Rank (`cm_iteration1`) | **both actors**, all assigned the focal's rank | `behavior_rank_subject_only` **and** `behavior_rank_both_actors` |
| Valence (`plot_social_behavior_RI1_RI2`) | **subject-initiated only** | `behavior_valence_subject_only` **and** `behavior_valence_both_actors` |

With **both actors**, negative valence (RI2) reads *higher* than positive — not a flipped label:
in RI2 the partner does almost all the sniffing (focal median ~3 s vs both-actor ~62 s). Subject-only
restores Positive > Negative. The Aim1 BORIS actor labels were audited (2026-07-13) and are correct; the
remaining actor caveats are listed in [EXCLUSIONS_LOG.md](./EXCLUSIONS_LOG.md).

## 2. Signal cleaning — 400 Hz shared helper, not the original FFT loader

`cm_iteration1.ipynb` used a one-off local loader: FFT `scipy.signal.resample` + band-pass
**0.1–15 Hz**. The script uses the maintained shared helper: `resample_poly` + **0.1–20 Hz** at
**400 Hz** — identical to the feature-extraction notebooks. Effect: small rate shifts (typically
< ~0.2 Hz); direction/story unchanged. The FFT method is retired (only `cm_iteration1` +
`r34_analysis5` still contain it — see the FFT audit in [DROPS_AND_INPUTS.md](./DROPS_AND_INPUTS.md) §2).

## 3. Recording set — canonical QC set, not globbed/hard-coded

`cm_iteration1` iterated a hard-coded ~8–9-recording dict (incl. no-BORIS + duplicate-3_5 sessions).
The script builds a canonical set: matching non-empty BORIS required, documented drops applied,
`*_VT` exports preferred. Fewer, cleaner, mutually-consistent points across all plots.

## 4. Bout cap — 20 s, not 30 s

`cm_iteration1` kept `Duration ≤ 30 s`; the script keeps `0.5 ≤ Duration ≤ 20 s` (the feature-
extraction convention), which also removes the single ~400 s anogenital scoring error and sub-0.5 s
noise.

## 5. Valence resp-rate box — newly constructed

No confirmed original whole-session Positive-vs-Negative resp-rate box existed (the nearest valence
resp plots used a different quantity, per-bout BreathMetrics `breathing_rate_hz`). The script's
`resp_rate_valence` is whole-session peak-count rate, RI1 vs RI2, with `5_13` dropped (outlier /
broken RI2). Result: clean Positive ≫ Negative.

---

## Dead-signal handling (answers "was dead signal cut out?")

**No within-recording dead-signal trimming is done.** Whole-session rate spans the entire
`[t[0], t[-1])`; there is no detector for flat/saturated/disconnected stretches, so a dead segment
would simply contribute few/no peaks and **dilute (lower)** that recording's rate. Dead signal is
handled only at the **whole-subject** level, manually: valence **`5_13`** is dropped because its
RI2 trace is flat/broken (std ≈ 20 vs ~700–1000). If a kept recording has a partial dead stretch it
is *not* excised — flag it upstream if that matters. (Same for the interaction-rate plot: each sniff
bout's `[Start, Stop)` window is used as-is.)
