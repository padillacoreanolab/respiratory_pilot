# Build / audit note: respiration methods deck (2026-10-13)

Deck: `respiration_methods_deck.pptx` (8 slides). Figures: `figures/` (PNG 300 dpi + SVG).
Provenance per plot: `figures/manifest.txt`; computed numbers: `figures/facts.json`.
Reproduce: `make_figures.py` (biopipeline-env), then `node build_deck.js` (pptxgenjs).
Nothing in the detector, the loaders, the H5s or the notebook was modified.

## Sources and verification
- The dated report was not attached and was not found on disk. Its numbers were taken from the request and checked against the notebook's saved run (cell 28) and an independent recomputation. All of them match.
  - 30 Aim1 recordings; 111,872 peaks; 262.813 min analysed; 31.21 min excluded; 22/30 recordings affected; 7.0945 Hz.
  - Section 1: OLD 0 / NEW 14, 17, 13 (11_3). Controls: 10_3 13/13; 11_2 dropout 0/0; 11_2 after dropout 15/15; 11_2 499.8 s 0/15.
- Detector code: cells A–C are executed directly from the notebook (`deck_common.py`). A replay keeps the intermediates and asserts that its final peaks, troughs and baseline equal `detect_breaths()` on every recording used. It also matches snapshot v1 on all 30 recordings, and the notebook matches `adaptive_breath_detection.py` (Section 5 output: True).
- The suspected notebook/support mismatch (KeyError on `candidates` / `blocks` / `params`) was not found. The current cell C returns all three, and the saved run completed.
- Slide 2: the stages were recomputed from the raw 20 kHz H5. The final stage equals `workbench_cache/CM_2_4_sub2_3.npz` exactly (max |diff| = 0).
- Slide 1: the H5 `/resp` has 12,439,272 samples, identical to the Trodes export `ECU_Ain1.dat` (merged, filler kept).
  - The real `timestamps.dat` has 28,059 repeated ticks, which stretches index time by 0.226% (+1.40 s by the session end).
  - The inset on slide 1 is computed from that file.

## Differences between the request and the repository / data
1. **There are no `shallow_trough` rejections anywhere in the accessible data**: 0 in all 30 Aim1 recordings and 0 in the 3 cohort-3 Section 1 sessions. Slide 7 Pair B is therefore a second accepted pair: the closest accepted pair in all of Aim1 (BLRI_1_1, 529.49–529.63 s, Δ = 297.4 against need = 297.3, ratio 1.0004). No failure case was fabricated.
2. **Slide 4, top example (BLRI_3_6, 15–16 s):** the block really is classified `low_amp (periodic)` (rel 0.016, ACF 0.82). However, its ACF maximum lies at the 60 ms lag boundary on a smooth drift, so the "periodicity" is drift, not an oscillation.
   - 109 of 315 Aim1 `low_amp (periodic)` passes have their ACF maximum at that boundary. This is a weakness of the periodicity test worth knowing.
   - A better replacement example was identified but not yet swapped in, for lack of time: RI2_2_4, block 413.5 s, ACF 0.82 with an interior peak at 175 ms lag, inside a frozen-QC "dead" stretch.
3. Request numbering: "cohort 2" on slide 1 is used as written. The BLA pilot (overall cohort 2) does use `/analog/...ECU_Ain1 + /analog/...timestamps` and plain `.rec` files. The Section 1 sessions are from **BLA cohort 2 = overall cohort 3** (July 2026), labelled "BLA cohort 2" on slide 8 as requested.
4. Slide 5 and slide 6 use the same real segment (BLRI_4_7, 530.0–534.5 s). Pass A and Pass B both give 24 peaks there (the counts are equal; no improvement is claimed).
5. Slide 7's main panel shows the final y (baseline recomputed from the final peaks) with the final troughs. The pair zooms show y_A, the array that `_validate_cycles` actually tests.
6. Slide 8 draws the same 400 Hz input trace on both sides. The OLD cutoff (714) applies to legacy's own baseline-corrected signal and is quoted as text only.
7. The workbench (and this deck) keeps RI1_3_5, RI2_3_5 and CM_3_5 (160001), which `EXCLUSIONS_LOG.md` drops from the analyses (no BORIS / short session). This is fine for detector checks but differs from the analysis sets.

## Known cosmetic issues left for lack of time (fix list started in make_figures.py)
- s1: the recording tag overlaps the panel title.
- s3 main: the "degraded / dead / artifact" labels collide near 300–310 s.
- s3 zoom (a): the window includes 3 s of degraded before the dead stretch.
- s5: the "HP₁Hz" subscript glyph is missing in the title, and the prominence and width labels overlap in the geometry zoom.
- s6: the tag overlaps the title.
- s7 pairs: the long titles and tags overlap.
- No micro-inset was added to the slide 8 strip. All Section 1 controls are in `figures/s8_supp_*.png`.
