# Data inventory: raw recordings, h5s and BORIS per cohort

> Built 2026-10-13 by `build_data_inventory.py` (read-only scan of UF Dropbox + repo).
> Per-recording detail: `DATA_INVENTORY_recordings.csv`. Re-run the script after adding files.
> Exclusions are NOT applied here; see `EXCLUSIONS_LOG.md` for what each analysis drops.

**Raw complete** = session folder has the `.rec`, an analog export with `ECU_Ain1` + `timestamps.dat`,
`.videoTimeStamps`, and the mp4 (everything the new h5 converter needs).

## Summary

| Cohort | Experiment | Recordings | Raw complete | h5 | BORIS | Recording type |
|---|---|---|---|---|---|---|
| **1 Aim1** (`2025\ECG_cohort1`) | Cagemate baseline | 9 | 9 | 9 | 9 | wireless, merged |
| | Cagemate 1-day isolation | 8 | 8 | 8 | 0 | wireless, merged |
| | Cagemate 10-day isolation | 8 | 8 | 8 | 0 | wireless, merged |
| | Valence BLRI (baseline) | 8 | **0 (on E: drive)** | 8 | 0 | wireless, merged |
| | Valence RI1 (positive) | 9 | **0 (on E: drive)** | 9 | 8 | wireless, merged |
| | Valence RI2 (negative) | 9 | **0 (on E: drive)** | 9 | 9 | wireless, merged |
| **2 BLA pilot 1** (`2025\BLA_ChR_resp_pilot1 (SS)`, Jan 2026) | Cagemate home cage | 24 | 24 | 24 | 24 | tethered |
| | Baseline (`bl_cc`) | 23 | 23 | **0** | 0 | tethered |
| | RI1 day 1 | 24 | 24 | 24 | 24 | tethered |
| | RI1 day 2 | 24 | 23 | 24 | 24 | tethered |
| | RI2 post day 1 | 23 | 23 | 23 | 23 | tethered |
| | RI2 post day 2 | 23 | 23 | 23 | 23 | tethered |
| | OFT | 24 | 24 | 24 | 0 | tethered |
| | `wrong_pulse` | 7 | 6 | **0** | 0 | tethered |
| **3 BLA cohort 2** (`2026\BLA_ChR_resp_cohort2`, Jul 2026) | Cagemate home cage | 11 | 11 | 11 | 11 | tethered |
| | RI1 day 1+2 | 44 | 42 | 44 | 44 | tethered |
| | OFT | 22 | 22 | **0** | 0 | tethered |

## Where things are and what format the h5s are

| Cohort / experiment | h5 folder | h5 format |
|---|---|---|
| Aim1 cagemate baseline | `Day1_new\CM interactions\cm_h5\baseline_cagemate_interactions_h5_outputs` | old: top-level `resp`, **no timestamps, no video mapping** |
| Aim1 cagemate 1d / 10d | repo `src\file_conversion\h5_outputs\cagemate_day_1_h5_outputs`, `cagemate_day_10_h5_outputs` (10d folder also holds baseline copies) | same old format |
| Aim1 valence | `Day1_new\Resp_h5` (26 files) | same old format; `BLRI_s5_13` has **no resp dataset** |
| Pilot 1 cagemate | `BLA_resp_ChR_cm_hc\h5_outputs` (+ SLEAP exports in `SLEAP_id_corrected_files`) | old: `analog`/`dio`/`time` + `resp_clean`; has Trodes timestamps, **no video mapping** |
| Pilot 1 RI1 d1/d2, RI2 d1/d2, OFT | each folder's `h5_outputs` / `BLA_OFT_h5_outputs` | old: `analog`/`dio`/`time` (ECU only); has timestamps, no video mapping |
| Cohort 3 cagemate | `BLA_ChR_resp_CM_hc\h5_outputs` | **NEW v2** (timestamps + video mapping + BLA-compatible groups) |
| Cohort 3 RI1 d1+2 | `BLA_ChR_resp_RI1_hc_day1_2\h5_outputs` | old: `analog`/`dio`/`time` (ECU only) |
| **All cagemate, cohorts 1–3** (rebuilt 2026-10-27; 60 sessions incl. Aim1 1d/10d) | repo `exports\cagemate_h5\` (git-ignored; to be uploaded to Dropbox) | **NEW v2**, plus `behavior_aligned/` BORIS→resp CSVs and `manifest.csv`. Not yet used by any analysis. |

BORIS folders: Aim1 `CM interactions\cm_boris\baseline csv`, `Day1_new\boris_csv_RI1 and RI2`;
pilot 1 `BLA_resp_ChR_cm_boris`, `RI1_day1_boris`, `RI1_day2_boris`, `RI2_day1_boris_post`, `RI2_day2_boris_post`;
cohort 3 `…\BORIS_csv` inside each experiment folder.

## Issues to resolve

1. **Aim1 has no BLRI recording for mouse 3_5** (RI1 and RI2 for 3_5 exist).
2. **Aim1 1-day and 10-day isolation cagemate sessions have no BORIS.**
3. **Pilot 1 rank-label conflict:** session `…20260107_152812` is `8_1_d_cm_8_2_s` (raw folder, one BORIS) but `8_1_s_cm_8_2_d` (old h5 and the inner `.rec`, other BORIS). Which mouse is dominant? The 2026-10-27 export uses the folder name.
4. **Pilot 1 RI1 day 2:** raw folder `6_2_ri2_m2_n_on_20260110_190944` sits in the RI1 day 2 folder and its h5 is named `ri1`. RI1 or RI2?
5. **Missing analog exports** (raw `.rec` exists; needs `trodesexport -analogio` before conversion): pilot 1 `5_1_ri1_m2_m_off_20260110_174649`, `wrong_pulse 1_1_ri1_f1_a_on_20260109_190822`; cohort 3 `12_4_ri1_j_on_20260711_141535`, `11_4_ri1_p_off_20260711_153254`.
6. **No h5 yet:** pilot 1 `bl_cc` (23) and `wrong_pulse` (7); cohort 3 OFT (22).
7. **Name typos (data are fine):** cohort 3 raw folder `11_3_ri1_a_on_60711_094020` (date missing "202"); several pilot 1 BORIS files insert an extra `_2` (e.g. `1_1_2_i_cm_…`).
8. Pilot 1 cagemate BORIS: 8 sessions have two exports, which split the actors: `_VT_CM` holds only `subject` bouts and `_VT_SA` only `social_agent` bouts. Merge them (`cagemate_export.py` does).
9. Unmatched BORIS: `CM_s4_8_sub4_7_20250623.csv` (Aim1; already excluded as a bad duplicate).
