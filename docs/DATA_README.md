# Respiration data: cohorts, recording setups, conversion

Read this before you load any respiration H5 or align it with BORIS or SLEAP. The cohorts were
recorded on different setups, and those differences change the timing.

Session-level inventory: [DATA_INVENTORY.md](../notebooks/thomas_notebooks/resp_classification/docs/DATA_INVENTORY.md).
Binding drop list: [EXCLUSIONS_LOG.md](../notebooks/thomas_notebooks/resp_classification/docs/EXCLUSIONS_LOG.md).

Dropbox root: `UF Dropbox/Thomas Heeps/Padilla-Coreano Lab/`

---

## 1. Cohorts

Cohorts are numbered by recording date. The Dropbox folder names count only BLA cohorts, so
"BLA cohort 2" is **cohort 3**.

| | **Cohort 1: Aim1** | **Cohort 2: BLA pilot** | **Cohort 3: BLA cohort 2** |
|---|---|---|---|
| Folder | `2025/ECG_cohort1/Aim1/AIM1/Day1_new` | `2025/BLA_ChR_resp_pilot1 (SS)` | `2026/BLA_ChR_resp_cohort2` |
| Recorded | 2025-06-21 to 06-23 | 2026-01-05 to 01-20 | 2026-07-11 to 07-29 |
| Sessions | 9 CM; 26 valence (BLRI/RI1/RI2) | 24 CM, 48 RI1, 46 RI2, 24 OFT, 23 bl_cc | 11 CM, 44 RI1, 22 OFT |
| Rig | **Wireless** logger headstage + MCU, merged (`*_merged.rec`) | Tethered, single `.rec`, no headstage | Same as cohort 2 |
| Trodes | 2.6.1 BETA | 2.8.0 | 2.8.0 |
| IMU | Accel and gyro, about 100 Hz (magnetometer off) | **None** | **None** |
| Merge filler samples | About 28k per session (must be removed) | 0 | 0 |
| Camera (C920 webcam) | 30 fps | 15 fps | 15 fps |
| Laser TTL (`ECU_Din1`) | none | RI and OFT sessions | RI and OFT sessions |
| Raw `.rec` on Dropbox | CM yes; **valence no** (H5 only) | yes | yes |

**The same for every cohort:** respiration is on `ECU_Ain1` of the same MCU (serial 01703 00102).
It is sampled at 20 kHz on the Trodes clock and stored as int16 **ADC counts**, not volts.

**Laser:** 5 ms pulses every 55 ms (about 18 Hz) for about 3 min, starting 60 to 75 s into the
session. Only sessions with `_on` in the name have pulses; `_off` sessions have none. The
pulses are stored as state changes `(time tick, state)` in `/dio/*dio_ECU_Din1`.

---

## 2. Timing: behavior alignment

BORIS and SLEAP times are **mp4 seconds**, that is `frame_index / mp4_fps`. They are not
recording seconds. There are three ways this goes wrong:

1. **Wrong mp4 frame rates.** Many mp4s were encoded at a frame rate other than the capture rate,
   so the mp4 timeline is stretched or squeezed. For example, `3_1_i_cm_3_2_d` is 606 s of real
   capture but plays as 378 s at 24 fps.

   | Cohort | Capture fps | Sessions whose mp4 fps differs |
   |---|---|---|
   | 1 CM | 30 | 1 of 9 (`CM_s4_7_d4_8`, 23 fps) |
   | 2 | 15 | **about 92 of 142**: CM 12 of 24, RI1 d1 18 of 24, RI1 d2 20 of 24, RI2 about 26 of 46, OFT 16 of 24 (mp4 at 13 to 32 fps; bl_cc not checked) |
   | 3 | 15 | CM 1 of 11 (`13_2_d…111038`, 16 fps), RI1 5 of 44, OFT 2 of 22 |

2. **Camera starts before the respiration data (cohort 1 only).** The wireless merge makes the
   video start 0 to 1.1 s before the respiration data. In cohorts 2 and 3 the offset is under 0.1 s.

3. **Recording pauses (cohort 2 only).** In these sessions Trodes stopped and then resumed. The
   respiration data has a gap, and `index / 20000` is wrong after the gap.

   | Session | Pause |
   |---|---|
   | `bl_cc/5_3_bl_cc_20260105_132320` | 71.9 s |
   | `OFT/1_1_oft_20260114_151355` | 68.3 s. The video also runs 283 s past the end of the respiration data. |
   | `RI2_post_d2/4_3_RI2_f2_e_on_20260120_143307` | 71.4 s |
   | `RI2_post_d2/7_1_RI2_m1_j_on_20260120_122806` | 41.3 s |
   | `RI2_post_d2/3_1_RI2_f1_b_on_20260120_135800` | `.videoTimeStamps` is **corrupt**. Do not align this video. |

**The fix for all three:** map BORIS and SLEAP times through the frame index.

```python
from src.file_conversion.ekg_resp_to_h5 import video_time_to_resp_time
t_resp = video_time_to_resp_time(h5_path, boris_df["Start (s)"])  # seconds from /resp[0]
# or by hand: frame = round(t * f['video'].attrs['mp4_fps']); i = f['video/frame_to_resp_index'][frame]
```

This only works on H5s made by the current `ekg_resp_to_h5.py`, which writes `/video`. Older
H5s have no video timing (see section 4).

---

## 3. Converting: `src/file_conversion/ekg_resp_to_h5.py`

This is the canonical script for converting Trodes `.rec` files to H5. Use it for all new conversions.

**Steps**

1. In each session folder, run `trodesexport -analogio -dio -rec <file>.rec`. For a wireless
   session, run it on `*_merged.rec`. The script reads these exports and does not create them.
2. Set `data_root` at the bottom of the script, then run it in **biopipeline-env** (it needs
   neurokit2 to write `/resp_clean`). `ffprobe` must be on PATH; without it the script assumes
   30 fps and flags the value. The script recurses into subfolders and uses one `.rec` per
   session folder, choosing `*_merged.rec` if there is one.
3. Read every `WARNING` the script prints. The same warnings are saved in `metadata.attrs['warnings']`.

**What it handles automatically:**
- It removes merge filler samples (cohort 1).
- It writes IMU data only when the headstage recorded it.
- It fits frame times separately on each side of a pause.
- It flags corrupt frame timestamps and wrong mp4 frame rates.
- It names each output after the session **folder**. In cohort 2 the `.rec` file inside some
  folders has a different, truncated, or role-swapped name.

**Per cohort**

| Cohort / type | Status | Notes |
|---|---|---|
| 1 CM | Can be re-run | Existing H5s were made with the Aug-2025 version, which kept the filler samples and had no video timing. All cohort 1 results come from those files. |
| 1 valence | The raw `.rec` files are not on Dropbox. The H5s still contain the filler samples (about 0.22 % time stretch) and the camera offset. |
| 2 (all types) | Will be re-run, not done yet | Write to a **new** folder (e.g. `h5_outputs_v2`), because `h5_outputs/` holds the old layout and some names differ. `RI1_hc_day2/5_1_RI1_m2_m_off_20260110_174649` has lost its exports: re-export it first. bl_cc has never been converted. |
| 3 CM | **Done** (2026-10-01) | Reproduced exactly by the current script. |
| 3 RI1 | Old layout | The exports for `11_4_RI1_p_off_20260711_153254` and `12_4_RI1_j_on_20260711_141535` are missing: re-export them before re-running. |
| 3 OFT | No H5s | |

**Output layout** (one file per session)

| Path | Contents |
|---|---|
| `/resp` | float32, 20 kHz ADC counts, filler samples removed |
| `/timestamps` | uint32 Trodes tick for each sample. **Use this for time when there are gaps.** |
| `/resp_metadata`, `/metadata` (attrs) | fs, start tick, wall-clock start (`resp_start_unix_ms`), QC counts, warnings |
| `/imu/accel_*`, `/imu/gyro_*` | cohort 1 only, raw counts, sample-and-hold at 20 kHz |
| `/video/frame_to_resp_index`, `frame_time_sec` | frame → resp mapping (`-1` means outside the recording). attrs: `mp4_fps`, `fit_fps`, `mp4_time_scale`, `camera_gaps_sec` |
| `/analog/<name>.analog_ECU_Ain1` (`'voltage'`), `/resp_clean/{signal,time}` (100 Hz) | Copies in the BLA notebook layout, so older notebooks work unchanged |
| `/dio/<name>.dio_ECU_Din1` | laser TTL state changes (when exported) |

---

## 4. Existing H5s: which script made them

| Folder | Script | Layout and gaps |
|---|---|---|
| C1 valence `Day1_new/Resp_h5` | `analog_h5.py` (deleted) | `resp` (N,1) float64, `ekg`; filler samples kept; no timestamps or video |
| C1 CM `CM interactions/cm_h5/baseline_cagemate_interactions_h5_outputs` | `ekg_resp_to_h5.py`, Aug 2025 | `resp` float32; filler samples kept; `units='volts'` is wrong (the values are ADC counts) |
| C2 CM `BLA_resp_ChR_cm_hc/h5_outputs` | `trodes_ECU_to_h5_clean_raw.py` | `analog/`, `dio/`, `time/`, `resp_clean`; no video timing |
| C2 RI1/RI2/OFT `*/h5_outputs`, `BLA_OFT_h5_outputs` | `resp_to_h5_ecu_only.py` | `analog/`, `dio/`, `time/`; no `resp_clean`; no video timing |
| C3 CM `BLA_ChR_resp_CM_hc/h5_outputs` | current `ekg_resp_to_h5.py` | full layout (section 3) |
| C3 RI1 `BLA_ChR_resp_RI1_hc_day1_2/h5_outputs` | `resp_to_h5_ecu_only.py` | as C2 RI |

All of these layouts load through `resp_helper_functions`. It reads `analog/*ECU_Ain1` if present,
otherwise `resp`, then low-pass filters to 200 Hz, downsamples to 400 Hz and band-passes 0.1 to 20 Hz.
**That loader uses `index / fs` as time and BORIS seconds as-is**, so behavior-locked results for
the sessions in section 2 carry those errors until the analysis switches to frame-index mapping.

The two legacy scripts (`trodes_ECU_to_h5_clean_raw.py`, `resp_to_h5_ecu_only.py`) are kept only
to explain the old files. They hard-code 20 kHz, store no frame timing, and have placeholder paths.

---

## 5. Naming pitfalls

- **Match files on the `YYYYMMDD_HHMMSS` stamp, not on the full name.** In cohort 2 the BORIS file
  names disagree with the session names, for example `1_1_2_i_cm_1_2_d_…_BORIS_SS.csv` against
  `1_1_i_cm_1_2_d_…`.
- In cohort 2, the folder and the inner `.rec` have different names in three sessions:
  - `1_1_i_cm_1_2_d` contains `1_1_d_cm_1_2_i`.
  - `8_1_d_cm_8_2_s` contains `8_1_s_cm_8_2_d`.
  - `2_1_d_cm_2.3_i_20260106_105109` contains `2_1_d_cm_2.rec`.

  It is not known which d/s role is correct.
- Cohort 2 also has:
  - a duplicate file, `2_2_s_cm_2_1_d_20260108_120453.h5` / `.rec.h5`;
  - an RI2-named session in the RI1 day-2 folder (`6_2_RI2_m2_n_on_20260110_190944`).
- Cohort 3: `11_3_RI1_a_on_60711_094020` is missing "202" in its date.
- Role letters: `s` = subject, `d` = dominant, `i` = intermediate. `_on`/`_off` = laser.

## 6. Signal quality (summary; see EXCLUSIONS_LOG for decisions)

- Cohort 2: mice 2_3, 5_1, 5_2, 5_3 and 6_3 have noise-only respiration in all four RI sessions,
  but normal CM sessions.
- Cohort 3: 16 of 44 RI1 sessions have low signal for the whole session.
- Cohort 3 CM, proposed QC (not yet adopted): `10_2_d`, `11_3_s` and `14_1_d` are bad; `11_2_s` is borderline.
