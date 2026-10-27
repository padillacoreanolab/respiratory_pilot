# Exclusions log

This is the binding list of everything left out of each dataset. Update it in the same change as
any new drop, and **ask before diverging from it**. For recording and timing caveats, see
[docs/DATA_README.md](../../../../docs/DATA_README.md).

> **Rule:** a recording with no usable BORIS export is dropped everywhere. That includes
> window-level pipelines that never read behavior.

---

## Part 1: Data drops

### Cohorts

| Dataset | Uses | Excluded |
|---|---|---|
| Rank (cagemate) | Cohort 1 (Aim1) baseline CM | Cohort 2 (BLA pilot): no baseline. Cohort 3: recorded after these datasets were built. |
| Valence | Cohort 1 (Aim1) BLRI / RI1 / RI2 | — |

### Rank: cohort 1 cagemate

| Excluded | Pipelines | Why |
|---|---|---|
| Subject `3_6` | all | BORIS export is empty |
| `CM_s3_5_d3_6_…170708` | all | Duplicate 3_5 session with no BORIS (`…160001` is kept) |
| `CM_s3_5_d3_6_…160001` | full-recording only | Short (355 s vs about 605 s) and the only recording left in cage 3, so 3_5 is absent there |
| BORIS `CM_s4_8_sub4_7_20250623.csv` | all | Bad duplicate: tab-separated, no timestamp. Use `…182649_VT`. |
| *Kept:* `1_1`, `1_2` | — | Fighting occurred, but the cohort is too small to drop a cage |

### Valence: cohort 1 RI

| Excluded | Pipelines | Why |
|---|---|---|
| Subject `5_13` (RI1, RI2, BLRI) | all | RI1 rate is an outlier (about 4.9 vs 7.5 Hz); RI2 is flat (std about 20 vs 700–1000); the BLRI H5 has no `resp` dataset |
| `RI1_3_5` | all | No BORIS |
| `RI2_3_5` | full-recording only | Incomplete (213 s) and unpaired, so 3_5 has no valence recording there |

### Resulting sets

| Pipeline | Rank | Valence |
|---|---|---|
| 3-sec windows and interactions | 7 mice / 4 cages: Sub {1_1, 2_3, 3_5, 4_7}, Dom {1_2, 2_4, 4_8} | Pos n = 7, Neg n = 8 |
| Full-recording (`data/whole-session-final/`) | 6 mice / 3 complete cages | 7 mice |

The full-recording set totals 27 recordings, 7,780 windows and 7 mice.

---

## Part 2: Processing drops

| Stage | Rule | Applies to |
|---|---|---|
| Bouts | Drop bouts shorter than 0.5 s or longer than 20 s (the long cutoff removes one ~400 s anogenital scoring error) | Behavior and bout-rate analyses |
| Bouts | Min-sniff gate: at least 10 sniff bouts per subject, **counting both actors** (this keeps `1_2`) | Rank interactions |
| Windows | Min-breaths gate on random windows | Random-window feature extraction |
| Windows | Every 2 s window is kept; dead signal inside a recording is **not** trimmed | Full-recording feature extraction |
| Features | `cv_inhale_pause_dur` and `cv_exhale_pause_dur` dropped (>50 % NaN), leaving 25 | Full-recording |
| Features | Redundant pairs pruned at \|ρ\| > 0.85; PCA runs on the pruned set | Full-recording |

### Known caveats (kept in the data)

- **Actor convention is undecided.** Rank originally counted both actors; valence counted the
  subject only. The poster script plots both.
- `RI1_s2_3_…104059_ss.csv` has only `subject` bouts scored, so for RI1 2_3 the both-actor count
  equals the subject-only count.
- `CM_s1_2_sub1_1_…133932.1.csv` (non-VT) has 7 stray `no focal subject` bouts. The pipelines use
  the `_VT` export instead.
- Valence BORIS was re-scored with longer bouts after the old plots were made. The source of truth
  is `Day1_new\boris_csv_RI1 and RI2`.
- Inhale-pause features are 91 % exact zeros and exhale-pause features 56 %. This matters for
  Gaussian or Euclidean models.
- In 3-sec LOCO, cage 3 is a singleton {3_5}, so its fold is single-class.
