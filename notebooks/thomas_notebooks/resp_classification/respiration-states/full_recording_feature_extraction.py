"""
Full-recording BreathMetrics feature extraction (3 s windows).

Tiles every recording end-to-end instead of sampling 100 random windows: one row per
consecutive WINDOW_SEC window that survives the dead-signal + min-breath gates.
`WindowIndex` keeps the temporal slot so a recording reads back as an ordered sequence.

  valence : Resp_h5/  BLRI (baseline) + RI1 (positive) + RI2 (negative)
  rank    : CM interactions/cm_h5/baseline_cagemate_interactions_h5_outputs/

Outputs to --out (default ../data/whole-session-final):
  full-recording-features-2sec.parquet (.pkl fallback)   window-level feature matrix
  full-recording-manifest.csv                            one row per recording + QC

  python full_recording_feature_extraction.py --jobs 8
  RESP_DATA_ROOT=/blue/.../Day1_new python full_recording_feature_extraction.py --jobs 16
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time as _time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

_HELPERS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HELPERS))

from resp_helper_functions import (  # noqa: E402
    extract_features_from_windows,
    fit_bm_session,
    load_clean_resp_signal,
    load_rank_resp_signal_interactions,
    sample_random_nonoverlapping_windows,
)

DEFAULT_ROOT = Path(os.environ.get(
    "RESP_DATA_ROOT",
    r"C:\Users\thoma\UF Dropbox\Thomas Heeps\Padilla-Coreano Lab\2025"
    r"\ECG_cohort1\Aim1\AIM1\Day1_new",
))
DEFAULT_OUT = _HELPERS / "data" / "whole-session-final"

WINDOW_SEC = 2.0             # output filenames follow this, e.g. ...-features-2sec
TARGET_SRATE = 400
MIN_BREATHS_PER_WINDOW = 4   # n_breaths is stored per row, so a stricter gate is a filter
_ALL_WINDOWS = 10 ** 7       # n_windows >= n candidates -> returns all valid windows

RANK_MAP = {"1_1": "Subordinate", "1_2": "Dominant", "2_3": "Subordinate",
            "2_4": "Dominant", "3_5": "Subordinate", "3_6": "Dominant",
            "4_7": "Subordinate", "4_8": "Dominant"}
VALENCE_CONDITION = {"BLRI": "Baseline", "RI1": "Positive", "RI2": "Negative"}

# Excluded recordings. Mirrors docs/EXCLUSIONS_LOG.md -- do NOT diverge from that list
# without asking the user first, including for analyses that appear not to need BORIS.
DROP = {
    # Unusable signal.
    "BLRI_s5_13_p5_4_nRB3_20250622_175809_merged.h5": "no resp dataset in H5",
    "RI2_s5_13_p5_4_nRB3_20250622_184136_merged.h5": "flat/broken resp signal (5_13)",
    "RI1_s5_13_p5_4_nRB3_20250622_181801_merged.h5": "resp-rate outlier (5_13)",
    "CM_s3_5_d3_6_20250623_170708_merged.h5": "duplicate 3_5 cagemate session",
    # No BORIS export -> dropped project-wide, signal quality notwithstanding.
    "RI1_s3_5_p5_4_nRB3_20250621_105014_merged.h5": "no BORIS export",
    "CM_s3_6_sub3_5_20250623_174348_merged.h5": "BORIS export is empty",
    # Incomplete session (213 s vs ~605 s) and, with RI1_3_5 gone, unpaired.
    "RI2_s3_5_p5_4_nRB3_20250621_112618_merged.h5": "incomplete session (213 s), unpaired",
    # Short (355 s) and, with 3_6's cagemate recording gone, the lone occupant of cage 3.
    "CM_s3_5_d3_6_20250623_160001_merged.h5": "short session (355 s), singleton cage",
}

_VAL_RE = re.compile(r"^(BLRI|RI1|RI2)_s(\d+_\d+)_", re.IGNORECASE)
_CM_RE = re.compile(r"^CM_s(\d+_\d+)_(?:d|sub)(\d+_\d+)_", re.IGNORECASE)


def build_catalog(root: Path) -> pd.DataFrame:
    """One row per usable recording on disk."""
    dirs = [("valence", root / "Resp_h5", _VAL_RE),
            ("rank", root / "CM interactions" / "cm_h5"
             / "baseline_cagemate_interactions_h5_outputs", _CM_RE)]
    rows = []
    for dataset, d, rex in dirs:
        for p in sorted(d.glob("*.h5")):
            m = rex.match(p.name)
            if not m:
                print(f"  ! unparsed filename, skipped: {p.name}")
                continue
            if p.name in DROP:
                print(f"  - dropped {p.name}: {DROP[p.name]}")
                continue
            if dataset == "valence":
                subj, cond, cm = m.group(2), VALENCE_CONDITION[m.group(1).upper()], ""
            else:
                subj, cond, cm = m.group(1), "CagemateInteraction", m.group(2)
            rows.append(dict(
                RecordingID=p.stem, Dataset=dataset, Condition=cond, Subject=subj,
                Cagemate=cm, Cage=subj.split("_")[0], Rank=RANK_MAP.get(subj, ""),
                Path=str(p)))

    if not rows:
        raise FileNotFoundError(f"No H5 files found under {root}")
    return pd.DataFrame(rows)


def extract_one(meta: dict) -> tuple:
    """Tile one recording into 3 s windows and extract BreathMetrics features."""
    t0 = _time.time()
    stat = {k: v for k, v in meta.items() if k != "Path"}
    loader = (load_clean_resp_signal if meta["Dataset"] == "valence"
              else load_rank_resp_signal_interactions)
    signal, time, fs, _ = loader(meta["Path"], target_rate=TARGET_SRATE, verbose=False)

    bm = fit_bm_session(signal, fs, data_type="rodentAirflow") if signal is not None else None
    if bm is None:
        stat.update(status="failed", n_windows_kept=0)
        return [], stat

    session_dur = float(time[-1] - time[0])
    windows = sample_random_nonoverlapping_windows(
        time=time, signal=signal, window_dur=WINDOW_SEC, n_windows=_ALL_WINDOWS,
        seed=0, allow_partial_if_short=False)
    rows = extract_features_from_windows(
        bm=bm, windows=windows, min_breaths_per_window=MIN_BREATHS_PER_WINDOW)

    for row in rows:
        # Temporal slot index: gaps = windows rejected by the gates, so the sequence
        # structure survives for downstream state / HMM work.
        row["WindowIndex"] = int(round((row["Start"] - float(time[0])) / WINDOW_SEC))
        row.update({k: v for k, v in meta.items() if k != "Path"})

    n_possible = int(session_dur // WINDOW_SEC)
    stat.update(status="ok", session_duration_sec=round(session_dur, 2),
                n_breaths_total=int(len(getattr(bm, "inhaleOnsets", []))),
                n_windows_possible=n_possible, n_windows_kept=len(rows),
                window_yield=round(len(rows) / n_possible, 3) if n_possible else np.nan,
                elapsed_sec=round(_time.time() - t0, 1))
    return rows, stat


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--dataset", choices=("valence", "rank"))
    ap.add_argument("--limit", type=int, help="process only the first N recordings")
    ap.add_argument("--jobs", type=int, default=1)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Data root: {args.root}")
    cat = build_catalog(args.root)
    if args.dataset:
        cat = cat[cat["Dataset"] == args.dataset]
    if args.limit:
        cat = cat.head(args.limit)
    metas = cat.to_dict("records")
    print(f"Extracting {len(metas)} recordings, {WINDOW_SEC:g}s windows, jobs={args.jobs}")

    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            results = list(ex.map(extract_one, metas))
    else:
        results = [extract_one(m) for m in metas]
    for rows, stat in results:
        print(f"  {stat['RecordingID']}: {stat['status']}, {stat['n_windows_kept']} windows")

    manifest = pd.DataFrame([s for _, s in results])
    manifest.to_csv(args.out / "full-recording-manifest.csv", index=False)

    all_rows = [r for rows, _ in results for r in rows]
    if not all_rows:
        print("No windows extracted.")
        return 1
    df = (pd.DataFrame(all_rows)
          .sort_values(["Dataset", "RecordingID", "WindowIndex"]).reset_index(drop=True))
    path = args.out / f"full-recording-features-{WINDOW_SEC:g}sec.parquet"
    try:
        df.to_parquet(path, index=False)
    except Exception as exc:  # noqa: BLE001  (pyarrow may be absent)
        print(f"  parquet write failed ({exc}); falling back to pickle")
        path = path.with_suffix(".pkl")
        df.to_pickle(path)

    print(f"\nwrote {path} and the manifest")
    print(f"{len(df)} windows x {len(df.columns)} cols from "
          f"{df['RecordingID'].nunique()} recordings")
    print(df.groupby(["Dataset", "Condition"]).size().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
