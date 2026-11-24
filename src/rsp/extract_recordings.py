"""
Full-recording BreathMetrics features: every recording tiled into consecutive windows.

One BreathMetrics fit per recording, then one row per window that passes the
dead-signal and min-breath gates. `WindowIndex` is the window's slot in time, so a
recording reads back as an ordered sequence (gaps = rejected windows).

Input: the curated folder (one subfolder per cohort, then rank/ and valence/,
bad recordings already removed):
    <root>/cohort1_2025_aim1/{rank,valence}/...h5
Output (--out): full-recording-features-<window>s.csv + full-recording-manifest.csv

    python -m src.rsp.full_recording_feature_extraction --jobs 8
    python -m src.rsp.full_recording_feature_extraction --cohort cohort1_2025_aim1 --window 3 --step 1
"""
from __future__ import annotations

import argparse
import os
import re
import time as _time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from .bm_features import (SUPPORTED_WINDOW_FEATURES, extract_features_from_windows,
                          fit_bm_session, tile_windows)
from .resp_processing import load_raw_resp, preprocess_resp

DEFAULT_ROOT = Path(os.environ.get(
    "RESP_DATA_ROOT",
    r"C:\Users\thoma\UF Dropbox\Thomas Heeps\Padilla-Coreano Lab\2026\valence_rank_respiration"))
DEFAULT_OUT = Path(__file__).resolve().parents[2] / "notebooks" / "thomas_notebooks" / "data" / "full-recording-features"

# Excluded from full-recording pipelines only (RESP_ANALYSIS_GUIDE.md, Exclusions); the
# curated folder already removed every other documented drop.
FULL_RECORDING_DROPS = {
    "CM_s3_5_d3_6_20250623_160001_merged": "short session (355 s), singleton cage",
    "RI2_s3_5_p5_4_nRB3_20250621_112618_merged": "incomplete session (213 s), unpaired",
}
RANK_MAP_COHORT1 = {"1_1": "Subordinate", "1_2": "Dominant", "2_3": "Subordinate", "2_4": "Dominant",
                    "3_5": "Subordinate", "3_6": "Dominant", "4_7": "Subordinate", "4_8": "Dominant"}
VALENCE_CONDITION = {"BLRI": "Baseline", "RI1": "Positive", "RI2": "Negative"}


def build_catalog(root=DEFAULT_ROOT, drop=FULL_RECORDING_DROPS) -> pd.DataFrame:
    """One row per H5 under <root>/<cohort>/<rank|valence>/ (any depth below that)."""
    rows = []
    for p in sorted(Path(root).glob("cohort*/*/**/*.h5")):
        cohort, dataset = p.relative_to(root).parts[:2]
        if p.stem in drop:
            print(f"  - dropped {p.stem}: {drop[p.stem]}")
            continue
        # subject: leading "10_2_..." (cohorts 2-3) or "..._s1_1_..." (cohort 1)
        m = re.match(r"(\d+_\d+)_", p.stem) or re.search(r"_s(\d+_\d+)_", p.stem)
        subject = m.group(1) if m else ""
        cond = next((v for k, v in VALENCE_CONDITION.items() if re.search(rf"(^|_){k}_", p.stem, re.I)), "")
        laser = (re.search(r"_(on|off)_", p.stem) or [None, ""])[1]
        rows.append(dict(RecordingID=p.stem, Cohort=cohort, Dataset=dataset,
                         Session=p.parent.name if p.parent.name != dataset else "",
                         Subject=subject, Condition=cond if dataset == "valence" else "Cagemate",
                         Laser=laser,
                         Rank=RANK_MAP_COHORT1.get(subject, "") if cohort.startswith("cohort1") else "",
                         Path=str(p)))
    if not rows:
        raise FileNotFoundError(f"No cohort*/<dataset>/*.h5 files under {root}")
    return pd.DataFrame(rows)


def extract_recording_features(h5_path, window_sec=2.0, step_sec=None, min_breaths=4,
                               target_fs=400, peak_method="src", **meta):
    """
    Fit BreathMetrics on one recording and return its window features.

    Returns (DataFrame with one row per kept window, dict of per-recording QC).
    Extra keyword arguments (e.g. Subject=...) are added as columns.
    """
    t0 = _time.time()
    meta = {"RecordingID": Path(h5_path).stem, **meta}   # a passed RecordingID wins
    stat = dict(meta)
    resp, time = preprocess_resp(load_raw_resp(h5_path), target_fs=target_fs)
    time = time - time[0]   # bmObject indexes breaths from sample 0
    bm = fit_bm_session(resp, target_fs, peak_method=peak_method)
    if bm is None:
        return pd.DataFrame(), dict(stat, status="failed", n_windows_kept=0)

    windows = tile_windows(time, window_sec, step_sec, signal=resp)
    rows = extract_features_from_windows(bm, windows, min_breaths_per_window=min_breaths)
    step = window_sec if step_sec is None else step_sec
    df = pd.DataFrame(rows)
    if len(df):
        df.insert(0, "WindowIndex", np.round(df["Start"] / step).astype(int))
        df = df.assign(**meta)

    n_possible = len(tile_windows(time, window_sec, step_sec))
    stat.update(status="ok", duration_sec=round(float(time[-1]), 2),
                n_breaths_total=len(bm.inhaleOnsets), onset_local_fallbacks=bm.nOnsetLocalFallbacks,
                n_windows_possible=n_possible, n_windows_kept=len(df),
                window_yield=round(len(df) / n_possible, 3) if n_possible else np.nan,
                elapsed_sec=round(_time.time() - t0, 1))
    return df, stat


def _extract_row(meta, **kw):
    meta = dict(meta)
    return extract_recording_features(meta.pop("Path"), **kw, **meta)


def run(catalog, out_dir=DEFAULT_OUT, jobs=1, **kw):
    """Extract every catalog row; write features CSV + manifest CSV. Returns the features."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metas = catalog.to_dict("records")
    work = partial(_extract_row, **kw)
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            results = list(ex.map(work, metas))
    else:
        results = [work(m) for m in metas]

    manifest = pd.DataFrame([s for _, s in results])
    manifest.to_csv(out_dir / "full-recording-manifest.csv", index=False)
    df = pd.concat([d for d, _ in results if len(d)], ignore_index=True)
    df = df.sort_values(["Cohort", "Dataset", "RecordingID", "WindowIndex"]).reset_index(drop=True)
    path = out_dir / f"full-recording-features-{kw.get('window_sec', 2.0):g}s.csv"
    df.to_csv(path, index=False)
    print(f"wrote {path}: {len(df)} windows from {df.RecordingID.nunique()} recordings")
    print(f"Supported features: {SUPPORTED_WINDOW_FEATURES}; all other feature columns are estimated.")
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--cohort", help="e.g. cohort1_2025_aim1 (default: all)")
    ap.add_argument("--dataset", choices=("valence", "rank"))
    ap.add_argument("--window", type=float, default=2.0, help="window length (s)")
    ap.add_argument("--step", type=float, help="window spacing (s); < window = overlap")
    ap.add_argument("--min-breaths", type=int, default=4)
    ap.add_argument("--peak-method", choices=("src", "legacy"), default="src")
    ap.add_argument("--limit", type=int, help="process only the first N recordings")
    ap.add_argument("--jobs", type=int, default=1)
    a = ap.parse_args()

    cat = build_catalog(a.root)
    if a.cohort:
        cat = cat[cat.Cohort == a.cohort]
    if a.dataset:
        cat = cat[cat.Dataset == a.dataset]
    if a.limit:
        cat = cat.head(a.limit)
    print(f"Extracting {len(cat)} recordings, {a.window:g} s windows, jobs={a.jobs}")
    run(cat, a.out, jobs=a.jobs, window_sec=a.window, step_sec=a.step,
        min_breaths=a.min_breaths, peak_method=a.peak_method)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
