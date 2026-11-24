"""
Rebuild the cohort 1 (Aim1) valence sessions (BLRI / RI1 / RI2) with
ekg_resp_to_h5.py. Replaces the July-2025 `Day1_new/Resp_h5` files, which kept
the merge filler samples and had no /timestamps or /video timing.

Input: one `<session>.rec/` folder per session directly under Day1_new, each
with `<session>_merged.analog/` (trodesexport -analogio) and the
`.videoTimeStamps` + `.1.mp4`.

Output:
    <out>/{RI1,RI2,BLRI}/<session>_merged.h5
    <out>/manifest.csv     one row per session: duration, QC, video timing, warnings, notes

RI1 and RI2 are converted first, BLRI last. Existing H5s are skipped, so the
script can be re-run after a missing export is added (--force rebuilds them).

Run (biopipeline-env):  python valence_export.py [out_dir]
"""
import os
import sys
import glob
import json
from concurrent.futures import ProcessPoolExecutor

import h5py
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ekg_resp_to_h5 as E

LAB = r"C:\Users\thoma\UF Dropbox\Thomas Heeps\Padilla-Coreano Lab"
SRC = LAB + r"\2025\ECG_cohort1\Aim1\AIM1\Day1_new"
OUT = LAB + r"\2025\ECG_cohort1\Aim1\new_resp_h5_valence"
CONDITIONS = ("RI1", "RI2", "BLRI")  # processing order: non-baseline first

# Documented in notebooks/.../docs/EXCLUSIONS_LOG.md. Exported anyway, flagged here.
NOTES = {
    "s5_13": "subject 5_13 excluded in all valence analyses (EXCLUSIONS_LOG)",
    "RI1_s3_5": "no BORIS: excluded (EXCLUSIONS_LOG)",
    "RI2_s3_5": "incomplete recording, excluded from full-recording pipeline (EXCLUSIONS_LOG)",
    "RI1_s2_3": "two BORIS files; final one not yet chosen",
    "BLRI_s3_5": "new: not in the old Day1_new/Resp_h5 set",
}


def session_jobs(out_dir):
    jobs = []
    for cond in CONDITIONS:
        dest = os.path.join(out_dir, cond)
        os.makedirs(dest, exist_ok=True)
        for folder in sorted(glob.glob(os.path.join(SRC, cond + "_s*.rec"))):
            recs = glob.glob(os.path.join(folder, "*merged.rec")) or glob.glob(os.path.join(folder, "*.rec"))
            if not recs:
                print("no .rec in", folder)
                continue
            jobs.append((cond, recs[0], os.path.join(dest, E.session_name(recs[0]) + ".h5")))
    return jobs


def convert_one(job, force=False):
    _, rec, out_h5 = job
    if os.path.exists(out_h5) and not force:
        return out_h5, "exists, skipped"
    try:
        data = E.get_data_from_rec_file(rec)
    except Exception as e:  # e.g. analog export not made yet
        return out_h5, f"FAILED: {e}"
    if data is None:
        return out_h5, "FAILED: resp channel missing"
    tmp = out_h5 + ".part"
    E.export_to_h5(tmp, data)
    os.replace(tmp, out_h5)  # Dropbox never sees a half-written .h5
    return out_h5, "ok"


def manifest_row(cond, h5_path):
    name = os.path.basename(h5_path)[:-3]
    row = {"condition": cond, "session": name, "h5": os.path.relpath(h5_path, os.path.dirname(os.path.dirname(h5_path)))}
    with h5py.File(h5_path, "r") as f:
        rm, md = f["resp_metadata"].attrs, f["metadata"].attrs
        row.update(duration_min=round(float(rm["duration_sec"]) / 60, 2),
                   n_samples=int(rm["num_samples"]),
                   duplicates_removed=int(rm["n_duplicate_samples_removed"]),
                   timestamp_gaps=int(rm["n_timestamp_gaps"]),
                   imu=",".join(sorted(f["imu"].keys())))
        if "video" in f:
            va = f["video"].attrs
            row.update(mp4_fps=float(va.get("mp4_fps", float("nan"))),
                       fit_fps=float(va.get("fit_fps", float("nan"))))
        row["warnings"] = "; ".join(json.loads(md["warnings"]))
    row["notes"] = "; ".join(v for k, v in NOTES.items() if k in name)
    return row


def _convert_forced(job):
    return convert_one(job, force=True)


def run(out_dir=OUT, workers=3, force=False):
    jobs = session_jobs(out_dir)
    print(f"{len(jobs)} valence sessions -> {out_dir}", flush=True)
    with ProcessPoolExecutor(workers) as ex:
        for out_h5, status in ex.map(_convert_forced if force else convert_one, jobs):
            print(f"{status}: {out_h5}", flush=True)
    rows = [manifest_row(c, h) for c, _, h in jobs if os.path.exists(h)]
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "manifest.csv"), index=False)
    print(f"manifest: {len(rows)} of {len(jobs)} sessions")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--force"]  # --force: rebuild existing H5s in place
    run(args[0] if args else OUT, force="--force" in sys.argv)
