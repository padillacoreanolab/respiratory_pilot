"""
Rebuild every cagemate (CM) session of cohorts 1-3 with ekg_resp_to_h5.py and
align BORIS behavior to respiration through the video frame index.

Output (one folder per cohort, ready to upload):
    <out>/cohort1_aim1_CM/{baseline,1d_isolation,10d_isolation}/<session>.h5
    <out>/cohort2_bla_pilot_CM/<session>.h5
    <out>/cohort3_bla_cohort2_CM/<session>.h5
    <cohort>/behavior_aligned/<session>.csv   BORIS bouts + respiration times
    <cohort>/manifest.csv                     one row per session: timing QC, warnings, notes

Why frame index: BORIS times are mp4 seconds, and many mp4s are encoded at the
wrong fps (cohort 2: 13-32 fps vs a 15 fps capture). Every BORIS export here
was scored on the raw session mp4 (media duration and FPS match exactly) and
carries `Image index start/stop`, i.e. the mp4 frame, which /video maps to the
respiration clock. See docs/DATA_README.md.

Run (biopipeline-env):  python cagemate_export.py <out_dir>
"""
import os
import re
import sys
import glob
import json
from concurrent.futures import ProcessPoolExecutor

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ekg_resp_to_h5 as E

LAB = r"C:\Users\thoma\UF Dropbox\Thomas Heeps\Padilla-Coreano Lab"
C1 = LAB + r"\2025\ECG_cohort1\Aim1\AIM1\Day1_new\CM interactions"
C2 = LAB + r"\2025\BLA_ChR_resp_pilot1 (SS)"
C3 = LAB + r"\2026\BLA_ChR_resp_cohort2\BLA_ChR_resp_CM_hc"

COHORTS = {
    "cohort1_aim1_CM": {
        "conditions": {"baseline": C1 + r"\Baseline cagemate interactions",
                       "1d_isolation": C1 + r"\1_day_isolation",
                       "10d_isolation": C1 + r"\10_day_isolation"},
        "boris_dir": C1 + r"\cm_boris\baseline csv",
    },
    "cohort2_bla_pilot_CM": {"conditions": {"": C2 + r"\BLA_resp_ChR_cm_hc"},
                             "boris_dir": C2 + r"\BLA_resp_ChR_cm_boris"},
    "cohort3_bla_cohort2_CM": {"conditions": {"": C3}, "boris_dir": C3 + r"\BORIS_csv"},
}

# Documented in notebooks/.../docs/EXCLUSIONS_LOG.md. Exported anyway, flagged here.
NOTES = {
    "CM_s3_6_sub3_5_20250623_174348_merged": "BORIS export empty: excluded (EXCLUSIONS_LOG)",
    "CM_s3_5_d3_6_20250623_170708_merged": "duplicate 3_5 session, no BORIS: excluded (EXCLUSIONS_LOG)",
    "CM_s3_5_d3_6_20250623_160001_merged": "short analog export; dropped from full-recording pipeline (EXCLUSIONS_LOG)",
    "1_1_i_cm_1_2_d_20260106_134350": "inner .rec named 1_1_d_cm_1_2_i (roles swapped vs folder)",
    "8_1_d_cm_8_2_s_20260107_152812": "inner .rec named 8_1_s_cm_8_2_d (roles swapped vs folder)",
    "2_1_d_cm_2.3_i_20260106_105109": "inner .rec truncated to 2_1_d_cm_2.rec",
}
ACTORS = ("subject", "social_agent")


def stamp(name):
    m = re.search(r"\d{8}_\d{6}", name)
    return m.group(0) if m else None


def load_boris(paths):
    """
    Combine the BORIS exports of one session. Cohort 2 splits actors across
    files (_VT_CM = subject, _VT_SA = social_agent), so disjoint files are
    concatenated; overlapping ones (cohort 1 CM_s1_2) keep the _VT export.
    Unreadable / empty files (CM_s4_8_sub4_7_20250623.csv, CM_s3_6) are skipped.
    """
    dfs = []
    for p in sorted(paths):
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if "Behavior" not in df or df.empty:
            continue
        df = df[df["Subject"].isin(ACTORS)].copy()
        df["boris_file"] = os.path.basename(p)
        dfs.append(df)
    if len(dfs) > 1:
        actor_sets = [set(d["Subject"]) for d in dfs]
        if any(a & b for i, a in enumerate(actor_sets) for b in actor_sets[i + 1:]):
            vt = [d for d in dfs if "_VT" in d["boris_file"].iloc[0]]
            dfs = vt[:1] or dfs[:1]
    return pd.concat(dfs, ignore_index=True) if dfs else None


def align_behavior(df, h5_path):
    """Add respiration times (s from /resp[0]) and /resp indices to each bout."""
    with h5py.File(h5_path, "r") as f:
        v = f["video"]
        frame_time = v["frame_time_sec"][:]
        frame_idx = v["frame_to_resp_index"][:]
        fps = float(v.attrs["mp4_fps"])
    n = frame_time.size
    out = df[["boris_file", "Subject", "Behavior", "Behavior type", "Start (s)", "Stop (s)",
              "Duration (s)", "Image index start", "Image index stop"]].copy()
    for edge in ("start", "stop"):
        fr = out[f"Image index {edge}"]
        # Image index is BORIS's mp4 frame; fall back to t * fps if missing.
        fr = fr.fillna((out[f"{edge.capitalize()} (s)"] * fps).round()).astype(int)
        valid = (fr >= 0) & (fr < n)
        frc = fr.clip(0, n - 1).to_numpy()
        t = np.where(valid, frame_time[frc], np.nan)
        i = np.where(valid, frame_idx[frc], -1)
        t[i < 0] = np.nan
        out[f"{edge}_frame"] = fr
        out[f"{edge}_resp_sec"] = t
        out[f"{edge}_resp_idx"] = i
    out["duration_resp_sec"] = out["stop_resp_sec"] - out["start_resp_sec"]
    return out.sort_values("start_resp_sec").reset_index(drop=True)


def convert_one(job):
    rec, out_h5 = job
    if os.path.exists(out_h5):
        return out_h5, None
    data = E.get_data_from_rec_file(rec)
    if data is None:
        return out_h5, "resp channel missing"
    E.export_to_h5(out_h5, data)
    return out_h5, None


def manifest_row(cohort, condition, h5_path, rec, beh):
    name = os.path.basename(h5_path)[:-3]
    with h5py.File(h5_path, "r") as f:
        rm, md = f["resp_metadata"].attrs, f["metadata"].attrs
        v = f["video"].attrs if "video" in f else {}
        row = {
            "cohort": cohort, "condition": condition or "CM", "session": name,
            "rec_file": os.path.relpath(rec, LAB),
            "duration_sec": round(float(rm["duration_sec"]), 2),
            "n_duplicate_samples_removed": int(rm["n_duplicate_samples_removed"]),
            "n_timestamp_gaps": int(rm["n_timestamp_gaps"]),
            "imu_channels": " ".join(sorted(f["imu"].keys())),
            "capture_fps": round(float(v.get("fit_fps", np.nan)), 3),
            "mp4_fps": float(v.get("mp4_fps", np.nan)),
            "mp4_time_scale": round(float(v.get("mp4_time_scale", np.nan)), 4),
            "video_start_minus_resp_start_sec": round(float(v.get("video_start_minus_resp_start_sec", np.nan)), 3),
            "video_end_minus_resp_end_sec": round(float(v.get("video_end_minus_resp_end_sec", np.nan)), 3),
            "n_camera_gaps": int(v.get("n_camera_gaps", 0)),
            "boris_files": "",
            "n_bouts": 0, "n_bouts_outside_resp": 0,
            "max_naive_error_sec": np.nan,
            "notes": NOTES.get(name, ""),
            "warnings": " | ".join(json.loads(md["warnings"])),
        }
    if beh is not None:
        row["boris_files"] = " + ".join(sorted(beh["boris_file"].unique()))
        row["n_bouts"] = len(beh)
        row["n_bouts_outside_resp"] = int(beh["start_resp_sec"].isna().sum())
        # How far off the old approach (BORIS seconds used as resp seconds) would be.
        row["max_naive_error_sec"] = round(float(np.nanmax(np.abs(beh["Start (s)"] - beh["start_resp_sec"]))), 3)
    elif condition in ("", "baseline"):
        row["notes"] = (row["notes"] + "; " if row["notes"] else "") + "no usable BORIS"
    return row


def run(out_dir, workers=3):
    jobs, meta = [], []
    for cohort, cfg in COHORTS.items():
        for condition, root in cfg["conditions"].items():
            dest = os.path.join(out_dir, cohort, condition)
            os.makedirs(dest, exist_ok=True)
            for rec in E.find_merged_rec_files(root):
                out_h5 = os.path.join(dest, E.session_name(rec) + ".h5")
                jobs.append((rec, out_h5))
                meta.append((cohort, condition, rec, out_h5))
    print(f"{len(jobs)} cagemate sessions")
    with ProcessPoolExecutor(workers) as ex:
        for out_h5, err in ex.map(convert_one, jobs):
            print(("FAILED " + err + ": " if err else "ok ") + out_h5, flush=True)

    for cohort, cfg in COHORTS.items():
        boris = {}
        for p in glob.glob(os.path.join(cfg["boris_dir"], "*.csv")):
            boris.setdefault(stamp(os.path.basename(p)), []).append(p)
        beh_dir = os.path.join(out_dir, cohort, "behavior_aligned")
        os.makedirs(beh_dir, exist_ok=True)
        rows = []
        for c, condition, rec, out_h5 in meta:
            if c != cohort or not os.path.exists(out_h5):
                continue
            name = os.path.basename(out_h5)[:-3]
            df = load_boris(boris.get(stamp(name), [])) if condition in ("", "baseline") else None
            beh = align_behavior(df, out_h5) if df is not None else None
            if beh is not None:
                beh.to_csv(os.path.join(beh_dir, name + ".csv"), index=False)
            rows.append(manifest_row(cohort, condition, out_h5, rec, beh))
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, cohort, "manifest.csv"), index=False)
        print(f"{cohort}: manifest with {len(rows)} sessions")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "cagemate_exports")
