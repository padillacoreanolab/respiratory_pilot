"""
BORIS behavior -> respiration clock.

Pipeline:  read_boris -> clean_boris -> combine_boris -> boris_to_frames
           -> frames_to_resp -> (optional) merge_bouts

Evidence behind each rule (all 225 BORIS exports on Dropbox, scanned 2026-11-10):
- BORIS seconds are mp4 seconds. On all 43 exports with a matching H5, the
  BORIS FPS equals the mp4 fps and "Media duration" equals the mp4 duration
  (= (n_frames - 1) / fps), so frame = seconds x FPS is exact and 0-based.
- "Image index" is 0-1 frames later than seconds x FPS (BORIS rounding), and
  Mac exports (valence RI1/RI2) have no Image index at all. So we use
  seconds x FPS for every file (R34 hub: "one frame convention, media
  seconds x fps") and keep Image index only as a check.
- Mac exports were scored on .mov copies whose FPS/length differ from the raw
  mp4 (e.g. 13 fps, 715 s). Those are mapped by fraction of the video and
  flagged "scaled".
- One export is tab-separated with a .csv name; one has " Observation id"
  (leading space); names vary in case ("Posturing"); 3 files use the
  "No focal subject"; 76 bouts are shorter than 1 frame (double key presses);
  cohort 2 splits actors over _VT_CM / _VT_SA files.
Frame -> respiration time goes through the H5's /video group (written by
ekg_resp_to_h5.py), which already handles wrong mp4 fps, the cohort 1 camera
offset, and camera pauses (docs/DATA_README.md section 2).
"""
import os

import numpy as np
import pandas as pd

ACTORS = ("subject", "social_agent")

# Old/Mac BORIS export names -> current names
_RENAME = {"FPS": "FPS (frame/s)", "Media file": "Media file name",
           "Total length": "Media duration (s)"}


def read_boris(path):
    """Read one BORIS export (comma or tab separated). None if empty/unreadable."""
    try:
        # sep=None sniffs the delimiter IOT read tab-separated files saved as .csv
        df = pd.read_csv(path, sep=None, engine="python")
    except Exception:
        return None
    # Strip column names IOT fix exports like " Observation id"
    df.columns = df.columns.str.strip()
    # Rename old/Mac columns IOT give every export the same columns
    df = df.rename(columns=_RENAME)
    if "Behavior" not in df.columns or df.empty:
        return None
    df["boris_file"] = os.path.basename(path)
    return df


def clean_boris(df, actors=ACTORS):
    """Normalize names, keep real actors, drop duplicates and <1-frame bouts."""
    df = df.copy()
    # Lower-case + strip IOT match "Posturing" with "posturing", "subject " with "subject"
    for col in ("Behavior", "Subject"):
        df[col] = df[col].astype(str).str.strip().str.lower()
    # Keep the scored animals IOT drop "No focal subject" rows
    df = df[df["Subject"].isin(actors)]
    # Drop repeated rows IOT avoid double counting re-exported bouts
    df = df.drop_duplicates(subset=["Subject", "Behavior", "Start (s)", "Stop (s)"])
    # Drop bouts shorter than one frame IOT remove accidental double key presses
    fps = pd.to_numeric(df["FPS (frame/s)"], errors="coerce")
    df = df[(df["Stop (s)"] - df["Start (s)"]) >= 1.0 / fps]
    return df.sort_values("Start (s)").reset_index(drop=True)


def combine_boris(dfs):
    """
    One table per session from several cleaned exports (match files on the
    YYYYMMDD_HHMMSS stamp). Files with different actors (cohort 2 _VT_CM +
    _VT_SA) are stacked; files scoring the same actor are re-scores, so only
    one is kept (the _VT one if present).
    """
    dfs = [d for d in dfs if d is not None and not d.empty]
    if not dfs:
        return None
    actor_sets = [set(d["Subject"]) for d in dfs]
    overlap = any(a & b for i, a in enumerate(actor_sets) for b in actor_sets[i + 1:])
    if overlap:
        vt = [d for d in dfs if "_VT" in d["boris_file"].iloc[0]]
        dfs = vt[:1] or dfs[:1]
    return pd.concat(dfs, ignore_index=True).sort_values("Start (s)").reset_index(drop=True)


def boris_to_frames(df, timing):
    """
    Add start_frame / stop_frame (fractional, 0-based mp4 frames).
    `timing` comes from video.load_video_timing(h5_path).
    """
    df = df.copy()
    n, fps = timing["n_frames"], timing["mp4_fps"]
    boris_fps = float(pd.to_numeric(df["FPS (frame/s)"], errors="coerce").iloc[0])
    media_dur = float(df["Media duration (s)"].iloc[0])

    # Same video IOT trust seconds x FPS: BORIS fps and length match the mp4
    same_video = abs(boris_fps - fps) < 0.01 and abs(media_dur - (n - 1) / fps) <= 1.0 / fps
    for edge in ("start", "stop"):
        t = df[f"{edge.capitalize()} (s)"]
        if same_video:
            # seconds x FPS = exact frame IOT handle every export the same way
            df[f"{edge}_frame"] = t * fps
        else:
            # Fraction of the video IOT map Mac/.mov scores whose FPS/length
            # differ from the raw mp4 (assumes the copy spans the whole video)
            df[f"{edge}_frame"] = t / media_dur * (n - 1)
    df["frame_method"] = "exact" if same_video else "scaled"

    # Image index check IOT catch a BORIS file scored on a different video
    if "Image index start" in df.columns:
        off = (df["Image index start"] - df["start_frame"]).abs()
        df["image_index_offset"] = off  # expect 0-1 (BORIS rounding)
    return df


def frames_to_resp(df, timing):
    """Add start/stop_resp_sec: seconds from /resp[0] (NaN outside the recording)."""
    df = df.copy()
    frame_t = np.where(timing["frame_in_resp"], timing["frame_time_sec"], np.nan)
    for edge in ("start", "stop"):
        # Interpolate frame times IOT keep sub-frame precision; frame times
        # already include camera offset, wrong mp4 fps and camera pauses
        df[f"{edge}_resp_sec"] = np.interp(df[f"{edge}_frame"], np.arange(frame_t.size),
                                           frame_t, left=np.nan, right=np.nan)
    df["duration_resp_sec"] = df["stop_resp_sec"] - df["start_resp_sec"]
    return df


def merge_bouts(df, min_gap_s=0.0, min_bout_s=0.0):
    """
    Per subject + behavior: join bouts separated by < min_gap_s, then drop
    bouts shorter than min_bout_s (same rule as ephys_analysis threshold_bouts).
    """
    out = []
    for (subj, beh), g in df.dropna(subset=["start_resp_sec"]).groupby(["Subject", "Behavior"]):
        g = g.sort_values("start_resp_sec")
        bouts = []
        for s, e in zip(g["start_resp_sec"], g["stop_resp_sec"]):
            # Join close bouts IOT treat a brief pause as one continuous bout
            if bouts and s - bouts[-1][1] < min_gap_s:
                bouts[-1][1] = max(bouts[-1][1], e)
            else:
                bouts.append([s, e])
        for s, e in bouts:
            # Drop short bouts IOT keep only bouts long enough to analyze
            if e - s >= min_bout_s:
                out.append({"Subject": subj, "Behavior": beh, "start_resp_sec": s,
                            "stop_resp_sec": e, "duration_resp_sec": e - s})
    return pd.DataFrame(out).sort_values("start_resp_sec").reset_index(drop=True)
