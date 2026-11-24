"""
Load BORIS aggregated-event exports and align bouts to the respiration clock.

Pipeline:
    read_boris -> clean_boris -> combine_boris -> boris_to_frames
    -> frames_to_resp -> merge_bouts (optional)

Frame timing comes from video.load_video_timing(h5_path).
"""
import os

import numpy as np
import pandas as pd

ACTORS = ("subject", "social_agent")

# Old/Mac BORIS column names -> current names
_RENAME = {"FPS": "FPS (frame/s)", "Media file": "Media file name",
           "Total length": "Media duration (s)"}


def read_boris(path):
    """
    Read one BORIS export.

    Args:
        path: BORIS .csv (comma or tab separated)
    Returns:
        DataFrame, or None if the file is empty or not a BORIS table
    """
    try:
        # Sniff the delimiter IOT read tab-separated files saved as .csv
        df = pd.read_csv(path, sep=None, engine="python")
    except Exception:
        return None
    # Strip column names IOT fix names like " Observation id"
    df.columns = df.columns.str.strip()
    # Rename old/Mac columns IOT give every export the same columns
    df = df.rename(columns=_RENAME)
    if "Behavior" not in df.columns or df.empty:
        return None
    df["boris_file"] = os.path.basename(path)
    return df


def clean_boris(df, actors=ACTORS):
    """
    Normalize names and drop rows that are not real bouts.

    Args:
        df: output of read_boris
        actors: subjects to keep
    Returns:
        DataFrame sorted by start time
    """
    df = df.copy()
    # Lower-case + strip IOT match names that differ only in case/spaces
    for col in ("Behavior", "Subject"):
        df[col] = df[col].astype(str).str.strip().str.lower()
    # Keep scored animals IOT drop "No focal subject" rows
    df = df[df["Subject"].isin(actors)]
    # Drop repeated rows IOT avoid double counting
    df = df.drop_duplicates(subset=["Subject", "Behavior", "Start (s)", "Stop (s)"])
    # Drop STATE bouts shorter than one frame IOT remove accidental double key presses
    fps = pd.to_numeric(df["FPS (frame/s)"], errors="coerce")
    is_point = df.get("Behavior type", pd.Series("", index=df.index)).eq("POINT")
    df = df[is_point | ((df["Stop (s)"] - df["Start (s)"]) >= 1.0 / fps)]
    return df.sort_values("Start (s)").reset_index(drop=True)


def combine_boris(dfs):
    """
    Merge the cleaned exports of one session into one table.

    Exports with different actors are stacked; exports scoring the same actor
    are treated as re-scores and only one is kept (the "_VT" file if present).

    Args:
        dfs: list of clean_boris outputs for one session
    Returns:
        DataFrame, or None if nothing usable
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
    Convert BORIS seconds to 0-based mp4 frames (fractional).

    Args:
        df: combine_boris output
        timing: video.load_video_timing output
    Returns:
        DataFrame with start_frame, stop_frame, frame_method ("exact"/"scaled")
        and image_index_offset (if BORIS exported Image index)
    """
    df = df.copy()
    n, fps = timing["n_frames"], timing["mp4_fps"]
    boris_fps = float(pd.to_numeric(df["FPS (frame/s)"], errors="coerce").iloc[0])
    media_dur = float(df["Media duration (s)"].iloc[0])

    # Check the scored video is the mp4 IOT know if seconds x FPS is valid
    same_video = abs(boris_fps - fps) < 0.01 and abs(media_dur - (n - 1) / fps) <= 1.0 / fps
    for edge in ("start", "stop"):
        t = df[f"{edge.capitalize()} (s)"]
        if same_video:
            # seconds x FPS IOT get the exact mp4 frame
            frame = t * fps
        else:
            # Fraction of the video IOT map scores made on a re-encoded copy (e.g. Mac .mov)
            frame = t / media_dur * (n - 1)
        # Clamp up to 1 frame past the end IOT keep bouts that stop on the last frame
        df[f"{edge}_frame"] = np.where((frame > n - 1) & (frame <= n), n - 1, frame)
    df["frame_method"] = "exact" if same_video else "scaled"

    # Image index offset IOT flag a file scored on a different video (expect 0-2)
    if "Image index start" in df.columns:
        df["image_index_offset"] = (df["Image index start"] - df["start_frame"]).abs()
    return df


def frames_to_resp(df, timing):
    """
    Convert mp4 frames to seconds from /resp[0].

    Args:
        df: boris_to_frames output
        timing: video.load_video_timing output
    Returns:
        DataFrame with start_resp_sec, stop_resp_sec, duration_resp_sec
        (NaN where the frame is outside the respiration recording)
    """
    df = df.copy()
    frame_t = np.where(timing["frame_in_resp"], timing["frame_time_sec"], np.nan)
    for edge in ("start", "stop"):
        # Interpolate frame times IOT keep sub-frame precision
        df[f"{edge}_resp_sec"] = np.interp(df[f"{edge}_frame"], np.arange(frame_t.size),
                                           frame_t, left=np.nan, right=np.nan)
    df["duration_resp_sec"] = df["stop_resp_sec"] - df["start_resp_sec"]
    return df


def merge_bouts(df, min_gap_s=0.0, min_bout_s=0.0, by_behavior=True):
    """
    Join close bouts, then drop short ones (lab threshold_bouts rule).

    Args:
        df: frames_to_resp output
        min_gap_s: bouts separated by less than this are joined
        min_bout_s: bouts shorter than this (after joining) are dropped
        by_behavior: True = per subject + behavior; False = per subject,
            treating all behaviors in df as one (filter df first)
    Returns:
        DataFrame of Subject, Behavior, start/stop/duration_resp_sec
    """
    keys = ["Subject", "Behavior"] if by_behavior else ["Subject"]
    out = []
    for key, g in df.dropna(subset=["start_resp_sec"]).groupby(keys):
        subj = key[0] if isinstance(key, tuple) else key
        beh = key[1] if by_behavior else "+".join(sorted(g["Behavior"].unique()))
        bouts = []
        for s, e in sorted(zip(g["start_resp_sec"], g["stop_resp_sec"])):
            # Join close bouts IOT treat a brief pause as one bout
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
