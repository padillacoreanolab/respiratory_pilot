import os
import re
import glob
import json
import shutil
import subprocess
from datetime import date

import h5py
import numpy as np


"""
===========================================================
Trodes .rec → HDF5 Conversion Script  (canonical version)
===========================================================

This script converts SpikeGadgets/Trodes `.rec` session folders and
their exported `.analog` folders into HDF5 files for analysis. It is the
one to use for respiration + IMU + video alignment in every cohort (an
older version produced the Aug-2025 Aim1 cagemate h5s). For all cagemate
sessions with aligned BORIS, see cagemate_export.py; per-cohort notes are
in docs/DATA_README.md.

Prerequisites
-------------
1. Python environment with: numpy, h5py
   (optional) ffprobe on PATH (ships with conda's ffmpeg) → used to read
   the mp4's frame count/fps as a sanity check. Skipped if missing.

2. Each session folder must already contain a Trodes analog export
   (`<name>_merged.analog/`). Create it with the Trodes GUI export or:
       trodesexport -analogio -rec "<path>\\<name>_merged.rec"
   This script does NOT run trodesexport itself.

Usage
-----
- From terminal:
    1. Edit `data_root` and `output_dir` at the bottom of this file.
    2. Run: `python ekg_resp_to_h5.py`
- From Jupyter:
    from ekg_resp_to_h5 import convert_all_rec_to_h5
    convert_all_rec_to_h5(data_root, output_dir="...")

Output HDF5 layout
------------------
/resp                         float32 (N,)  ECU_Ain1 respiration, raw ADC counts, 20 kHz
/resp_metadata   (attrs)      sampling_frequency, num_samples, duration_sec, units, ...
/timestamps                   uint32  (N,)  Trodes clock tick of every /resp and /imu sample
/imu/accel_x ... gyro_z       int16   (N,)  headstage IMU, raw counts, same samples as /resp
                                            (sample-and-hold: values only update ~100 Hz)
/video/frame_timestamps       uint32  (F,)  Trodes tick when each mp4 frame arrived (raw)
/video/frame_timestamps_fit   float64 (F,)  linear fit of the above (removes USB jitter)
/video/frame_time_sec         float64 (F,)  fitted frame time, seconds relative to /resp[0]
/video/frame_to_resp_index    int64   (F,)  index into /resp for each frame (-1 = outside)
/video/camera_hwsync          compound      raw .cameraHWSync records (if present)
/metadata        (attrs)      session info, clock info, wall-clock start, QC counts
/ekg_metadata    (group)      empty; kept so older readers that open it don't break
BLA-compatible copies (same samples, for notebooks written for the BLA format):
/analog/<name>.analog_ECU_Ain1  structured int16 'voltage', Trodes header as attrs
/analog/<name>.timestamps       uint32, same as /timestamps
/resp_clean/signal, /time       100 Hz cleaned resp (needs neurokit2 → biopipeline-env)
/dio/<name>.dio_*               digital I/O state changes (only if a .DIO export exists)

/imu is empty for tethered sessions with no headstage (e.g. BLA cohort 2).

Time conventions (IMPORTANT for alignment)
------------------------------------------
- Everything shares one clock: the Trodes timestamp (integer ticks, 20000/s).
- resp time in seconds:   (timestamps - timestamps[0]) / clockrate
  After duplicate removal (see FIX notes) the timestamps are contiguous, so
  this equals  np.arange(N) / sampling_frequency  exactly.
- BORIS / SLEAP / any tool that works in "video time" uses mp4 time,
  i.e. frame_index / mp4_fps. To get the matching respiration sample:
      frame = int(round(t_video * video.attrs['mp4_fps']))
      resp_idx = f['video/frame_to_resp_index'][frame]
  or use `video_time_to_resp_time()` below. Do NOT use t_video * 20000
  directly: the camera usually starts up to ~1 s before the analog data,
  and some mp4s are encoded at 23/24/29 fps although the camera captured
  30 fps, so their mp4 seconds are not real seconds (see
  video.attrs['mp4_time_scale']).

Changelog
---------
2026-09-30  Fixes and additions (each marked with `# FIX (2026-09-30)` or
            `# NEW (2026-09-30)` at the place it happened):
  - Drop ~27k merge-filler samples per session that repeat a timestamp.
  - Sampling rate now read from the file's `Clockrate` header.
  - `units` corrected from 'volts' to 'adc_counts'.
  - `start_time` now filled from `First_timestamp`.
  - Removed SpikeInterface dependency (was only used for the sampling rate,
    and a failure there silently skipped the whole session).
  - parse_fields: no deprecated NumPy (type, 1) shape; regex escape fixed.
  - Only the .analog folder is read (previously every export folder,
    including large .raw neural exports, was loaded then thrown away).
  - Missing resp channel is reported and the session is skipped instead of
    writing an h5 with no /resp.
  - Added per-sample timestamps, IMU channels, video frame timing, QC metadata.
  - Removed dead commented-out code (filename-metadata parser, EKG block).
2026-10-01  Plain (non-merged) .rec sessions supported; BLA-compatible
            /analog, /resp_clean and /dio groups added (`# FIX/NEW (2026-10-01)`).
2026-10-24  Video frame times fitted per unbroken run (camera pauses), corrupt
            frame timestamps flagged, outputs named after the session folder
            (`# FIX (2026-10-24)`). See docs/DATA_README.md for per-cohort notes.
"""

# NEW (2026-09-30): channel names as they appear in the .analog export
# (file suffix after "analog_" and the `ID:` header line).
RESP_CHANNEL = "ECU_Ain1"
IMU_CHANNELS = {
    "accel_x": "Headstage_AccelX",
    "accel_y": "Headstage_AccelY",
    "accel_z": "Headstage_AccelZ",
    "gyro_x": "Headstage_GyroX",
    "gyro_y": "Headstage_GyroY",
    "gyro_z": "Headstage_GyroZ",
    "mag_x": "Headstage_MagX",
    "mag_y": "Headstage_MagY",
    "mag_z": "Headstage_MagZ",
}
DEFAULT_CLOCKRATE = 20000.0
SCRIPT_VERSION = "2026-10-24"


def find_merged_rec_files(datapath):
    """
    One .rec per session folder.

    FIX (2026-10-01): previously only `*merged.rec` was found, so tethered
    recordings (e.g. BLA cohort 2, which have a single plain .rec and no
    merge step) were silently skipped. Now: use merged.rec if the folder has
    one (wireless sessions), otherwise the plain .rec.
    """
    rec_files = []
    for root, dirs, files in os.walk(datapath):
        recs = [f for f in files if f.endswith('.rec')]
        merged = [f for f in recs if f.endswith('merged.rec')]
        for f in (merged or recs):
            rec_files.append(os.path.join(root, f))
    return sorted(rec_files)


def session_name(rec_file):
    """
    FIX (2026-10-24): name sessions after the session folder (`<name>.rec/`),
    not the .rec file inside it. In the BLA pilot the inner file name differs
    from the folder (`2_1_d_cm_2.3_i_..._105109.rec/2_1_d_cm_2.rec`; d/s or
    i/d roles swapped in 1_1_i_cm_1_2_d and 8_1_d_cm_8_2_s), while the folder
    name is what the older h5s and the BORIS files use. Merged sessions keep
    their `_merged` suffix so Aim1 names are unchanged.
    """
    folder = os.path.basename(os.path.dirname(rec_file))
    inner = os.path.splitext(os.path.basename(rec_file))[0]
    if not folder.endswith(".rec"):
        return inner
    name = folder[:-len(".rec")]
    if inner.endswith("_merged") and not name.endswith("_merged"):
        name += "_merged"
    return name


def read_trodes_extracted_data_file(filename, default_dtype=None):
    """
    Read a Trodes-exported binary file: ASCII header between
    <Start settings> / <End settings>, then packed binary records
    described by the header's `Fields:` line.

    Returns dict of lowercased header keys plus 'data' (NumPy array).
    """
    with open(filename, "rb") as f:
        if f.readline().decode("ascii").strip() != "<Start settings>":
            raise Exception("Settings format not supported")

        fields_text = {}
        for line in f:
            line = line.decode("ascii", errors="replace").strip()
            if line == "<End settings>":
                break
            # FIX (2026-09-30): split on the first ": " only. Values such as
            # the camera name contain ": " / "\\?\" and made split(": ") raise.
            key, value = line.split(": ", 1)
            fields_text[key.lower()] = value

        # FIX (2026-09-30): `.videoTimeStamps` files have no `Fields:` line
        # (records are bare uint32), so allow the caller to supply a dtype.
        if "fields" in fields_text:
            dt = parse_fields(fields_text["fields"])
        elif default_dtype is not None:
            dt = np.dtype(default_dtype)
        else:
            raise Exception("No 'Fields' line in header and no default dtype given")

        data = np.fromfile(f, dt)
        fields_text["data"] = data
        return fields_text


def parse_fields(field_str):
    """'<voltage int16>' → np.dtype([('voltage', int16)])"""
    # FIX (2026-09-30): raw-string regex (old "\s" raised an invalid-escape
    # warning) and single-count fields are now plain scalars. The old
    # (name, type, 1) form is deprecated in NumPy and produced a (N, 1)
    # array, which is why callers needed `.ravel()`.
    components = re.split(r"\s+", re.sub(r"[<>]", " ", field_str).strip())
    dtype_spec = []

    for i in range(0, len(components), 2):
        field_name = components[i]
        repeat_count = 1
        field_type_str = components[i + 1]

        if "*" in field_type_str:
            parts = field_type_str.split("*")
            if parts[0].isdigit():
                repeat_count = int(parts[0])
                field_type_str = parts[1]
            else:
                repeat_count = int(parts[1])
                field_type_str = parts[0]

        np_type = getattr(np, field_type_str)
        if repeat_count == 1:
            dtype_spec.append((field_name, np_type))
        else:
            dtype_spec.append((field_name, np_type, (repeat_count,)))

    return np.dtype(dtype_spec)


def load_analog_export(session_dir):
    """
    Read the `<name>.analog/` export folder of one session.

    FIX (2026-09-30): replaces organize_all_trodes_export /
    organize_single_trodes_export, which loaded *every* export folder
    (including multi-GB .raw neural exports) and keyed channels by a
    filename split. Now only .analog is read and channels are keyed by
    their name after "analog_" (e.g. 'ECU_Ain1', 'Headstage_AccelX').

    Returns (channels: {name: header_dict_with_data}, timestamps_header_dict)
    """
    analog_dirs = [d for d in glob.glob(os.path.join(session_dir, "*.analog"))
                   if os.path.isdir(d)]
    if not analog_dirs:
        raise FileNotFoundError(
            f"No .analog export folder in {session_dir}. "
            f"Run: trodesexport -analogio -rec <merged.rec>")
    if len(analog_dirs) > 1:
        # Prefer the export of the merged file if several exist
        merged = [d for d in analog_dirs if "merged" in os.path.basename(d)]
        analog_dirs = merged or analog_dirs
    analog_dir = analog_dirs[0]

    channels, timestamps = {}, None
    for file_name in sorted(os.listdir(analog_dir)):
        if not file_name.endswith(".dat"):
            continue
        path = os.path.join(analog_dir, file_name)
        if file_name.endswith("timestamps.dat"):
            timestamps = read_trodes_extracted_data_file(path)
            continue
        m = re.search(r"analog_(.+)\.dat$", file_name)
        if not m:
            continue
        try:
            channels[m.group(1)] = read_trodes_extracted_data_file(path)
            channels[m.group(1)]["file_stem"] = file_name[:-len(".dat")]
        except Exception as e:
            print(f"  Skipping {file_name}: {e}")

    if timestamps is None:
        raise FileNotFoundError(f"No timestamps.dat in {analog_dir}")
    return channels, timestamps, analog_dir


def load_dio_export(session_dir):
    """
    NEW (2026-10-01): digital I/O export (`trodesexport -dio`), e.g. laser
    TTLs on ECU_Din1 for ChR sessions. Each record is (time uint32, state
    uint8): the Trodes tick at which the line changed state. Returns {} if
    the session has no .DIO export.
    """
    dio = {}
    for d in glob.glob(os.path.join(session_dir, "*.[dD][iI][oO]")):
        if not os.path.isdir(d):
            continue
        for file_name in sorted(os.listdir(d)):
            if file_name.endswith(".dat"):
                try:
                    dio[file_name[:-len(".dat")]] = read_trodes_extracted_data_file(
                        os.path.join(d, file_name))
                except Exception as e:
                    print(f"  Skipping {file_name}: {e}")
    return dio


def clean_resp(raw, fs, target_rate=100):
    """
    NEW (2026-10-01): same lowpass → downsample → 0.1-20 Hz bandpass as
    trodes_ECU_to_h5_clean_raw.py, so BLA notebooks that read
    /resp_clean/signal and /resp_clean/time keep working.
    Returns (signal, time) or (None, None) if neurokit2 is unavailable.
    """
    try:
        import neurokit2 as nk
        from scipy.signal import butter, filtfilt, resample_poly
    except ImportError:
        return None, None
    b, a = butter(N=4, Wn=(target_rate / 2) / (fs / 2), btype='low')
    filtered = filtfilt(b, a, np.asarray(raw, dtype=float))
    downsampled = resample_poly(filtered, up=1, down=int(fs // target_rate))
    sig = nk.signal_filter(downsampled, lowcut=0.1, highcut=20, method="butterworth",
                           sampling_rate=target_rate, order=2)
    return sig, np.arange(len(sig)) / target_rate


def read_timestamp_offset(session_dir):
    # NEW (2026-09-30): offset Trodes applied when merging logger + MCU data.
    txts = glob.glob(os.path.join(session_dir, "*.timestampoffset", "*.txt"))
    if not txts:
        return np.nan
    try:
        with open(txts[0]) as fh:
            return float(fh.read().strip())
    except ValueError:
        return np.nan


def ffprobe_video_info(mp4_path):
    """
    NEW (2026-09-30): frame count and nominal fps of the mp4 via ffprobe.
    BORIS/SLEAP times are mp4 times, so these must match the Trodes frame
    timestamps. Returns {} if ffprobe is not available.
    """
    if not mp4_path or shutil.which("ffprobe") is None:
        return {}
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=nb_frames,r_frame_rate,avg_frame_rate,duration",
           "-of", "json", mp4_path]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        s = json.loads(out.stdout)["streams"][0]
        num, den = s["r_frame_rate"].split("/")
        return {
            "mp4_n_frames": int(s.get("nb_frames", -1)),
            "mp4_fps": float(num) / float(den),
            "mp4_duration_sec": float(s.get("duration", np.nan)),
        }
    except Exception as e:
        print(f"  ffprobe failed on {mp4_path}: {e}")
        return {}


def find_videos(session_dir):
    """
    NEW (2026-09-30): pair each camera's .videoTimeStamps with its mp4.
    Trodes names them <name>.<camera#>.videoTimeStamps / <name>.<camera#>.mp4
    (older sessions: <name>.<camera#>_converted.mp4 made from the .h264).
    """
    videos = []
    for ts_path in sorted(glob.glob(os.path.join(session_dir, "*.videoTimeStamps"))):
        stem = ts_path[: -len(".videoTimeStamps")]           # ...\<name>.1
        cam = stem.rsplit(".", 1)[-1]
        mp4 = next((p for p in (stem + ".mp4", stem + "_converted.mp4") if os.path.exists(p)), None)
        hw = ts_path + ".cameraHWSync"
        videos.append({
            "camera": cam,
            "timestamps_path": ts_path,
            "hwsync_path": hw if os.path.exists(hw) else None,
            "mp4_path": mp4,
        })
    return videos


def build_video_group_data(video, resp_timestamps, clockrate):
    """
    NEW (2026-09-30): per-frame Trodes time for one camera.

    Trodes stamps each frame when it *arrives* over USB, so raw frame
    timestamps jitter by about ±1 frame (~±35 ms) and often come in pairs
    with identical values. The webcam itself runs at a constant rate, so a
    straight-line fit of timestamp vs frame index gives cleaner frame times.
    Both raw and fitted values are saved; the fit is used for mapping.
    """
    vts = read_trodes_extracted_data_file(video["timestamps_path"], default_dtype=np.uint32)
    frame_ts = np.asarray(vts["data"]).astype(np.uint32)
    n_frames = frame_ts.size

    # FIX (2026-10-24): fit each unbroken run of frames separately. The camera
    # sometimes pauses mid-session (~70 s in BLA pilot 4_3_RI2_f2_e_on and
    # 1_1_oft); one straight line across the pause misplaced frames by up to
    # half the pause. Non-monotonic (corrupt) timestamps get no mapping at all.
    idx = np.arange(n_frames)
    d = np.diff(frame_ts.astype(np.int64))
    n_backwards = int((d < 0).sum())
    breaks = np.flatnonzero(d > 5 * np.median(d)) + 1
    fit_ts = frame_ts.astype(np.float64)
    seg_slopes = []
    for seg in np.split(idx, breaks):
        if seg.size >= 2:
            s, c = np.polyfit(seg, frame_ts[seg].astype(np.float64), 1)
            fit_ts[seg] = s * seg + c
            seg_slopes.append((seg.size, s))
    slope = max(seg_slopes)[1]          # capture rate from the longest run
    gap_sec = (d[breaks - 1] - slope) / clockrate
    if n_backwards:
        fit_ts[:] = np.nan
    resid_ms = (frame_ts - fit_ts) / clockrate * 1000.0

    resp_t0 = int(resp_timestamps[0])
    frame_time_sec = (fit_ts - resp_t0) / clockrate

    # Nearest resp sample for each frame; -1 when the frame is outside the
    # respiration recording (camera often starts ~0-1 s before analog data).
    ri = np.searchsorted(resp_timestamps, np.nan_to_num(fit_ts)).astype(np.int64)
    ri = np.clip(ri, 0, resp_timestamps.size - 1)
    outside = ~((fit_ts >= resp_timestamps[0]) & (fit_ts <= resp_timestamps[-1]))
    ri[outside] = -1

    attrs = {
        "camera": video["camera"],
        "camera_name": vts.get("camera_name", ""),
        "clockrate": float(vts.get("clock rate", clockrate)),
        "n_frames": int(n_frames),
        "timestamps_file": os.path.basename(video["timestamps_path"]),
        "mp4_file": os.path.basename(video["mp4_path"]) if video["mp4_path"] else "",
        "fit_fps": float(clockrate / slope),
        "fit_residual_ms_p99_abs": float(np.nanpercentile(np.abs(resid_ms), 99)) if not n_backwards else np.nan,
        "fit_residual_ms_max_abs": float(np.nanmax(np.abs(resid_ms))) if not n_backwards else np.nan,
        "n_camera_gaps": int(breaks.size),
        "camera_gaps_sec": gap_sec.astype(np.float64),
        "n_backwards_frame_timestamps": n_backwards,
        "n_frames_outside_resp": int(outside.sum()),
        "video_start_minus_resp_start_sec": float((fit_ts[0] - resp_t0) / clockrate),
        "video_end_minus_resp_end_sec": float((fit_ts[-1] - int(resp_timestamps[-1])) / clockrate),
        "mapping_note": "BORIS/SLEAP time t (s) -> frame = round(t * mp4_fps) -> "
                        "frame_to_resp_index[frame]",
    }
    info = ffprobe_video_info(video["mp4_path"])
    attrs.update(info)

    warnings = []
    if n_backwards:
        warnings.append(f"{n_backwards} backwards frame timestamps (corrupt .videoTimeStamps): "
                        f"frame_to_resp_index is all -1, do not align this video")
    if breaks.size:
        warnings.append(f"camera paused {breaks.size}x (longest {gap_sec.max():.1f} s): "
                        f"mp4/BORIS seconds after a pause are wrong. Map via frame index.")
    if "mp4_fps" not in attrs:
        # Some mp4s are encoded at 23/24/29 fps even though the camera captured
        # 30 fps, so never guess silently: fall back to 30 but flag it.
        attrs["mp4_fps"] = 30.0
        attrs["mp4_fps_source"] = "assumed (ffprobe unavailable) - VERIFY"
        warnings.append("ffprobe unavailable: mp4_fps assumed 30; verify before using video times")
    else:
        attrs["mp4_fps_source"] = "ffprobe"
    if info.get("mp4_n_frames", n_frames) not in (n_frames, -1):
        warnings.append(f"mp4 has {info['mp4_n_frames']} frames but "
                        f"videoTimeStamps has {n_frames}: frame→time mapping unreliable")
    # The mp4 plays at its *encoded* rate, but frames were captured at fit_fps.
    # When they differ, BORIS/mp4 seconds are not real seconds (e.g. 23 fps
    # encode of a 30 fps capture plays 1.3x too slow). Mapping through the
    # frame index (frame_to_resp_index / video_time_to_resp_time) is still
    # correct; using BORIS seconds directly as resp seconds is not.
    attrs["mp4_time_scale"] = float(attrs["mp4_fps"] / attrs["fit_fps"])
    if abs(attrs["mp4_time_scale"] - 1) > 0.001:
        warnings.append(f"mp4 encoded at {attrs['mp4_fps']:.3f} fps but camera captured "
                        f"{attrs['fit_fps']:.3f} fps: 1 s of mp4/BORIS time = "
                        f"{1 / attrs['mp4_time_scale']:.3f} s real time. Map via frame index.")
    if attrs["video_end_minus_resp_end_sec"] > 5:
        warnings.append(f"video runs {attrs['video_end_minus_resp_end_sec']:.1f} s past the end "
                        f"of the respiration data (short/truncated analog export?)")

    datasets = {
        "frame_timestamps": frame_ts,
        "frame_timestamps_fit": fit_ts,
        "frame_time_sec": frame_time_sec,
        "frame_time_sec_raw": (frame_ts.astype(np.float64) - resp_t0) / clockrate,
        "frame_to_resp_index": ri,
    }
    if video["hwsync_path"]:
        try:
            datasets["camera_hwsync"] = read_trodes_extracted_data_file(video["hwsync_path"])["data"]
        except Exception as e:
            print(f"  Could not read cameraHWSync: {e}")
    return datasets, attrs, warnings


def get_data_from_rec_file(rec_file, resp_channel=RESP_CHANNEL, drop_duplicate_timestamps=True):
    """
    Load respiration, IMU and timestamps for one session.

    Returns dict with keys: resp, resp_meta, timestamps, imu, imu_meta,
    session_meta, videos (list), warnings — or None if resp is missing.
    """
    # FIX (2026-09-30): no longer opens the .rec with SpikeInterface. It was
    # only used to get the sampling rate (which is in the export header as
    # `Clockrate`), and if it failed the whole session was skipped.
    session_dir = os.path.dirname(rec_file)
    channels, ts_hdr, analog_dir = load_analog_export(session_dir)
    warnings = []

    # FIX (2026-09-30): report and skip cleanly when the resp channel is
    # missing (e.g. older recordings with only Controller_Ain1) instead of
    # writing an h5 with no /resp dataset.
    if resp_channel not in channels:
        print(f"  {resp_channel} not found in {analog_dir}. "
              f"Available: {sorted(channels)}")
        return None

    resp_hdr = channels[resp_channel]
    timestamps = np.asarray(ts_hdr["data"]["time"]).astype(np.uint32)
    resp_raw = np.asarray(resp_hdr["data"]["voltage"])
    if resp_raw.size != timestamps.size:
        raise ValueError(f"{resp_channel} has {resp_raw.size} samples but "
                         f"timestamps.dat has {timestamps.size}")

    # FIX (2026-09-30): sampling rate comes from the header's `Clockrate`.
    # The old code looked up 'sample_rate', which never exists in Trodes
    # exports, so it always fell back to SpikeInterface / a hard-coded 20 kHz.
    clockrate = float(resp_hdr.get("clockrate", DEFAULT_CLOCKRATE))

    # FIX (2026-09-30): drop merge-filler samples. In merged .rec exports,
    # ~4-5 samples every ~2000 (≈27k per 10-min session) repeat the previous
    # timestamp AND the previous value exactly — they are padding, not data.
    # Keeping them stretched the signal by ~0.22%, so `index / 20000` drifted
    # ~1.3 s behind true Trodes time by the end of a session.
    d = np.diff(timestamps.astype(np.int64))
    n_backwards = int((d < 0).sum())
    n_gaps = int((d > 1).sum())
    n_missing = int((d[d > 1] - 1).sum())
    keep = np.ones(timestamps.size, dtype=bool)
    n_dupes = 0
    if drop_duplicate_timestamps:
        keep[1:] = d != 0
        n_dupes = int((~keep).sum())
    if n_backwards:
        warnings.append(f"{n_backwards} backwards timestamp jumps")
    if n_gaps:
        warnings.append(f"{n_gaps} timestamp gaps ({n_missing} samples missing): "
                        f"use /timestamps, not index/fs, for time")

    timestamps = timestamps[keep]
    resp_signal = resp_raw[keep].astype(np.float32)
    first_ts = int(timestamps[0])

    resp_meta = {
        'sampling_frequency': clockrate,
        # FIX (2026-09-30): was always NaN. Now the Trodes time (s) of the
        # first sample; the raw tick value is in `first_timestamp`.
        'start_time': first_ts / clockrate,
        'first_timestamp': first_ts,
        'num_samples': int(resp_signal.size),
        'duration_sec': resp_signal.size / clockrate,
        'channel': f"analog_{resp_channel}",
        'byte_order': resp_hdr.get('byte_order', 'unknown'),
        # FIX (2026-09-30): was 'volts'. Header says `Fields: <voltage int16>`,
        # i.e. raw integer ADC counts. Fine for rate/shape; don't report volts.
        'units': 'adc_counts',
        'n_duplicate_samples_removed': n_dupes,
        'n_timestamp_gaps': n_gaps,
        'source_file': os.path.basename(analog_dir),
    }

    # NEW (2026-09-30): headstage IMU. Exported at the full 20 kHz clock but
    # sample-and-hold: the sensor only updates about every 200 samples
    # (~100 Hz). Magnetometer is off in these recordings (header
    # `Magnetometer: 0`, data all zeros), so all-zero channels are skipped.
    imu, imu_meta = {}, {}
    for short, ch in IMU_CHANNELS.items():
        if ch not in channels:
            continue
        x = np.asarray(channels[ch]["data"]["voltage"])
        if x.size != keep.size:
            warnings.append(f"{ch} length {x.size} != timestamps {keep.size}; skipped")
            continue
        x = x[keep]
        if not np.any(x):
            imu_meta[f"{short}_skipped"] = "all zeros (sensor disabled)"
            continue
        imu[short] = x.astype(np.int16)
        changes = np.flatnonzero(np.diff(x) != 0)
        if changes.size > 2:
            imu_meta[f"{short}_update_rate_hz"] = float(clockrate / np.median(np.diff(changes)))
    imu_meta.update({
        'units': 'raw sensor counts (int16), not converted to g or deg/s',
        'sampling_frequency': clockrate,
        'note': 'sample-and-hold at the 20 kHz clock; shares /timestamps with /resp',
        'accelerometer_enabled': resp_hdr.get('accelerometer', ''),
        'gyro_enabled': resp_hdr.get('gyro', ''),
        'magnetometer_enabled': resp_hdr.get('magnetometer', ''),
    })

    # NEW (2026-09-30): clock + wall-clock info needed to align other systems.
    sys_ms = float(resp_hdr.get("system_time_at_creation", np.nan))
    ts_at_creation = float(resp_hdr.get("timestamp_at_creation", np.nan))
    session_meta = {
        'session_name': session_name(rec_file),
        'rec_file': rec_file,
        'clockrate': clockrate,
        'first_timestamp': first_ts,
        'last_timestamp': int(timestamps[-1]),
        'timestamp_offset': read_timestamp_offset(session_dir),
        'system_time_at_creation_ms': sys_ms,
        'timestamp_at_creation': ts_at_creation,
        # Unix time (ms) of /resp[0]: creation wall-clock + ticks elapsed since.
        'resp_start_unix_ms': sys_ms + (first_ts - ts_at_creation) / clockrate * 1000.0,
        'trodes_version': resp_hdr.get('trodes_version', ''),
        'controller_serialnum': resp_hdr.get('controller_serialnum', ''),
        'headstage_serialnum': resp_hdr.get('headstage_serialnum', ''),
        'conversion_script': os.path.basename(__file__),
        'conversion_script_version': SCRIPT_VERSION,
        'conversion_date': date.today().isoformat(),
    }

    return {
        "resp": resp_signal, "resp_meta": resp_meta, "timestamps": timestamps,
        "imu": imu, "imu_meta": imu_meta, "session_meta": session_meta,
        "videos": find_videos(session_dir), "clockrate": clockrate,
        "warnings": warnings,
        # NEW (2026-10-01): for the BLA-compatible groups in export_to_h5
        "resp_raw_int16": resp_raw[keep], "resp_header": resp_hdr,
        "timestamps_header": ts_hdr, "dio": load_dio_export(session_dir),
    }


def _set_attrs(obj, d):
    for k, v in d.items():
        obj.attrs[k] = v


def export_to_h5(output_path, data):
    """Write one session (dict from get_data_from_rec_file) to HDF5."""
    comp = dict(compression="gzip", compression_opts=4, shuffle=True)
    warnings = list(data["warnings"])
    with h5py.File(output_path, 'w') as f:
        # Unchanged location/name so existing readers (resp_helper_functions
        # `_read_raw_resp_from_h5`) keep working.
        f.create_dataset('resp', data=data["resp"])
        _set_attrs(f.create_group('resp_metadata'), data["resp_meta"])

        # NEW (2026-09-30)
        ts = f.create_dataset('timestamps', data=data["timestamps"], **comp)
        ts.attrs['clockrate'] = data["clockrate"]
        ts.attrs['note'] = 'Trodes tick of each /resp and /imu sample; seconds = (t - t[0]) / clockrate'

        imu_grp = f.create_group('imu')
        for k, v in data["imu"].items():
            imu_grp.create_dataset(k, data=v, **comp)
        _set_attrs(imu_grp, data["imu_meta"])

        for i, video in enumerate(data["videos"]):
            try:
                dsets, attrs, vwarn = build_video_group_data(
                    video, data["timestamps"], data["clockrate"])
            except Exception as e:
                warnings.append(f"camera {video['camera']}: could not read video timestamps: {e}")
                continue
            name = 'video' if i == 0 else f"video_camera_{video['camera']}"
            vg = f.create_group(name)
            for k, v in dsets.items():
                vg.create_dataset(k, data=v)
            _set_attrs(vg, attrs)
            warnings.extend(f"camera {video['camera']}: {w}" for w in vwarn)
        if not data["videos"]:
            warnings.append("no .videoTimeStamps found; no video alignment data")

        # NEW (2026-10-01): BLA-compatible layout (same as resp_to_h5_ecu_only.py /
        # trodes_ECU_to_h5_clean_raw.py) so notebooks that read
        # f['analog'][...ECU_Ain1]['voltage'] or f['resp_clean'] work unchanged.
        # Same samples as /resp (duplicates removed), stored as the original
        # structured int16 'voltage' records with the Trodes header as attrs.
        hdr_attrs = lambda h: {k: v for k, v in h.items() if k not in ("data", "file_stem")}
        ag = f.create_group('analog')
        rh = data["resp_header"]
        rec = np.empty(data["resp_raw_int16"].size, dtype=[('voltage', np.int16)])
        rec['voltage'] = data["resp_raw_int16"]
        _set_attrs(ag.create_dataset(rh["file_stem"], data=rec, **comp), hdr_attrs(rh))
        ts_name = rh["file_stem"].split(".analog_")[0] + ".timestamps"
        _set_attrs(ag.create_dataset(ts_name, data=f['timestamps'][:], **comp),
                   hdr_attrs(data["timestamps_header"]))

        sig, t = clean_resp(data["resp"], data["clockrate"])
        if sig is not None:
            rc = f.create_group('resp_clean')
            rc.create_dataset('signal', data=sig)
            rc.create_dataset('time', data=t)
            _set_attrs(rc, {"sampling_rate": 100, "original_fs": data["clockrate"],
                            "source_channel": rh["file_stem"],
                            "note": "time 0 = /resp[0]; built from duplicate-free samples"})
        else:
            warnings.append("neurokit2 not installed: /resp_clean not written "
                            "(run in biopipeline-env)")

        if data["dio"]:
            dg = f.create_group('dio')
            for name, d in data["dio"].items():
                _set_attrs(dg.create_dataset(name, data=d["data"]), hdr_attrs(d))

        meta = dict(data["session_meta"])
        meta['warnings'] = json.dumps(warnings)
        _set_attrs(f.create_group('metadata'), meta)

        # Kept (empty) for backward compatibility: testing_rec.py opens it.
        f.create_group('ekg_metadata')
    return warnings


def video_time_to_resp_time(h5_path, t_video_sec, camera_group="video"):
    """
    NEW (2026-09-30): convert BORIS/SLEAP video times (seconds from the start
    of the mp4) to respiration times (seconds from /resp[0], i.e. the same
    axis as np.arange(len(resp)) / fs).

    Works on scalars or arrays; returns NaN for times outside the video.
    """
    with h5py.File(h5_path, "r") as f:
        vg = f[camera_group]
        fps = float(vg.attrs["mp4_fps"])
        frame_time = vg["frame_time_sec"][:]
    t = np.atleast_1d(np.asarray(t_video_sec, dtype=float))
    frame_pos = t * fps
    out = np.interp(frame_pos, np.arange(frame_time.size), frame_time,
                    left=np.nan, right=np.nan)
    return out if np.ndim(t_video_sec) else float(out[0])


def convert_all_rec_to_h5(datapath, output_dir="h5_outputs", skip_existing=True):
    rec_files = find_merged_rec_files(datapath)
    print(f"Found {len(rec_files)} .rec sessions.")

    os.makedirs(output_dir, exist_ok=True)

    for rec_file in rec_files:
        fname = session_name(rec_file)
        output_h5 = os.path.join(output_dir, fname + ".h5")
        # NEW (2026-09-30): don't silently overwrite earlier outputs.
        if skip_existing and os.path.exists(output_h5):
            print(f"Exists, skipping: {output_h5}")
            continue

        print(f"\nProcessing {fname}")
        try:
            data = get_data_from_rec_file(rec_file)
        except Exception as e:
            print(f"  FAILED: {e}")
            continue
        if data is None:
            continue

        warnings = export_to_h5(output_h5, data)
        for w in warnings:
            print(f"  WARNING: {w}")
        print(f"  removed {data['resp_meta']['n_duplicate_samples_removed']} duplicate samples; "
              f"IMU channels: {sorted(data['imu'])}")
        print(f"Exported: {output_h5}")


# Run this if executing the script
if __name__ == "__main__":
    data_root = r"C:\Users\thoma\UF Dropbox\Thomas Heeps\Padilla-Coreano Lab\2026\BLA_ChR_resp_cohort2\BLA_ChR_resp_CM_hc"

    output_dir = os.path.join(data_root, "h5_outputs")
    convert_all_rec_to_h5(data_root, output_dir=output_dir)
