# ======================================================
# Core libraries for respiration + social behavior analysis
# ======================================================

# --- Core Python ---
import os
import h5py
import re
from pathlib import Path
from math import gcd

# --- Numerical & data analysis ---
import numpy as np
import pandas as pd

# --- Plotting ---
import matplotlib.pyplot as plt
import seaborn as sns

# --- Signal processing ---
from scipy.signal import butter, filtfilt, resample_poly, find_peaks

# --- Statistics ---
from scipy.stats import wilcoxon

# --- Machine learning & metrics ---
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

# --- Specialized neurophysiology tools ---
import neurokit2 as nk

# Breathmetrics
from breathmetrics import bmObject

# --- Pandas display settings ---
pd.set_option("display.max_rows", 50)
pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 1000)

print("Libraries loaded successfully")



def preprocess_resp_to_target_rate(raw_signal, fs, target_rate=400, verbose=False):
    """
    Lowpass -> downsample -> bandpass pipeline used for cagemate interaction H5s.
    Same steps as trodes_ECU_to_h5_clean_raw.py but target_rate is typically 400 Hz
    (not the 100 Hz stored in resp_clean).
    """
    raw_signal = np.asarray(raw_signal, dtype=np.float64).flatten()
    fs = float(fs)
    target_rate = float(target_rate)

    nyquist = fs / 2
    norm_cutoff = (target_rate / 2) / nyquist
    b, a = butter(N=4, Wn=norm_cutoff, btype="low")
    filtered_resp = filtfilt(b, a, raw_signal)

    downsample_factor = max(1, int(fs // target_rate))
    downsampled = resample_poly(filtered_resp, up=1, down=downsample_factor)

    rsp_cleaned = nk.signal_filter(
        downsampled,
        lowcut=0.1,
        highcut=20,
        method="butterworth",
        sampling_rate=target_rate,
        order=2,
    )
    time_vector = np.arange(len(rsp_cleaned)) / target_rate

    if verbose:
        print(f"  preprocess: raw_n={len(raw_signal)}, fs={fs} -> out_n={len(rsp_cleaned)}, fs={target_rate}")

    return rsp_cleaned, time_vector


def _read_raw_resp_from_h5(f):
    """
    Read highest-rate respiration trace from an H5 file.

    Returns (signal_1d, fs, source_key) or (None, None, None).
    - BLA Trodes H5: analog/...ECU_Ain1 voltage @ ~20 kHz
    - Aim1 merged H5: top-level resp @ ~20 kHz
    """
    if "analog" in f:
        analog_grp = f["analog"]
        analog_key = None
        for key in analog_grp.keys():
            if "ECU_Ain1" in key or key.lower().endswith("ecu_ain1"):
                analog_key = key
                break
        if analog_key is None and len(analog_grp.keys()) > 0:
            analog_key = next(iter(analog_grp.keys()))

        if analog_key is not None:
            dset = analog_grp[analog_key]
            data = np.asarray(dset[:]).squeeze()
            if data.dtype.names and "voltage" in data.dtype.names:
                raw = data["voltage"].astype(np.float64)
            else:
                raw = np.asarray(data, dtype=np.float64).flatten()
            attrs = dict(dset.attrs)
            fs = float(attrs.get("clockrate", attrs.get("sampling_frequency", 20000.0)))
            return raw, fs, f"analog/{analog_key}"

    if "resp" in f:
        raw = np.asarray(f["resp"][:]).flatten().astype(np.float64)
        fs = 20000.0
        if "resp_metadata" in f and isinstance(f["resp_metadata"], h5py.Group):
            meta = dict(f["resp_metadata"].attrs)
            fs = float(meta.get("sampling_frequency", fs))
        return raw, fs, "resp"

    return None, None, None


def load_rank_resp_signal_interactions(h5_path, target_rate=400, prefer_raw=True, verbose=False):
    """
    Load respiration for rank cagemate interaction feature extraction.

    Prefers raw high-rate data (BLA analog or Aim1 resp), then applies the shared
    20 kHz -> target_rate cleaning pipeline. Falls back to resp_clean/signal only
    when prefer_raw=False or raw data are missing.
    """
    try:
        with h5py.File(h5_path, "r") as f:
            if prefer_raw:
                raw, fs, source_key = _read_raw_resp_from_h5(f)
                if raw is not None:
                    signal, time = preprocess_resp_to_target_rate(
                        raw, fs, target_rate=target_rate, verbose=verbose
                    )
                    meta = {
                        "h5_path": h5_path,
                        "source_key": source_key,
                        "original_fs": float(fs),
                        "target_rate": float(target_rate),
                        "final_fs": float(target_rate),
                        "resampled": True,
                        "from_raw": True,
                        "n_samples": int(len(signal)),
                        "duration_sec": float(time[-1]) if len(time) else 0.0,
                    }
                    return signal, time, float(target_rate), meta

            if "resp_clean" in f and "signal" in f["resp_clean"]:
                if verbose:
                    print(f"  Falling back to resp_clean/signal for {h5_path}")
                return load_clean_resp_signal_cagemate_rank(
                    h5_path, target_rate=target_rate
                )

        raise KeyError("No raw resp or resp_clean/signal found")

    except Exception as e:
        print(f"Error loading {h5_path}: {e}")
        return None, None, None, None


def load_clean_resp_signal(h5_file, target_rate=100, verbose=True):
    """
    Loads, filters, downsamples respiration from .h5, returns cleaned signal, time vector, and metadata.
    """
    try:
        with h5py.File(h5_file, "r") as f:
            resp = f["resp"][:].flatten()

            metadata = {}
            if "resp_metadata" in f:
                metadata.update(dict(f["resp_metadata"].attrs))
            if "ekg_metadata" in f:
                metadata.update(dict(f["ekg_metadata"].attrs))
            if "metadata" in f:
                metadata.update(dict(f["metadata"].attrs))

            if "sampling_frequency" in metadata:
                fs = metadata["sampling_frequency"]
            else:
                duration_sec = metadata.get("duration_sec", None)
                fs = len(resp) / duration_sec if duration_sec else 20000.0

            duration_sec = metadata.get("duration_sec", len(resp) / fs)

            if verbose:
                print("original resp len:", len(resp))
                print("original fs:", fs)
                print("duration_sec:", duration_sec)
                print("expected duration from len/fs:", len(resp) / fs)

    except Exception as e:
        print(f"Error loading {h5_file}: {e}")
        return None, None, None, None

    rsp_cleaned, time_vector = preprocess_resp_to_target_rate(
        resp, fs, target_rate=target_rate, verbose=False
    )
    return rsp_cleaned, time_vector, target_rate, metadata





# for cagemate rank data
def load_clean_resp_signal_cagemate_rank(h5_path, target_rate=400):
    """
    Load cleaned respiration signal from the cagemate-rank H5 files.

    Expected H5 structure for this dataset:
        resp_clean/
            signal
            time

    Why this loader is separate from the old one:
    - Your valence dataset appears to use a different H5 layout.
    - These cagemate files store the cleaned respiration under resp_clean/signal
      rather than a top-level dataset named 'resp'.

    Parameters
    ----------
    h5_path : str
        Full path to one H5 recording.
    target_rate : int or float
        Desired output sampling rate after optional resampling.

    Returns
    -------
    signal : np.ndarray or None
        1D cleaned respiration signal.
    time : np.ndarray or None
        Time vector aligned to the returned signal.
    fs_out : float or None
        Final sampling rate after any resampling.
    meta : dict
        Useful metadata about loading/resampling.
    """
    try:
        with h5py.File(h5_path, "r") as f:
            # -------------------------------
            # Confirm required structure exists
            # -------------------------------
            if "resp_clean" not in f:
                raise KeyError("Missing group 'resp_clean'")

            if "signal" not in f["resp_clean"]:
                raise KeyError("Missing dataset 'resp_clean/signal'")

            if "time" not in f["resp_clean"]:
                raise KeyError("Missing dataset 'resp_clean/time'")

            # -------------------------------
            # Load cleaned respiration + time
            # -------------------------------
            raw_signal = np.asarray(f["resp_clean"]["signal"][:]).squeeze()
            raw_time = np.asarray(f["resp_clean"]["time"][:]).squeeze()

        # -------------------------------
        # Basic shape checks
        # -------------------------------
        if raw_signal.ndim != 1:
            raise ValueError(f"resp_clean/signal is not 1D. shape={raw_signal.shape}")

        if raw_time.ndim != 1:
            raise ValueError(f"resp_clean/time is not 1D. shape={raw_time.shape}")

        if len(raw_signal) == 0 or len(raw_time) == 0:
            raise ValueError("Signal or time array is empty")

        # If lengths mismatch, trim to shortest so downstream code does not break
        if len(raw_signal) != len(raw_time):
            min_len = min(len(raw_signal), len(raw_time))
            print(
                f"Warning: signal/time length mismatch in {h5_path}. "
                f"Trimming from signal={len(raw_signal)}, time={len(raw_time)} to {min_len}."
            )
            raw_signal = raw_signal[:min_len]
            raw_time = raw_time[:min_len]

        # -------------------------------
        # Estimate original sampling rate from time vector
        # -------------------------------
        dt = np.diff(raw_time)
        dt = dt[np.isfinite(dt)]

        if len(dt) == 0:
            raise ValueError("Could not estimate sampling rate from time vector")

        median_dt = np.median(dt)
        if median_dt <= 0:
            raise ValueError(f"Non-positive median dt detected: {median_dt}")

        original_fs = 1.0 / median_dt

        # -------------------------------
        # Resample if needed
        # -------------------------------
        if target_rate is not None and not np.isclose(original_fs, target_rate, rtol=1e-3):
            # Use rational resampling for cleaner signal preservation
            target_rate_int = int(round(target_rate))
            original_fs_int = int(round(original_fs))

            common_div = gcd(target_rate_int, original_fs_int)
            up = target_rate_int // common_div
            down = original_fs_int // common_div

            signal = resample_poly(raw_signal, up, down)
            fs_out = float(target_rate_int)
            time = np.arange(len(signal)) / fs_out

            resampled = True
        else:
            signal = raw_signal.astype(float, copy=False)
            time = raw_time.astype(float, copy=False)
            fs_out = float(original_fs)
            resampled = False

        meta = {
            "h5_path": h5_path,
            "source_group": "resp_clean",
            "source_signal_key": "resp_clean/signal",
            "source_time_key": "resp_clean/time",
            "original_fs": float(original_fs),
            "target_rate": None if target_rate is None else float(target_rate),
            "final_fs": float(fs_out),
            "resampled": resampled,
            "n_samples": int(len(signal)),
            "duration_sec": float(time[-1] - time[0]) if len(time) > 1 else 0.0,
        }

        return signal, time, fs_out, meta

    except Exception as e:
        print(f"Error loading {h5_path}: {e}")
        return None, None, None, None


def get_sniff_respiratory_rate(signal, time, sniff_start, sniff_end, sampling_rate=100):
    # mask the signal window
    sniff_mask = (time >= sniff_start) & (time < sniff_end)
    signal_sniff = signal[sniff_mask]
    time_sniff = time[sniff_mask]

    # detect peaks with a tighter minimum distance 12 Hz max rate)
    peaks, _ = find_peaks(signal_sniff, distance=sampling_rate * 0.0833)  # 0.0833 s = 12 Hz
    peak_times = time_sniff[peaks]

    # need at least 2 peaks to compute IBI
    if len(peak_times) < 2:
        return np.nan

    # compute IBI
    ibi = np.diff(peak_times)  # seconds

    # instantaneous rate
    inst_rate = 1.0 / ibi      # Hz

    # return average instantaneous rate (one number, same format as old code)
    avg_rate = np.mean(inst_rate)

    return avg_rate


import pandas as pd

def load_clean_boris(csv_path, subject_only=True):
    """
    Load a BORIS CSV file and standardize columns for behavior alignment.

    Parameters
    ----------
    csv_path : str
        Path to a BORIS export CSV.
    subject_only : bool, default True
        If True, keep only bouts initiated by the recorded subject.
        If False, keep bouts initiated by the subject or social agent.

    Returns
    -------
    df : DataFrame
        Columns: ['Behavior', 'Initiator', 'Start', 'Stop', 'Duration']
        Initiator is the BORIS actor label (e.g. 'subject', 'social_agent').
        Only the three social sniffing behaviors are retained.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"⚠️ Could not load {csv_path}: {e}")
        return pd.DataFrame()

    # --- Detect and rename possible column variants ---
    rename_map = {
        "Behavior": "Behavior",
        "Subject": "Subject",
        "Start (s)": "Start",
        "Stop (s)": "Stop",
        "Duration (s)": "Duration",
        "Start": "Start",
        "Stop": "Stop",
        "Duration": "Duration"
    }
    df = df.rename(columns=rename_map)

    # --- Keep essential columns only ---
    keep_cols = [c for c in ["Behavior", "Subject", "Start", "Stop", "Duration"] if c in df.columns]
    df = df[keep_cols].copy()

    # --- Clean ---
    df = df.dropna(subset=["Behavior", "Subject", "Start", "Stop"])
    df["Behavior"] = df["Behavior"].str.lower().str.strip()
    df["Subject"] = df["Subject"].str.lower().str.strip()

    # --- Filter by initiator ---
    if subject_only:
        df = df[df["Subject"] == "subject"]
    else:
        df = df[df["Subject"].isin(["subject", "social_agent"])]

    # --- Focus on relevant social behaviors ---
    behaviors_keep = ["facial sniffing", "body sniffing", "anogenital sniffing"]
    df = df[df["Behavior"].isin(behaviors_keep)]

    # Rename BORIS actor column (distinct from mouse Subject ID used later)
    df = df.rename(columns={"Subject": "Initiator"})

    # --- Sort chronologically ---
    df = df.sort_values("Start").reset_index(drop=True)

    return df


def _bla_boris_filename_patterns():
    boris_pat_cm = re.compile(
        r"^(?P<subA>\d+)_(?P<subB>\d+)(?:_\d+)?_(?P<subRole>[dis])_cm_(?P<parA>\d+)_(?P<parB>\d+)_(?P<parRole>[dis])_(?P<date>\d{8})_(?P<time>\d{6})(?:\.\d+)?_(?P<tag>.+)$",
        re.IGNORECASE,
    )
    boris_pat_nocm = re.compile(
        r"^(?P<subA>\d+)_(?P<subB>\d+)(?:_\d+)?_(?P<subRole>[dis])_(?P<parA>\d+)_(?P<parB>\d+)_(?P<parRole>[dis])_(?P<date>\d{8})_(?P<time>\d{6})(?:\.\d+)?_(?P<tag>.+)$",
        re.IGNORECASE,
    )
    return boris_pat_cm, boris_pat_nocm


def _parse_bla_boris_export(csv_path, boris_pat_cm, boris_pat_nocm):
    base = Path(csv_path).stem
    m = boris_pat_cm.match(base) or boris_pat_nocm.match(base)
    if not m:
        return None

    sub_id = f"{m.group('subA')}_{m.group('subB')}"
    par_id = f"{m.group('parA')}_{m.group('parB')}"
    date = m.group("date")
    time = m.group("time")
    tag = re.sub(r"[^A-Za-z0-9]+", "_", m.group("tag")).strip("_")
    session_key = (sub_id, par_id, date, time)
    session_id = f"{sub_id}_{par_id}_{date}_{time}"
    return session_key, session_id, tag, str(Path(csv_path).resolve())


def _build_bla_session_to_h5(bla_cm_h5_dir):
    h5_pat = re.compile(
        r"^(?P<subA>\d+)_(?P<subB>\d+)_(?P<subRole>[dis])_cm_(?P<parA>\d+)[._](?P<parB>\d+)_(?P<parRole>[dis])_(?P<date>\d{8})_(?P<time>\d{6})(?:\.rec)?\.h5$",
        re.IGNORECASE,
    )
    session_to_h5 = {}
    for h5_path in sorted(Path(bla_cm_h5_dir).glob("*.h5")):
        m = h5_pat.match(h5_path.name)
        if not m:
            continue

        sub_id = f"{m.group('subA')}_{m.group('subB')}"
        par_id = f"{m.group('parA')}_{m.group('parB').replace('.', '_')}"
        key = (sub_id, par_id, m.group("date"), m.group("time"))

        if key not in session_to_h5:
            session_to_h5[key] = str(h5_path.resolve())
        else:
            existing = session_to_h5[key]
            if (
                ".rec.h5" in os.path.basename(existing).lower()
                and ".rec.h5" not in h5_path.name.lower()
            ):
                session_to_h5[key] = str(h5_path.resolve())
    return session_to_h5


def discover_bla_boris_sa_cm_pairs(bla_cm_boris_dir, bla_cm_h5_dir=None, verbose=False):
    """
    Find BLA sessions exported as separate VT_SA (subject) and VT_CM (social agent)
    BORIS files. Most sessions have a single export; paired splits are the exception.

    Returns
    -------
    dict[str, dict]
        session_id -> {
            "session_key": (sub_id, par_id, date, time),
            "subject_boris": path to VT_SA export,
            "social_agent_boris": path to VT_CM export,
            "h5": matched respiration H5 path (when bla_cm_h5_dir is provided),
        }
    """
    boris_pat_cm, boris_pat_nocm = _bla_boris_filename_patterns()
    session_to_h5 = (
        _build_bla_session_to_h5(bla_cm_h5_dir) if bla_cm_h5_dir is not None else {}
    )

    by_session = {}
    for csv_path in sorted(Path(bla_cm_boris_dir).glob("*.csv")):
        parsed = _parse_bla_boris_export(csv_path, boris_pat_cm, boris_pat_nocm)
        if parsed is None:
            if verbose:
                print(f"  ⚠️ Could not parse BLA BORIS filename: {csv_path.name}")
            continue
        session_key, session_id, tag, path = parsed
        by_session.setdefault(
            session_id,
            {"session_key": session_key, "tags": {}},
        )
        by_session[session_id]["tags"][tag] = path

    pairs = {}
    for session_id, info in sorted(by_session.items()):
        tags = info["tags"]
        subject_boris = tags.get("VT_SA")
        social_agent_boris = tags.get("VT_CM")
        if not (subject_boris and social_agent_boris):
            continue

        entry = {
            "session_key": info["session_key"],
            "subject_boris": subject_boris,
            "social_agent_boris": social_agent_boris,
        }
        h5_path = session_to_h5.get(info["session_key"])
        if h5_path is not None:
            entry["h5"] = h5_path
        elif bla_cm_h5_dir is not None and verbose:
            print(f"  ⚠️ No matching BLA H5 for paired session {session_id}")
        pairs[session_id] = entry

    return pairs


def build_interaction_boris_catalog(
    aim1_h5_dir,
    aim1_boris_dir,
    bla_cm_h5_dir,
    bla_cm_boris_dir,
    aim1_skip_boris_stems=None,
    aim1_exclude_boris_names=None,
    prefer_bla_tag=None,
    verbose=True,
):
    """
    Build session-aligned Aim1 + BLA cagemate interaction path catalogs.

    Matches each respiration H5 to one BORIS CSV export. Aim1 duplicate exports
    for the same session prefer ``_VT`` in the filename (e.g. ``.1_VT.csv`` over
    ``.1.csv``). Names in aim1_exclude_boris_names are skipped. BLA keeps every
    matched export; sessions split into VT_SA / VT_CM are retained separately.

    Returns
    -------
    resp_paths : dict[str, str]
        trial_key -> absolute h5 path
    boris_paths : dict[str, str]
        trial_key -> absolute BORIS csv path
    bla_rank_map : dict[str, str]
        BLA subject_id -> Dominant/Subordinate (from H5 filename role letters)
    bla_sa_cm_pairs : dict[str, dict]
        BLA sessions with separate subject/social-agent exports (see
        discover_bla_boris_sa_cm_pairs)
    """
    aim1_h5_dir = Path(aim1_h5_dir)
    aim1_boris_dir = Path(aim1_boris_dir)
    bla_cm_h5_dir = Path(bla_cm_h5_dir)
    bla_cm_boris_dir = Path(bla_cm_boris_dir)
    aim1_skip_boris_stems = set(aim1_skip_boris_stems or [])
    aim1_exclude_boris_names = set(
        aim1_exclude_boris_names or {"CM_s4_8_sub4_7_20250623.csv"}
    )
    if prefer_bla_tag is not None and verbose:
        print(
            "  note: prefer_bla_tag is ignored; VT_SA and VT_CM exports are kept "
            "when both exist."
        )

    aim1_cm_h5_pat = re.compile(
        r"^CM_s(?P<subA>\d+)_(?P<subB>\d+)_(?P<rank>d|sub)(?P<parA>\d+)_(?P<parB>\d+)_(?P<date>\d{8})_(?P<time>\d{6})_merged\.h5$"
    )
    aim1_cm_boris_pat = re.compile(
        r"^CM_s(?P<subA>\d+)_(?P<subB>\d+)_(?P<rank>d|sub)(?P<parA>\d+)_(?P<parB>\d+)_(?P<date>\d{8})_(?P<time>\d{6})(?:\.\d+)?(?:_VT)?\.csv$",
        re.IGNORECASE,
    )

    def _aim1_session_stem(m):
        return (
            f"CM_s{m.group('subA')}_{m.group('subB')}_{m.group('rank')}"
            f"{m.group('parA')}_{m.group('parB')}_{m.group('date')}_{m.group('time')}"
        )

    def _pick_aim1_boris(files):
        if not files:
            return None
        if len(files) == 1:
            return files[0]
        vt_files = [f for f in files if re.search(r"_VT", f.name, re.IGNORECASE)]
        pool = vt_files if vt_files else files
        chosen = sorted(pool, key=lambda p: p.name)[0]
        if verbose:
            others = [f.name for f in files if f != chosen]
            print(
                f"  → Aim1 multiple BORIS exports for one session; using {chosen.name}; "
                f"also found: {others}"
            )
        return chosen

    boris_by_stem_aim1 = {}
    for csv_path in sorted(aim1_boris_dir.glob("*.csv")):
        if csv_path.name in aim1_exclude_boris_names:
            if verbose:
                print(f"  → Excluding Aim1 BORIS: {csv_path.name}")
            continue
        m = aim1_cm_boris_pat.match(csv_path.name)
        if not m:
            if verbose:
                print(f"  ⚠️ Unparsed Aim1 BORIS: {csv_path.name}")
            continue
        boris_by_stem_aim1.setdefault(_aim1_session_stem(m), []).append(csv_path)

    resp_paths = {}
    boris_paths = {}

    for h5_path in sorted(aim1_h5_dir.glob("*.h5")):
        m = aim1_cm_h5_pat.match(h5_path.name)
        if not m:
            if verbose:
                print(f"  ⚠️ Unparsed Aim1 H5: {h5_path.name}")
            continue
        stem = _aim1_session_stem(m)
        if stem in aim1_skip_boris_stems:
            if verbose:
                print(f"  ⚠️ Skipping empty BORIS stem: {stem}")
            continue
        candidates = boris_by_stem_aim1.get(stem, [])
        if not candidates:
            if verbose:
                print(f"  ⚠️ No matching BORIS for {h5_path.name} (stem={stem})")
            continue
        sub_id = f"{m.group('subA')}_{m.group('subB')}"
        par_id = f"{m.group('parA')}_{m.group('parB')}"
        date, time = m.group("date"), m.group("time")
        trial_key = f"AIM1cm_{sub_id}_{par_id}_{date}_{time}"
        boris_path = _pick_aim1_boris(candidates)
        if boris_path is None:
            continue
        resp_paths[trial_key] = str(h5_path.resolve())
        boris_paths[trial_key] = str(boris_path.resolve())

    h5_pat = re.compile(
        r"^(?P<subA>\d+)_(?P<subB>\d+)_(?P<subRole>[dis])_cm_(?P<parA>\d+)[._](?P<parB>\d+)_(?P<parRole>[dis])_(?P<date>\d{8})_(?P<time>\d{6})(?:\.rec)?\.h5$",
        re.IGNORECASE,
    )
    boris_pat_cm, boris_pat_nocm = _bla_boris_filename_patterns()

    session_to_h5 = _build_bla_session_to_h5(bla_cm_h5_dir)
    bla_rank_map = {}

    for h5_path in sorted(bla_cm_h5_dir.glob("*.h5")):
        m = h5_pat.match(h5_path.name)
        if not m:
            continue

        sub_id = f"{m.group('subA')}_{m.group('subB')}"
        role = m.group("subRole").lower()
        if role == "d":
            bla_rank_map[sub_id] = "Dominant"
        elif role == "s":
            bla_rank_map[sub_id] = "Subordinate"

    for csv_path in sorted(bla_cm_boris_dir.glob("*.csv")):
        parsed = _parse_bla_boris_export(csv_path, boris_pat_cm, boris_pat_nocm)
        if parsed is None:
            if verbose:
                print(f"  ⚠️ Could not parse BLA BORIS filename: {csv_path.name}")
            continue

        session_key, _session_id, tag, boris_path = parsed
        sub_id, par_id, date, time = session_key

        if session_key not in session_to_h5:
            if verbose:
                print(
                    f"  ⚠️ No matching BLA .h5 for BORIS {csv_path.name} "
                    f"(key={session_key})"
                )
            continue

        trial_key = f"BLAcm_{sub_id}_{par_id}_{date}_{time}_{tag}"
        resp_paths[trial_key] = session_to_h5[session_key]
        boris_paths[trial_key] = boris_path

    bla_sa_cm_pairs = discover_bla_boris_sa_cm_pairs(
        bla_cm_boris_dir, bla_cm_h5_dir, verbose=False
    )

    return resp_paths, boris_paths, bla_rank_map, bla_sa_cm_pairs


def normalize_subject_id(subject_id):
    """Normalize subject IDs to trial-key form (e.g. '1-1' -> '1_1')."""
    if subject_id is None or (isinstance(subject_id, float) and np.isnan(subject_id)):
        return subject_id
    return str(subject_id).replace("-", "_")


def subject_id_from_trial_key(trial_key):
    """Extract subject id from catalog trial keys like AIM1cm_1_1_1_2_..."""
    parts = str(trial_key).split("_")
    if len(parts) < 3:
        return None
    return f"{parts[1]}_{parts[2]}"


def cohort_from_trial_key(trial_key):
    """Map catalog trial keys to cohort labels used in behavior tables."""
    trial_key = str(trial_key)
    if trial_key.startswith("AIM1"):
        return "Aim1"
    if trial_key.startswith("BLA"):
        return "BLA"
    return "Other"


def boris_imbalanced_subjects(
    behavior_df,
    *,
    initiator="subject",
    subject_col="subject",
    cohort_col="cohort",
    recording_col="source_file",
    imbalance_ratio=0.4,
):
    """
    Subjects whose bout-count spread across recordings exceeds imbalance_ratio.

    Imbalance is computed within each cohort when cohort_col is present, so Aim1
    subject 1_1 and BLA subject 1_1 are evaluated independently.

    Returns list of (cohort, subject_id) tuples.
    """
    df = behavior_df
    if initiator is not None and "Initiator" in df.columns:
        df = df[df["Initiator"] == initiator]
    if df.empty:
        return []

    group_cols = [subject_col]
    if cohort_col and cohort_col in df.columns:
        group_cols = [cohort_col, subject_col]

    counts = (
        df.groupby(group_cols + [recording_col])
        .size()
        .reset_index(name="n_behaviors")
    )
    agg = counts.groupby(group_cols)["n_behaviors"].agg(["max", "min", "mean"])
    agg["imbalance_ratio"] = (agg["max"] - agg["min"]).abs() / agg["mean"]
    imbalanced = agg[agg["imbalance_ratio"] > imbalance_ratio]

    pairs = []
    if isinstance(imbalanced.index, pd.MultiIndex):
        for cohort, subj in imbalanced.index:
            pairs.append((str(cohort), normalize_subject_id(subj)))
    else:
        for subj in imbalanced.index:
            pairs.append((None, normalize_subject_id(subj)))
    return pairs


def boris_low_sniff_trials(
    behavior_df,
    *,
    initiator="subject",
    recording_col="source_file",
    min_sniffs=10,
):
    """Trial keys with fewer than min_sniffs subject-initiated bouts."""
    df = behavior_df
    if initiator is not None and "Initiator" in df.columns:
        df = df[df["Initiator"] == initiator]
    if df.empty:
        return []
    counts = df.groupby(recording_col).size()
    return counts[counts < min_sniffs].index.tolist()


def _merge_cohort_subject_exclusions(*mapping_dicts):
    """Merge dict[cohort, set[subject_id]] mappings."""
    merged = {}
    for mapping in mapping_dicts:
        if not mapping:
            continue
        for cohort, subjects in mapping.items():
            merged.setdefault(cohort, set()).update(
                {normalize_subject_id(s) for s in subjects}
            )
    return merged


def build_upstream_exclusion_set(
    behavior_df,
    *,
    imbalance_ratio=0.4,
    drop_aim1_subjects=None,
    min_sniffs_per_trial=10,
    extra_exclude_by_cohort=None,
):
    """
    Build cohort-aware subject/trial exclusion sets before BreathMetrics extraction.

    Subject exclusions are keyed by cohort (e.g. Aim1 1_1 is independent of BLA 1_1).

    Returns
    -------
    exclude_subjects_by_cohort : dict[str, set[str]]
    exclude_trials : set[str]
    report : dict
    """
    drop_aim1 = {normalize_subject_id(s) for s in (drop_aim1_subjects or [])}
    imbalanced_pairs = boris_imbalanced_subjects(
        behavior_df, imbalance_ratio=imbalance_ratio,
    )

    exclude_subjects_by_cohort = {}
    for cohort, subj in imbalanced_pairs:
        if cohort is None:
            for c in ("Aim1", "BLA", "Other"):
                exclude_subjects_by_cohort.setdefault(c, set()).add(subj)
        else:
            exclude_subjects_by_cohort.setdefault(cohort, set()).add(subj)

    if drop_aim1:
        exclude_subjects_by_cohort.setdefault("Aim1", set()).update(drop_aim1)

    if extra_exclude_by_cohort:
        exclude_subjects_by_cohort = _merge_cohort_subject_exclusions(
            exclude_subjects_by_cohort, extra_exclude_by_cohort,
        )

    low_sniff_trials = set(boris_low_sniff_trials(
        behavior_df, min_sniffs=min_sniffs_per_trial,
    ))

    return exclude_subjects_by_cohort, low_sniff_trials, {
        "imbalanced_subject_cohorts": sorted(imbalanced_pairs, key=lambda x: (x[0] or "", x[1])),
        "drop_aim1_subjects": sorted(drop_aim1),
        "low_sniff_trials": sorted(low_sniff_trials),
        "exclude_subjects_by_cohort": {
            cohort: sorted(subjects)
            for cohort, subjects in sorted(exclude_subjects_by_cohort.items())
        },
    }


def filter_interaction_paths(
    resp_paths,
    boris_paths,
    exclude_subjects=None,
    exclude_subjects_by_cohort=None,
    exclude_trials=None,
):
    """
    Filter interaction path dicts before feature extraction.

    Subject exclusions are cohort-aware when exclude_subjects_by_cohort is provided.
    The flat exclude_subjects set applies the same ids across all cohorts (legacy).

    Returns
    -------
    kept_resp, kept_boris : dict
    dropped : list[tuple[str, str]]
        (trial_key, reason) for each skipped trial
    """
    exclude_trials = set(exclude_trials or [])
    exclude_by_cohort = {
        cohort: {normalize_subject_id(s) for s in subjects}
        for cohort, subjects in (exclude_subjects_by_cohort or {}).items()
    }
    if exclude_subjects:
        legacy = {normalize_subject_id(s) for s in exclude_subjects}
        for cohort in ("Aim1", "BLA", "Other"):
            exclude_by_cohort.setdefault(cohort, set()).update(legacy)

    kept_resp, kept_boris, dropped = {}, {}, []
    for trial, h5_path in resp_paths.items():
        subj = subject_id_from_trial_key(trial)
        cohort = cohort_from_trial_key(trial)
        subj_excluded = subj in exclude_by_cohort.get(cohort, set())
        trial_excluded = trial in exclude_trials
        if subj_excluded or trial_excluded:
            reason = []
            if subj_excluded:
                reason.append(f"subject={subj}@{cohort}")
            if trial_excluded:
                reason.append("low_sniffs")
            dropped.append((trial, ", ".join(reason)))
            continue
        kept_resp[trial] = h5_path
        if trial in boris_paths:
            kept_boris[trial] = boris_paths[trial]

    return kept_resp, kept_boris, dropped


def process_all_trials(resp_paths, boris_paths, rank_map, duration_threshold=0.5):
    """
    Loops through respiration & BORIS files, computes mean respiration per bout,
    and preserves sessions even when BORIS is missing (resp-only baseline).
    """
    all_trials = []

    for trial, h5_path in resp_paths.items():
        print(f"Processing {trial}...")

        # --- Load respiration ---
        signal, time, fs, meta = load_clean_resp_signal(h5_path)
        if signal is None:
            print(f"❌ Resp load failed for {trial}")
            continue

        # --- Compute session-wide rate ---
        session_rate = get_sniff_respiratory_rate(
            signal=signal,
            time=time,
            sniff_start=time[0],
            sniff_end=time[-1],
            sampling_rate=fs
        )
        print(f"Session-wide respiratory rate for {trial}: {session_rate:.3f} Hz")

        # --- Determine subject, condition, rank ---
        subj = "_".join(trial.split("_")[1:3])
        condition = trial.split("_")[0]
        rank = rank_map.get(subj, np.nan)

        # --- Try loading BORIS ---
        if trial in boris_paths:
            boris_df = load_clean_boris(boris_paths[trial])
        else:
            boris_df = pd.DataFrame()  # missing BORIS entirely

        # --- Handle missing BORIS (resp-only) ---
        if boris_df.empty:
            print(f"⚠️ No BORIS data for {trial} — saving respiration-only session")
            all_trials.append(pd.DataFrame([{
                "Behavior": np.nan,
                "Start": np.nan,
                "Stop": np.nan,
                "Duration": np.nan,
                "MeanRate": np.nan,
                "Trial": trial,
                "Subject": subj,
                "Condition": condition,
                "Rank": rank,
                "SessionRate": session_rate,
                "Type": "Baseline"
            }]))
            continue

        # --- Compute respiration rate per behavior window ---
        boris_df["MeanRate"] = boris_df.apply(
            lambda row: get_sniff_respiratory_rate(
                signal, time, row["Start"], row["Stop"], sampling_rate=fs
            ),
            axis=1
        )

        # --- Filter short bouts ---
        pre_len = len(boris_df)
        boris_df = boris_df[boris_df["Duration"] >= duration_threshold].copy()
        post_len = len(boris_df)
        if pre_len != post_len:
            print(f"   → Filtered {pre_len - post_len} short bouts (<{duration_threshold}s)")

        # --- Add metadata ---
        boris_df["Trial"] = trial
        boris_df["Subject"] = subj
        boris_df["Condition"] = condition
        boris_df["Rank"] = rank
        boris_df["SessionRate"] = session_rate
        boris_df["Type"] = "Interaction"

        all_trials.append(boris_df)

    # --- Combine everything ---
    if not all_trials:
        print("⚠️ No valid trials processed.")
        return pd.DataFrame()

    master_df = pd.concat(all_trials, ignore_index=True)
    print(f"✅ Combined {len(master_df)} behavior windows across {len(all_trials)} trials.")
    print("Unique session types:", master_df["Type"].unique())
    return master_df


def process_all_trials_bout_level(
    resp_paths,
    boris_paths,
    rank_map,
    duration_threshold=0.5,
    pre_bout_window=2.0  # seconds before bout (optional)
):
    """
    Processes trials at the BOUT level.
    Each row = one behavioral bout with respiration metrics.
    Baseline sessions are stored as pseudo-bouts.
    """
    all_trials = []

    for trial, h5_path in resp_paths.items():
        print(f"Processing {trial}...")

        # --- Load respiration ---
        signal, time, fs, meta = load_clean_resp_signal(h5_path)
        if signal is None:
            print(f"❌ Resp load failed for {trial}")
            continue

        # --- Session-wide respiration rate (instantaneous 1/IBI) ---
        session_rate = get_sniff_respiratory_rate(
            signal=signal,
            time=time,
            sniff_start=time[0],
            sniff_end=time[-1],
            sampling_rate=fs
        )

        # --- Metadata ---
        subj = "_".join(trial.split("_")[1:3])
        condition = trial.split("_")[0]
        rank = rank_map.get(subj, np.nan)

        # --- Load BORIS if available ---
        if trial in boris_paths:
            boris_df = load_clean_boris(boris_paths[trial])
        else:
            boris_df = pd.DataFrame()

        # ============================
        # Baseline-only session
        # ============================
        if boris_df.empty:
            print(f"⚠️ No BORIS data for {trial} — saving baseline pseudo-bout")

            all_trials.append(pd.DataFrame([{
                "BoutID": 0,
                "Behavior": "Baseline",
                "Start": time[0],
                "Stop": time[-1],
                "Duration": time[-1] - time[0],
                "BoutRespRate": session_rate,
                "PreBoutRespRate": np.nan,
                "Trial": trial,
                "Subject": subj,
                "Condition": condition,
                "Rank": rank,
                "SessionRate": session_rate,
                "Type": "Baseline"
            }]))
            continue

        # ============================
        # Bout-level respiration
        # ============================
        def compute_bout_metrics(row):
            # --- Bout respiration ---
            bout_rate = get_sniff_respiratory_rate(
                signal, time,
                row["Start"], row["Stop"],
                sampling_rate=fs
            )

            # --- Pre-bout respiration ---
            pre_start = max(time[0], row["Start"] - pre_bout_window)
            pre_stop = row["Start"]

            if pre_stop <= pre_start:
                pre_rate = np.nan
            else:
                pre_rate = get_sniff_respiratory_rate(
                    signal, time,
                    pre_start, pre_stop,
                    sampling_rate=fs
                )

            return pd.Series({
                "BoutRespRate": bout_rate,
                "PreBoutRespRate": pre_rate
            })

        boris_df[["BoutRespRate", "PreBoutRespRate"]] = (
            boris_df.apply(compute_bout_metrics, axis=1)
        )

        # --- Filter short bouts ---
        pre_len = len(boris_df)
        boris_df = boris_df[boris_df["Duration"] >= duration_threshold].copy()
        post_len = len(boris_df)
        if pre_len != post_len:
            print(f"   → Filtered {pre_len - post_len} short bouts")

        # --- Add metadata ---
        boris_df["BoutID"] = np.arange(len(boris_df))
        boris_df["Trial"] = trial
        boris_df["Subject"] = subj
        boris_df["Condition"] = condition
        boris_df["Rank"] = rank
        boris_df["SessionRate"] = session_rate
        boris_df["Type"] = "Interaction"

        all_trials.append(boris_df)

    # ============================
    # Combine all trials
    # ============================
    if not all_trials:
        print("⚠️ No valid trials processed.")
        return pd.DataFrame()

    master_df = pd.concat(all_trials, ignore_index=True)

    # ============================
    # Optional derived features
    # ============================
    master_df["BoutRespRate_pre_delta"] = (
        master_df["BoutRespRate"] - master_df["PreBoutRespRate"]
    )

    master_df["BoutRespRate_z"] = (
        master_df
        .groupby("Trial")["BoutRespRate"]
        .transform(lambda x: (x - x.mean()) / x.std() if x.std() > 0 else np.nan)
    )

    print(f"✅ {len(master_df)} bouts across {master_df['Trial'].nunique()} trials")
    print("Types:", master_df["Type"].unique())

    return master_df



# ============================================================
# Standardize bout windows to a fixed duration.
# anchor="onset"  : window starts at behavior onset (interaction pipelines)
# anchor="center" : pad/crop symmetrically around bout midpoint (legacy)
# ============================================================

def standardize_bouts_to_target(boris_df, signal_start, signal_end,
                                target_dur=5.0, min_dur=1.0, anchor="center"):
    """
    Make every bout exactly target_dur seconds when possible.
    - Drop bouts shorter than min_dur
    - anchor="center": pad/crop symmetrically around bout midpoint (default)
    - anchor="onset": window starts at bout onset and extends target_dur forward
    - Clamp to session boundaries
    """
    boris_df = boris_df[boris_df["Duration"] >= min_dur].copy()

    for idx in boris_df.index:
        start = boris_df.at[idx, "Start"]
        stop = boris_df.at[idx, "Stop"]

        if anchor == "center":
            center = (start + stop) / 2.0
            new_start = center - target_dur / 2.0
            new_stop = center + target_dur / 2.0
        elif anchor == "onset":
            new_start = start
            new_stop = start + target_dur
        else:
            raise ValueError(f"Unknown anchor: {anchor!r}")

        # Shift if window goes out of bounds
        if new_start < signal_start:
            shift = signal_start - new_start
            new_start += shift
            new_stop += shift

        if new_stop > signal_end:
            shift = new_stop - signal_end
            new_start -= shift
            new_stop -= shift

        # Final clamp
        new_start = max(new_start, signal_start)
        new_stop = min(new_stop, signal_end)

        boris_df.at[idx, "Start"] = new_start
        boris_df.at[idx, "Stop"] = new_stop
        boris_df.at[idx, "Duration"] = new_stop - new_start

    return boris_df.reset_index(drop=True)


# ============================================================
# Sample non-overlapping 5-second windows from each baseline
# recording to use as "Baseline" pseudo-bouts in the matrix.
# ============================================================

def sample_baseline_windows(time, window_dur=5.0, n_windows=10,
                             seed=42):
    """
    Returns a list of (start, stop) tuples for non-overlapping
    random windows of window_dur seconds drawn from [time[0], time[-1]].
    """
    rng        = np.random.default_rng(seed)
    t_start    = time[0]
    t_end      = time[-1]
    max_start  = t_end - window_dur

    if max_start <= t_start:
        # Recording shorter than one window
        return [(t_start, t_start + window_dur)]

    starts = sorted(rng.uniform(t_start, max_start, size=n_windows * 3))
    windows, last_stop = [], -np.inf

    for s in starts:
        if s >= last_stop:               # non-overlapping
            stop = s + window_dur
            windows.append((s, stop))
            last_stop = stop
        if len(windows) >= n_windows:
            break

    return windows



# ==================================================================================================================
# Breathmetrics Helper Functions
# ==================================================================================================================


# =============================================================================
# Core helper: extract per-window features from an already-fitted bmObject
# =============================================================================

def get_bout_breath_indices(
    bm,
    bout_start_sec,
    bout_stop_sec,
    strict_within_window=False,
):
    """
    Return breath indices whose inhale onsets fall in [bout_start_sec, bout_stop_sec]
    (or fully inside the window when strict_within_window=True).
    """
    start_samp = bout_start_sec * bm.srate
    stop_samp = bout_stop_sec * bm.srate
    inhale_onsets = np.asarray(bm.inhaleOnsets, dtype=float)

    if strict_within_window:
        exhale_offsets = np.asarray(bm.exhaleOffsets[0], dtype=float)
        bout_mask = (
            (inhale_onsets >= start_samp) &
            (exhale_offsets <= stop_samp) &
            (~np.isnan(exhale_offsets))
        )
    else:
        bout_mask = (inhale_onsets >= start_samp) & (inhale_onsets <= stop_samp)

    return np.where(bout_mask)[0]


def plot_dropped_bout_diagnostic(
    signal,
    time,
    fs,
    bm,
    bout_start,
    bout_stop,
    feat_start,
    feat_stop,
    *,
    min_breaths_required=2,
    behavior=np.nan,
    trial="",
    pad_sec=0.5,
    ax=None,
    show=True,
):
    """
    Plot respiration around a dropped bout with BM inhale/exhale onsets overlaid.

    Gold shading = standardized bout window; blue shading = feature-extraction window.
    Green/red markers = inhale/exhale onsets counted inside the feature window.
    """
    n_breaths = len(get_bout_breath_indices(bm, feat_start, feat_stop))
    t0 = max(time[0], min(feat_start, bout_start) - pad_sec)
    t1 = min(time[-1], max(feat_stop, bout_stop) + pad_sec)
    mask = (time >= t0) & (time <= t1)
    t_view = time[mask]
    sig_view = signal[mask]

    if ax is None:
        _, ax = plt.subplots(figsize=(14, 3))

    ax.plot(t_view, sig_view, color="0.35", lw=0.8, label="resp")

    inhale_t = np.asarray(bm.inhaleOnsets, dtype=float) / fs
    exhale_t = np.asarray(bm.exhaleOnsets, dtype=float) / fs
    feat_inds = get_bout_breath_indices(bm, feat_start, feat_stop)

    inhale_in = inhale_t[(inhale_t >= t0) & (inhale_t <= t1)]
    exhale_in = exhale_t[(exhale_t >= t0) & (exhale_t <= t1)]
    counted_inhale = inhale_t[feat_inds] if len(feat_inds) else np.array([])
    counted_exhale = exhale_t[feat_inds] if len(feat_inds) else np.array([])

    other_inhale = np.setdiff1d(inhale_in, counted_inhale)
    other_exhale = np.setdiff1d(exhale_in, counted_exhale)

    if other_inhale.size:
        ax.scatter(
            other_inhale,
            np.interp(other_inhale, t_view, sig_view),
            s=14, c="0.75", marker="o", label="inhale onset (outside feat window)",
        )
    if other_exhale.size:
        ax.scatter(
            other_exhale,
            np.interp(other_exhale, t_view, sig_view),
            s=14, c="0.85", marker="x", label="exhale onset (outside feat window)",
        )
    if counted_inhale.size:
        ax.scatter(
            counted_inhale,
            np.interp(counted_inhale, t_view, sig_view),
            s=22, c="tab:green", marker="o", label="inhale onset (counted)",
        )
    if counted_exhale.size:
        ax.scatter(
            counted_exhale,
            np.interp(counted_exhale, t_view, sig_view),
            s=22, c="tab:red", marker="x", label="exhale onset (counted)",
        )

    ax.axvspan(bout_start, bout_stop, color="gold", alpha=0.2, label="bout window")
    ax.axvspan(feat_start, feat_stop, color="tab:blue", alpha=0.12, label="feature window")
    ax.set_title(
        f"DROPPED | {trial} | {behavior} | "
        f"n_breaths={n_breaths} (need >={min_breaths_required}) | "
        f"bout {bout_start:.2f}-{bout_stop:.2f}s | feat {feat_start:.2f}-{feat_stop:.2f}s"
    )
    ax.set_xlabel("Time (s)")
    ax.legend(loc="upper right", fontsize=8)
    if show:
        plt.tight_layout()
        plt.show()
    return ax


def plot_recording_bout_overview(
    signal,
    time,
    trial,
    kept_bouts,
    dropped_bouts,
    *,
    show=True,
    figsize=(16, 4),
    max_plot_points=80000,
):
    """
    Full-session respiration overview with kept (green) and dropped (red) bout windows.

    Parameters
    ----------
    kept_bouts, dropped_bouts : list[dict]
        Each dict must have Start and Stop (seconds). Behavior is optional (used in title only).
    """
    from matplotlib.patches import Patch

    t = np.asarray(time, dtype=float)
    s = np.asarray(signal, dtype=float)
    if len(t) > max_plot_points:
        step = int(np.ceil(len(t) / max_plot_points))
        t_plot = t[::step]
        s_plot = s[::step]
    else:
        t_plot, s_plot = t, s

    fig, ax = plt.subplots(figsize=figsize)

    for bout in kept_bouts:
        ax.axvspan(
            bout["Start"], bout["Stop"],
            color="tab:green", alpha=0.22, zorder=1,
        )
    for bout in dropped_bouts:
        ax.axvspan(
            bout["Start"], bout["Stop"],
            color="tab:red", alpha=0.30, zorder=2,
        )

    ax.plot(t_plot, s_plot, color="0.25", lw=0.4, zorder=3, label="resp")
    ax.set_xlim(t[0], t[-1])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Resp")
    ax.set_title(
        f"{trial} | kept={len(kept_bouts)} (green) | dropped={len(dropped_bouts)} (red)"
    )
    ax.legend(
        handles=[
            Patch(facecolor="tab:green", alpha=0.22, label="kept bout"),
            Patch(facecolor="tab:red", alpha=0.30, label="dropped bout"),
            Patch(facecolor="0.25", label="resp"),
        ],
        loc="upper right",
        fontsize=8,
    )
    if show:
        plt.tight_layout()
        plt.show()
    return ax


def extract_bout_features(
    bm,
    bout_start_sec,
    bout_stop_sec,
    min_breaths=4,
    strict_within_window=False,
):
    """
    Given a fitted bmObject (with estimateAllFeatures already called on the FULL
    recording), aggregate respiratory features for breaths belonging to a time
    window [bout_start_sec, bout_stop_sec].

    Why we still fit BreathMetrics on the full recording first:
    -----------------------------------------------------------
    This helper does NOT detect breaths from scratch. It only:
      1) selects breaths already detected by BreathMetrics
      2) summarizes those breath-level measurements inside this window

    That full-session fit is important because BreathMetrics estimates:
      - inhale/exhale onsets
      - pauses
      - offsets
      - durations
      - volumes
    using context from neighboring breaths and surrounding signal. Those
    estimates are usually more reliable on the full recording than on tiny
    clipped windows.

    Parameters
    ----------
    bm : fitted bmObject
        BreathMetrics object already fit on the full session.
    bout_start_sec : float
        Window start time in seconds.
    bout_stop_sec : float
        Window stop time in seconds.
    min_breaths : int, default=4
        Minimum number of breaths required to keep this window.

        CHANGED:
        The old code used 2 breaths. That is enough to compute some quantities,
        but it is weak for variability-style features (CVs), pause fractions,
        and volume summaries. Using 4 makes the window features more stable.
    strict_within_window : bool, default=False
        If False:
            include a breath if its inhale onset falls inside the window.
        If True:
            require the full breath to fit within the window, using inhale onset
            and exhale offset.

        This is optional because the original behavior is reasonable for many ML
        pipelines, but strict mode is useful as a sensitivity check.

    Returns
    -------
    dict or None
        Dictionary of aggregated features, or None if the window does not
        contain enough valid breaths.
    """

    inhale_onsets = np.asarray(bm.inhaleOnsets, dtype=float)
    bout_inds = get_bout_breath_indices(
        bm, bout_start_sec, bout_stop_sec,
        strict_within_window=strict_within_window,
    )

    # CHANGED:
    # old code required only 2 breaths. We now default to 4 because:
    # - 2 breaths gives only 1 IBI
    # - CV metrics become unstable
    # - pause percentages become very coarse
    if len(bout_inds) < min_breaths:
        return None

    # n_breaths:
    #   number of breaths contributing to this window summary
    # Naming reason:
    #   simple bookkeeping feature that also helps you later inspect whether
    #   some windows are based on too little data.
    n_breaths = len(bout_inds)

    # -------------------------------------------------------------------------
    # 1) RATE / INTER-BREATH TIMING
    # -------------------------------------------------------------------------
    # bout_inhale_onsets:
    #   inhale start times for only the breaths selected in this window
    bout_inhale_onsets = inhale_onsets[bout_inds]

    # ibis_sec:
    #   inter-breath intervals in seconds, computed between consecutive inhale
    #   onsets
    # Naming reason:
    #   IBI = inter-breath interval, a standard timing measure.
    ibis_sec = np.diff(bout_inhale_onsets) / bm.srate

    # If somehow we have no interval after filtering, stop here
    if len(ibis_sec) == 0:
        return None

    # mean_ibi_sec:
    #   mean time between breath starts in seconds
    # Naming reason:
    #   explicit "_sec" suffix makes the units clear.
    mean_ibi_sec = np.nanmean(ibis_sec)

    # breathing_rate_hz:
    #   breaths per second, derived from 1 / mean IBI
    # Naming reason:
    #   "_hz" here means events per second.
    breathing_rate_hz = 1.0 / mean_ibi_sec if mean_ibi_sec > 0 else np.nan

    # cv_ibi:
    #   variability of inter-breath timing
    #
    # CHANGED:
    # old code called this "cv_breathing_rate", but mathematically it was really
    # CV of the breath intervals, not CV of rate itself. This new name is more
    # honest and easier to interpret.
    cv_ibi = np.nan
    if len(ibis_sec) >= 2 and mean_ibi_sec > 0:
        cv_ibi = np.nanstd(ibis_sec) / mean_ibi_sec

    # -------------------------------------------------------------------------
    # 2) INHALE / EXHALE DURATIONS
    # -------------------------------------------------------------------------
    # These durations were already computed by BM from onset/offset landmarks on
    # the full session, and BM stores them in seconds.
    inhale_durs_sec = np.asarray(bm.inhaleDurations[0, bout_inds], dtype=float)
    exhale_durs_sec = np.asarray(bm.exhaleDurations[0, bout_inds], dtype=float)

    inhale_durs_sec = inhale_durs_sec[~np.isnan(inhale_durs_sec)]
    exhale_durs_sec = exhale_durs_sec[~np.isnan(exhale_durs_sec)]

    # mean_inhale_dur_sec / mean_exhale_dur_sec:
    #   average duration of inhale/exhale phases in this window
    mean_inhale_dur_sec = np.nanmean(inhale_durs_sec) if inhale_durs_sec.size else np.nan
    mean_exhale_dur_sec = np.nanmean(exhale_durs_sec) if exhale_durs_sec.size else np.nan

    # CVs only computed when there are enough values to make variability at
    # least somewhat meaningful.
    cv_inhale_dur = np.nan
    if inhale_durs_sec.size >= 2 and mean_inhale_dur_sec > 0:
        cv_inhale_dur = np.nanstd(inhale_durs_sec) / mean_inhale_dur_sec

    cv_exhale_dur = np.nan
    if exhale_durs_sec.size >= 2 and mean_exhale_dur_sec > 0:
        cv_exhale_dur = np.nanstd(exhale_durs_sec) / mean_exhale_dur_sec

    # ie_ratio:
    #   inhale-to-exhale duration ratio
    # Naming reason:
    #   short for inhale/exhale ratio; common compact respiratory shorthand.
    ie_ratio = (
        mean_inhale_dur_sec / mean_exhale_dur_sec
        if (
            not np.isnan(mean_inhale_dur_sec) and
            not np.isnan(mean_exhale_dur_sec) and
            mean_exhale_dur_sec > 0
        )
        else np.nan
    )

    # -------------------------------------------------------------------------
    # 3) PEAK FLOW FEATURES
    # -------------------------------------------------------------------------
    # peakInspiratoryFlows / troughExpiratoryFlows come from BM's detected
    # extrema on the full signal.
    peak_insp_flows = np.asarray(bm.peakInspiratoryFlows[bout_inds], dtype=float)
    peak_exp_flows  = np.asarray(bm.troughExpiratoryFlows[bout_inds], dtype=float)

    # mean_peak_insp_flow / mean_peak_exp_flow:
    #   average inspiratory and expiratory extrema amplitude
    # Naming reason:
    #   kept close to original naming for compatibility with your existing
    #   pipeline and plots.
    mean_peak_insp_flow = np.nanmean(peak_insp_flows) if peak_insp_flows.size else np.nan
    mean_peak_exp_flow  = np.nanmean(peak_exp_flows) if peak_exp_flows.size else np.nan

    valid_peak_insp_flows = peak_insp_flows[~np.isnan(peak_insp_flows)]

    cv_peak_insp_flow = np.nan
    if valid_peak_insp_flows.size >= 2 and np.nanmean(valid_peak_insp_flows) != 0:
        cv_peak_insp_flow = np.nanstd(valid_peak_insp_flows) / np.nanmean(valid_peak_insp_flows)

    # -------------------------------------------------------------------------
    # 4) VOLUME FEATURES
    # -------------------------------------------------------------------------
    inhale_vols = np.asarray(bm.inhaleVolumes[0, bout_inds], dtype=float)
    exhale_vols = np.asarray(bm.exhaleVolumes[0, bout_inds], dtype=float)

    inhale_vols = inhale_vols[~np.isnan(inhale_vols)]
    exhale_vols = exhale_vols[~np.isnan(exhale_vols)]

    mean_inhale_vol = np.nanmean(inhale_vols) if inhale_vols.size else np.nan
    mean_exhale_vol = np.nanmean(exhale_vols) if exhale_vols.size else np.nan

    # CHANGED:
    # old name: mean_tidal_vol
    #
    # Reason for rename:
    # this is not a perfectly paired per-breath tidal volume computation.
    # It is mean inhale volume + mean exhale volume, so "proxy" is more honest.
    mean_tidal_vol_proxy = (
        mean_inhale_vol + mean_exhale_vol
        if (not np.isnan(mean_inhale_vol) and not np.isnan(mean_exhale_vol))
        else np.nan
    )

    # CHANGED:
    # old code called this cv_tidal_vol, but it actually used inhale volumes
    # only. Renaming it to cv_inhale_vol matches the math.
    cv_inhale_vol = np.nan
    if inhale_vols.size >= 2 and mean_inhale_vol > 0:
        cv_inhale_vol = np.nanstd(inhale_vols) / mean_inhale_vol

    # CHANGED:
    # old name: minute_ventilation
    #
    # Reason for rename:
    # breathing_rate_hz is breaths/second, so multiplying by volume gives a
    # per-second proxy, not a literal per-minute quantity. This is still useful
    # for ML, but the old name overclaimed physiological precision.
    ventilation_proxy = (
        breathing_rate_hz * mean_tidal_vol_proxy
        if (not np.isnan(breathing_rate_hz) and not np.isnan(mean_tidal_vol_proxy))
        else np.nan
    )

    # -------------------------------------------------------------------------
    # 5) PAUSE FEATURES
    # -------------------------------------------------------------------------
    inh_pause_durs_sec = np.asarray(bm.inhalePauseDurations[0, bout_inds], dtype=float)
    exh_pause_durs_sec = np.asarray(bm.exhalePauseDurations[0, bout_inds], dtype=float)

    inh_pause_durs_sec = inh_pause_durs_sec[~np.isnan(inh_pause_durs_sec)]
    exh_pause_durs_sec = exh_pause_durs_sec[~np.isnan(exh_pause_durs_sec)]

    # pct_breaths_with_inhale_pause / pct_breaths_with_exhale_pause:
    #   fraction of breaths in this window that contain that pause type
    pct_breaths_with_inhale_pause = len(inh_pause_durs_sec) / n_breaths
    pct_breaths_with_exhale_pause = len(exh_pause_durs_sec) / n_breaths

    # If no pauses exist, using 0.0 is often more interpretable than NaN for
    # the mean pause duration itself.
    mean_inhale_pause_dur_sec = np.nanmean(inh_pause_durs_sec) if inh_pause_durs_sec.size else 0.0
    mean_exhale_pause_dur_sec = np.nanmean(exh_pause_durs_sec) if exh_pause_durs_sec.size else 0.0

    cv_inhale_pause_dur = np.nan
    if inh_pause_durs_sec.size >= 2 and mean_inhale_pause_dur_sec > 0:
        cv_inhale_pause_dur = np.nanstd(inh_pause_durs_sec) / mean_inhale_pause_dur_sec

    cv_exhale_pause_dur = np.nan
    if exh_pause_durs_sec.size >= 2 and mean_exhale_pause_dur_sec > 0:
        cv_exhale_pause_dur = np.nanstd(exh_pause_durs_sec) / mean_exhale_pause_dur_sec

    # -------------------------------------------------------------------------
    # 6) DUTY CYCLES
    # -------------------------------------------------------------------------
    # Duty cycle = fraction of the average breath cycle spent in a phase.
    inhale_duty_cycle = mean_inhale_dur_sec / mean_ibi_sec if mean_ibi_sec > 0 else np.nan
    exhale_duty_cycle = mean_exhale_dur_sec / mean_ibi_sec if mean_ibi_sec > 0 else np.nan

    # Pause duty cycles scale the mean pause duration by the fraction of breaths
    # that actually contained the pause, which matches the logic used in the BM
    # secondary-feature code.
    inhale_pause_duty_cycle = (
        (mean_inhale_pause_dur_sec * pct_breaths_with_inhale_pause) / mean_ibi_sec
        if mean_ibi_sec > 0 else np.nan
    )
    exhale_pause_duty_cycle = (
        (mean_exhale_pause_dur_sec * pct_breaths_with_exhale_pause) / mean_ibi_sec
        if mean_ibi_sec > 0 else np.nan
    )

    # -------------------------------------------------------------------------
    # Return a flat dictionary for downstream dataframe assembly
    # -------------------------------------------------------------------------
    return {
        # ---------------------------------------------------------------------
        # bookkeeping
        # ---------------------------------------------------------------------
        "n_breaths": n_breaths,

        # ---------------------------------------------------------------------
        # rate / timing
        # ---------------------------------------------------------------------
        "breathing_rate_hz": breathing_rate_hz,
        "mean_ibi_sec": mean_ibi_sec,
        "cv_ibi": cv_ibi,

        # ---------------------------------------------------------------------
        # inhale phase
        # ---------------------------------------------------------------------
        "mean_inhale_dur_sec": mean_inhale_dur_sec,
        "cv_inhale_dur": cv_inhale_dur,

        # ---------------------------------------------------------------------
        # exhale phase
        # ---------------------------------------------------------------------
        "mean_exhale_dur_sec": mean_exhale_dur_sec,
        "cv_exhale_dur": cv_exhale_dur,

        # ---------------------------------------------------------------------
        # inhale/exhale timing relationship
        # ---------------------------------------------------------------------
        "ie_ratio": ie_ratio,

        # ---------------------------------------------------------------------
        # peak flow summaries
        # ---------------------------------------------------------------------
        "mean_peak_insp_flow": mean_peak_insp_flow,
        "mean_peak_exp_flow": mean_peak_exp_flow,
        "cv_peak_insp_flow": cv_peak_insp_flow,

        # ---------------------------------------------------------------------
        # volume summaries
        # ---------------------------------------------------------------------
        "mean_inhale_vol": mean_inhale_vol,
        "mean_exhale_vol": mean_exhale_vol,
        "mean_tidal_vol_proxy": mean_tidal_vol_proxy,
        "cv_inhale_vol": cv_inhale_vol,
        "ventilation_proxy": ventilation_proxy,

        # ---------------------------------------------------------------------
        # inhale pause summaries
        # ---------------------------------------------------------------------
        "pct_breaths_with_inhale_pause": pct_breaths_with_inhale_pause,
        "mean_inhale_pause_dur_sec": mean_inhale_pause_dur_sec,
        "cv_inhale_pause_dur": cv_inhale_pause_dur,
        "inhale_pause_duty_cycle": inhale_pause_duty_cycle,

        # ---------------------------------------------------------------------
        # exhale pause summaries
        # ---------------------------------------------------------------------
        "pct_breaths_with_exhale_pause": pct_breaths_with_exhale_pause,
        "mean_exhale_pause_dur_sec": mean_exhale_pause_dur_sec,
        "cv_exhale_pause_dur": cv_exhale_pause_dur,
        "exhale_pause_duty_cycle": exhale_pause_duty_cycle,

        # ---------------------------------------------------------------------
        # overall phase fractions
        # ---------------------------------------------------------------------
        "inhale_duty_cycle": inhale_duty_cycle,
        "exhale_duty_cycle": exhale_duty_cycle,
    }


# =============================================================================
# Session-level fitter
# =============================================================================

def fit_bm_session(signal, srate, data_type="rodentAirflow"):
    """
    Fits a bmObject on a full session signal.

    Parameters
    ----------
    signal    : 1-D np.ndarray, baseline-corrected respiration (already cleaned)
    srate     : float, sampling rate in Hz
    data_type : str, passed to bmObject (default 'rodentAirflow')

    Returns
    -------
    bm : fitted bmObject, or None on failure
    """
    try:
        bm = bmObject(signal, srate, data_type)
        bm.estimateAllFeatures()
        return bm
    except Exception as e:
        print(f"  ⚠️  bmObject fitting failed: {e}")
        return None


# =============================================================================
# Main pipeline
# =============================================================================

def build_resp_feature_matrix(
    resp_paths,
    boris_paths,
    rank_map,
    duration_threshold=0.5,
    pre_bout_window=2.0,
    data_type="rodentAirflow",
    target_srate=100,
    min_breaths_per_bout=2,
):
    """
    Builds a per-bout respiratory feature matrix for valence classification.

    Parameters
    ----------
    resp_paths          : dict  {trial_id: h5_path}
    boris_paths         : dict  {trial_id: boris_csv_path}
    rank_map            : dict  {subject_id: rank}
    duration_threshold  : float, minimum bout duration in seconds to keep
    pre_bout_window     : float, seconds before bout used for pre-bout rate
    data_type           : str, bmObject data type (default 'rodentAirflow')
    target_srate        : float, expected sampling rate of loaded signals
    min_breaths_per_bout: int,   bouts with fewer breaths are dropped

    Returns
    -------
    master_df : pd.DataFrame, one row per bout with all respiratory features
    """

    all_rows = []

    for trial, h5_path in resp_paths.items():
        print(f"\nProcessing {trial}...")

        # ── 1. Load signal ────────────────────────────────────────────────
        signal, time, fs, meta = load_clean_resp_signal(h5_path, target_rate=target_srate)
        if signal is None:
            print(f"  ❌ Load failed, skipping.")
            continue

        # ── 2. Fit bmObject once for the full session ─────────────────────
        print(f"  Fitting bmObject on full session ({len(signal)/fs:.1f}s @ {fs}Hz)...")
        bm = fit_bm_session(signal, fs, data_type=data_type)
        if bm is None:
            continue
        print(f"  ✓ Found {len(bm.inhaleOnsets)} breaths in session")

        # ── 3. Metadata ───────────────────────────────────────────────────
        subj      = "_".join(trial.split("_")[1:3])
        condition = trial.split("_")[0]
        rank      = rank_map.get(subj, np.nan)

        # ── 4. Load BORIS ─────────────────────────────────────────────────
        if trial in boris_paths:
            boris_df = load_clean_boris(boris_paths[trial])
        else:
            boris_df = pd.DataFrame()

        # ── 5. Baseline-only session ──────────────────────────────────────
        if boris_df.empty:
            print(f"  ⚠️  No BORIS data — saving baseline pseudo-bout")
            feats = extract_bout_features(bm, time[0], time[-1])
            if feats is not None:
                row = {
                    "BoutID"    : 0,
                    "Behavior"  : "Baseline",
                    "Start"     : time[0],
                    "Stop"      : time[-1],
                    "Duration"  : time[-1] - time[0],
                    "Label"     : np.nan,      # valence label placeholder
                    "Trial"     : trial,
                    "Subject"   : subj,
                    "Condition" : condition,
                    "Rank"      : rank,
                    "Type"      : "Baseline",
                }
                row.update(feats)
                all_rows.append(row)
            continue

        # ── 6. Filter short bouts ─────────────────────────────────────────
        n_pre = len(boris_df)
        boris_df = boris_df[boris_df["Duration"] >= duration_threshold].copy()
        if len(boris_df) < n_pre:
            print(f"  → Filtered {n_pre - len(boris_df)} short bouts (<{duration_threshold}s)")

        # ── 7. Per-bout feature extraction ────────────────────────────────
        n_dropped = 0
        for bout_id, bout in boris_df.iterrows():

            feats = extract_bout_features(bm, bout["Start"], bout["Stop"])

            if feats is None or feats["n_breaths"] < min_breaths_per_bout:
                n_dropped += 1
                continue

            # --- Pre-bout rate (simple IBI-based, same logic as before) ---
            pre_start = max(time[0], bout["Start"] - pre_bout_window)
            pre_feats = extract_bout_features(bm, pre_start, bout["Start"])
            pre_rate  = pre_feats["breathing_rate_hz"] if pre_feats is not None else np.nan

            row = {
                "BoutID"          : bout_id,
                "Behavior"        : bout.get("Behavior", np.nan),
                "Start"           : bout["Start"],
                "Stop"            : bout["Stop"],
                "Duration"        : bout["Duration"],
                "Label"           : bout.get("Label", np.nan),   # valence label if already coded
                "PreBoutRespRate" : pre_rate,
                "Trial"           : trial,
                "Subject"         : subj,
                "Condition"       : condition,
                "Rank"            : rank,
                "Type"            : "Interaction",
            }
            row.update(feats)

            # Derived: delta from pre-bout rate
            row["resp_rate_delta"] = (
                row["breathing_rate_hz"] - pre_rate
                if not np.isnan(pre_rate) else np.nan
            )

            all_rows.append(row)

        if n_dropped:
            print(f"  → Dropped {n_dropped} bouts with <{min_breaths_per_bout} breaths")

        print(f"  ✓ {len(boris_df) - n_dropped} bouts extracted for {trial}")

    # ── 8. Assemble master DataFrame ──────────────────────────────────────
    if not all_rows:
        print("\n⚠️  No valid bouts processed.")
        return pd.DataFrame()

    master_df = pd.DataFrame(all_rows).reset_index(drop=True)

    # Session-normalised z-score of breathing rate (useful classifier feature)
    master_df["breathing_rate_z"] = (
        master_df
        .groupby("Trial")["breathing_rate_hz"]
        .transform(lambda x: (x - x.mean()) / x.std() if x.std() > 0 else np.nan)
    )

    print(f"\n✅ Feature matrix: {len(master_df)} bouts × {len(master_df.columns)} columns")
    print(f"   Trials: {master_df['Trial'].nunique()}, "
          f"Subjects: {master_df['Subject'].nunique()}")
    print(f"   Types : {master_df['Type'].value_counts().to_dict()}")

    return master_df


# =============================================================================
# Convenience: return only the feature columns for ML (drop metadata)
# =============================================================================

METADATA_COLS = [
    "BoutID", "Behavior", "Start", "Stop", "Duration", "Label",
    "PreBoutRespRate", "Trial", "Subject", "Condition", "Rank",
    "Type", "resp_rate_delta", "breathing_rate_z",
]

def get_feature_matrix(master_df, drop_incomplete=True):
    """
    Returns (X, y, groups) ready for sklearn.

    Parameters
    ----------
    master_df       : output of build_resp_feature_matrix()
    drop_incomplete : if True, drop rows with any NaN in feature columns

    Returns
    -------
    X      : pd.DataFrame of respiratory feature columns only
    y      : pd.Series of valence labels (from "Label" column)
    groups : pd.Series of subject IDs (for GroupKFold cross-validation)
    """
    feature_cols = [c for c in master_df.columns if c not in METADATA_COLS]
    X = master_df[feature_cols].copy()
    y = master_df["Label"].copy()
    groups = master_df["Subject"].copy()

    if drop_incomplete:
        valid = X.notna().all(axis=1)
        n_dropped = (~valid).sum()
        if n_dropped:
            print(f"Dropping {n_dropped} rows with NaN features")
        X, y, groups = X[valid], y[valid], groups[valid]

    return X, y, groups


# =============================================================================
# Creates Windows Throughout Sessions
# =============================================================================

def sample_random_nonoverlapping_windows(
    time,
    signal=None,
    window_dur=5.0,
    n_windows=100,
    seed=42,
    allow_partial_if_short=False,
    dead_std_frac=0.02,
    dead_ptp_frac=0.02,
    dead_abs_std_floor=1e-6,
    dead_abs_ptp_floor=1e-6,
):
    """
    Sample up to n_windows non-overlapping fixed-duration windows from a session.
    Optionally rejects windows that appear "dead" (near-flat signal).
    Sampling is spread-aware: it first draws across temporal strata, then fills
    any remaining slots with a distance-weighted random draw to reduce clumping.

    Parameters
    ----------
    time : np.ndarray
        Session time vector in seconds.
    signal : np.ndarray or None
        Respiration signal aligned to `time`. If provided, windows whose
        within-window variability is too low are excluded before random draw.
    window_dur : float
        Window duration in seconds.
    n_windows : int
        Maximum number of windows to sample.
    seed : int
        Random seed for reproducibility.
    allow_partial_if_short : bool
        If True and session is shorter than one full window, return one
        clamped session-spanning window. If False, return [].
    dead_std_frac : float
        Fraction of global signal std used as the minimum window std threshold.
    dead_ptp_frac : float
        Fraction of global signal peak-to-peak range used as minimum window
        range threshold.
    dead_abs_std_floor : float
        Absolute lower bound for window std threshold.
    dead_abs_ptp_floor : float
        Absolute lower bound for window range threshold.

    Returns
    -------
    windows : list of (start, stop) tuples
    """
    rng = np.random.default_rng(seed)

    t_start = float(time[0])
    t_end   = float(time[-1])
    session_dur = t_end - t_start

    if session_dur < window_dur:
        if allow_partial_if_short:
            return [(t_start, t_end)]
        return []

    # candidate starts for full non-overlapping windows
    possible_starts = np.arange(t_start, t_end - window_dur, window_dur)

    if len(possible_starts) == 0:
        return []

    # If signal is provided, drop candidate windows with very low variation.
    if signal is not None:
        signal = np.asarray(signal)
        if len(signal) != len(time):
            raise ValueError("signal and time must have the same length")

        global_std = float(np.nanstd(signal))
        global_ptp = float(np.nanmax(signal) - np.nanmin(signal))
        min_std = max(dead_abs_std_floor, dead_std_frac * global_std)
        min_ptp = max(dead_abs_ptp_floor, dead_ptp_frac * global_ptp)

        valid_starts = []
        for s in possible_starts:
            e = s + window_dur
            mask = (time >= s) & (time < e)
            seg = signal[mask]
            if seg.size == 0:
                continue

            seg_std = float(np.nanstd(seg))
            seg_ptp = float(np.nanmax(seg) - np.nanmin(seg))
            if np.isfinite(seg_std) and np.isfinite(seg_ptp):
                if (seg_std >= min_std) and (seg_ptp >= min_ptp):
                    valid_starts.append(float(s))

        possible_starts = np.array(valid_starts, dtype=float)
        if len(possible_starts) == 0:
            return []

    n_use = min(n_windows, len(possible_starts))

    # Spread-aware random sampling:
    # 1) stratified random draw over session timeline
    # 2) if some strata are empty, fill remaining slots with a
    #    distance-weighted random draw to avoid local clustering
    starts_sorted = np.sort(possible_starts.astype(float))
    session_last_start = t_end - window_dur
    bin_edges = np.linspace(t_start, session_last_start, n_use + 1)

    chosen_starts = []
    chosen_set = set()
    for i in range(n_use):
        left = bin_edges[i]
        right = bin_edges[i + 1]
        if i == n_use - 1:
            in_bin = starts_sorted[(starts_sorted >= left) & (starts_sorted <= right)]
        else:
            in_bin = starts_sorted[(starts_sorted >= left) & (starts_sorted < right)]
        if in_bin.size == 0:
            continue
        s = float(rng.choice(in_bin))
        chosen_starts.append(s)
        chosen_set.add(s)

    if len(chosen_starts) < n_use:
        remaining = starts_sorted[~np.isin(starts_sorted, np.array(chosen_starts, dtype=float))]
        while len(chosen_starts) < n_use and remaining.size > 0:
            if len(chosen_starts) == 0:
                idx = int(rng.integers(0, remaining.size))
            else:
                chosen_arr = np.array(chosen_starts, dtype=float)
                # Favor candidates farther from already selected windows.
                min_dists = np.min(np.abs(remaining[:, None] - chosen_arr[None, :]), axis=1)
                weights = min_dists + 1e-12
                probs = weights / weights.sum()
                idx = int(rng.choice(np.arange(remaining.size), p=probs))

            s = float(remaining[idx])
            chosen_starts.append(s)
            chosen_set.add(s)
            remaining = remaining[remaining != s]

    windows = sorted((float(s), float(s + window_dur)) for s in chosen_starts)
    return windows


def extract_features_from_windows(
    bm,
    windows,
    min_breaths_per_window=2,
):
    """
    Extract respiratory features for a list of (start, stop) windows
    from an already-fitted bmObject.

    Parameters
    ----------
    bm : fitted bmObject
    windows : list of (start, stop)
    min_breaths_per_window : int
        Minimum breaths required to keep a window.

    Returns
    -------
    rows : list of dict
        Each dict contains Start, Stop, Duration, and extracted features.
    """
    rows = []

    for win_id, (start, stop) in enumerate(windows):
        feats = extract_bout_features(bm, start, stop)

        if feats is None:
            continue
        if feats["n_breaths"] < min_breaths_per_window:
            continue

        row = {
            "WindowID": win_id,
            "Start": start,
            "Stop": stop,
            "Duration": stop - start,
        }
        row.update(feats)
        rows.append(row)

    return rows


# --- Rank / cagemate (paired mice) extraction ---------------------------------

_RANK_PRE_RE = re.compile(r"^(\d+_\d+)_2_([dis])$")
_RANK_POST_RE = re.compile(r"^(.+)_([dis])_(\d{8}_\d{6})$")


def canonical_rank_recording_basename(path_or_name, rename_map):
    """
    Map an on-disk .h5 basename to the canonical (renamed) basename from rename_map.

    Original cohort filenames are keys; values embed standardized mouse IDs and _SUB/_DOM.
    If the file is already canonical or unlisted, returns basename unchanged.
    """
    base = os.path.basename(str(path_or_name))
    lookup = base.replace(".rec.h5", ".h5")
    if lookup in rename_map:
        return rename_map[lookup]
    if base in rename_map:
        return rename_map[base]
    return base


def parse_rank_cagemate_canonical_basename(canonical_basename):
    """
    Parse metadata from a canonical rank .h5 name (rename_map value).

    The first mouse token (before ``_2_<role>_cm_``) is the recorded-airflow subject.
    Rank for that mouse is encoded as ``_SUB`` / ``_DOM`` before ``.h5``.
    Returns None if the name does not match the expected pattern.
    """
    base = os.path.basename(str(canonical_basename))
    stem = base.rsplit(".", 1)[0]
    rank = None
    if stem.endswith("_SUB"):
        rank = "Subordinate"
        stem = stem[: -len("_SUB")]
    elif stem.endswith("_DOM"):
        rank = "Dominant"
        stem = stem[: -len("_DOM")]
    else:
        return None

    if "_cm_" not in stem:
        return None
    pre, post = stem.split("_cm_", 1)
    m_pre, m_post = _RANK_PRE_RE.match(pre), _RANK_POST_RE.match(post)
    if not m_pre or not m_post:
        return None

    subject_id = m_pre.group(1)
    partner_id = m_post.group(1)
    session_stamp = m_post.group(3)
    partner_rank = "Dominant" if rank == "Subordinate" else "Subordinate"
    dyad_tokens = sorted([subject_id, partner_id])
    dyad_id = f"{dyad_tokens[0]}__{dyad_tokens[1]}"

    return {
        "subject_id": subject_id,
        "partner_id": partner_id,
        "dyad_id": dyad_id,
        "rank_recorded_mouse": rank,
        "partner_rank": partner_rank,
        "session_stamp": session_stamp,
        "subject_side_role": m_pre.group(2),
        "partner_side_role": m_post.group(2),
    }


def build_rank_cagemate_resp_paths_from_dir(h5_dir, rename_map):
    """
    Build trial_id -> absolute path for cagemate rank .h5 files under ``h5_dir``.

    Trial keys are the canonical recording stem (``rename_map`` value without ``.h5``),
    so they align with ``Recording`` in :func:`build_rank_cagemate_window_feature_matrix`.

    Only files whose canonical basename parses with
    :func:`parse_rank_cagemate_canonical_basename` are included (unknown files are skipped).
    If both ``name.h5`` and ``name.rec.h5`` map to the same canonical recording, the
    non-``.rec`` file is preferred.
    """
    from collections import defaultdict

    h5_dir = Path(h5_dir)
    candidates = defaultdict(list)
    for p in sorted(h5_dir.iterdir()):
        if not p.is_file():
            continue
        low = p.name.lower()
        if not low.endswith(".h5"):
            continue
        canonical_name = canonical_rank_recording_basename(p, rename_map)
        if parse_rank_cagemate_canonical_basename(canonical_name) is None:
            continue
        trial_key = Path(canonical_name).stem
        candidates[trial_key].append(p)

    out = {}
    for trial_key, plist in candidates.items():
        plain = [x for x in plist if not x.name.lower().endswith(".rec.h5")]
        chosen = plain[0] if plain else plist[0]
        out[trial_key] = str(chosen.resolve())
    return out


def build_rank_cagemate_window_feature_matrix(
    resp_paths,
    rename_map,
    data_type="rodentAirflow",
    target_srate=400,
    window_sec=3.0,
    n_windows_per_session=100,
    min_breaths_per_window=2,
    random_state=42,
):
    """
    Window-level breathmetrics features for cagemate rank recordings.

    Parameters
    ----------
    resp_paths : dict
        trial_id -> path to .h5 (keys can be any label; metadata comes from filenames).
    rename_map : dict
        original_basename -> canonical_basename (see rank notebooks).

    Each row includes Subject (recorded mouse), Partner, DyadID, Rank (of recorded mouse),
    and PartnerRank for grouped CV / dyad-level splits.
    """
    all_rows = []

    for trial, h5_path in resp_paths.items():
        orig_base = os.path.basename(str(h5_path))
        canonical = canonical_rank_recording_basename(h5_path, rename_map)
        meta = parse_rank_cagemate_canonical_basename(canonical)

        print(f"\n[Rank CM] trial={trial}")
        print(f"  file: {orig_base} -> canonical: {canonical}")

        if meta is None:
            print("  Could not parse canonical basename (expected rename_map target with _SUB/_DOM)")
            continue

        signal, time, fs, _meta = load_clean_resp_signal(
            h5_path,
            target_rate=target_srate,
        )
        if signal is None:
            print("  Load failed")
            continue

        bm = fit_bm_session(signal, fs, data_type=data_type)
        if bm is None:
            continue

        print(f"  {len(bm.inhaleOnsets)} breaths detected in session")

        windows = sample_random_nonoverlapping_windows(
            time=time,
            signal=signal,
            window_dur=window_sec,
            n_windows=n_windows_per_session,
            seed=random_state,
            allow_partial_if_short=False,
        )
        if len(windows) == 0:
            print("  No usable full windows")
            continue

        window_rows = extract_features_from_windows(
            bm=bm,
            windows=windows,
            min_breaths_per_window=min_breaths_per_window,
        )
        print(f"  {len(window_rows)} usable windows")

        for row in window_rows:
            row.update(
                {
                    "Trial": trial,
                    "Subject": meta["subject_id"],
                    "Partner": meta["partner_id"],
                    "DyadID": meta["dyad_id"],
                    "Recording": canonical,
                    "MouseSubject": meta["subject_id"],
                    "Condition": "CM",
                    "Rank": meta["rank_recorded_mouse"],
                    "PartnerRank": meta["partner_rank"],
                    "Type": "CagemateRank",
                    "session_duration_sec": time[-1] - time[0],
                    "n_breaths_total": len(bm.inhaleOnsets),
                    "WindowSec": window_sec,
                    "session_stamp": meta["session_stamp"],
                }
            )
            all_rows.append(row)

    if not all_rows:
        print("\nNo rows collected.")
        return pd.DataFrame()

    master_df = pd.DataFrame(all_rows).reset_index(drop=True)
    print(f"\nRank cagemate window matrix: {len(master_df)} rows x {len(master_df.columns)} cols")
    print(f"   Trials / recordings : {master_df['Recording'].nunique()}")
    print(f"   Recorded subjects    : {master_df['Subject'].nunique()}")
    print(f"   Dyads                : {master_df['DyadID'].nunique()}")
    print(f"   Rank:\n{master_df['Rank'].value_counts()}")
    return master_df