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

# Breathmetrics: bmObject and all BreathMetrics feature code live in src/rsp (moved 2026-11-24).
# Re-exported here so `from resp_helper_functions import *` keeps working in the notebooks.
import sys
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from src.rsp.breathmetrics import bmObject
from src.rsp.bm_features import *  # noqa: F401,F403  fit_bm_session, extract_*_features, windows, plots

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
    initiator="subject",
):
    """
    Build cohort-aware subject/trial exclusion sets before BreathMetrics extraction.

    Subject exclusions are keyed by cohort (e.g. Aim1 1_1 is independent of BLA 1_1).

    initiator : str or None
        Which BORIS actor to count for the low-sniff / imbalance gates.
        "subject" = subject-initiated bouts only (legacy); None = BOTH actors
        (subject + social_agent), matching build_feature_matrix's subject_only=False
        extraction. Rank uses None (both actors) so the gate is consistent with the
        both-actor feature windows.

    Returns
    -------
    exclude_subjects_by_cohort : dict[str, set[str]]
    exclude_trials : set[str]
    report : dict
    """
    drop_aim1 = {normalize_subject_id(s) for s in (drop_aim1_subjects or [])}
    imbalanced_pairs = boris_imbalanced_subjects(
        behavior_df, imbalance_ratio=imbalance_ratio, initiator=initiator,
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
        behavior_df, min_sniffs=min_sniffs_per_trial, initiator=initiator,
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
