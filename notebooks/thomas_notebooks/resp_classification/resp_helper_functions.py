# ======================================================
# Core libraries for respiration + social behavior analysis
# ======================================================

# --- Core Python ---
import os
import h5py
import re

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



def load_clean_resp_signal(h5_file, target_rate=100):
    """
    Loads, filters, downsamples respiration from .h5, returns cleaned signal, time vector, and metadata.
    """
    try:
        with h5py.File(h5_file, 'r') as f:
            resp = f['resp'][:].flatten()

            # Load metadata if available
            metadata = {}
            if 'resp_metadata' in f:
                metadata.update(dict(f['resp_metadata'].attrs))
            if 'ekg_metadata' in f:
                metadata.update(dict(f['ekg_metadata'].attrs))
            if 'metadata' in f:
                metadata.update(dict(f['metadata'].attrs))

            # Estimate sampling frequency
            if 'sampling_frequency' in metadata:
                fs = metadata['sampling_frequency']
            else:
                duration_sec = metadata.get('duration_sec', None)
                fs = len(resp) / duration_sec if duration_sec else 20000.0

            duration_sec = metadata.get('duration_sec', len(resp) / fs)

            # some useful debug statements to get idea of original vs filtered rec
            print("original resp len:", len(resp))
            print("original fs:", fs)
            print("duration_sec:", duration_sec)
            print("expected duration from len/fs:", len(resp) / fs)

    except Exception as e:
        print(f"Error loading {h5_file}: {e}")
        return None, None, None, None

    # Pre-filter before downsampling
    nyquist = fs / 2
    norm_cutoff = (target_rate / 2) / nyquist
    b, a = butter(N=4, Wn=norm_cutoff, btype='low')
    filtered_resp = filtfilt(b, a, resp)

    # Downsample
    downsample_factor = int(fs // target_rate)
    downsampled = resample_poly(filtered_resp, up=1, down=downsample_factor)

    # Bandpass filter with neurokit
    rsp_cleaned = nk.signal_filter(
        downsampled,
        lowcut=0.1,
        highcut=20,
        method="butterworth",
        sampling_rate=target_rate,
        order=2
    )

    # Generate matching time vector
    time_vector = np.arange(len(rsp_cleaned)) / target_rate

    return rsp_cleaned, time_vector, target_rate, metadata




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

def load_clean_boris(csv_path):
    """
    Load a BORIS CSV file and standardize columns for behavior alignment.

    Returns
    -------
    df : DataFrame
        Columns: ['Behavior', 'Subject', 'Start', 'Stop', 'Duration']
        Keeps only 'subject' rows and social behaviors.
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

    # --- Keep only subject-initiated behaviors ---
    df = df[df["Subject"] == "subject"]

    # --- Focus on relevant social behaviors ---
    behaviors_keep = ["facial sniffing", "body sniffing", "anogenital sniffing"]
    df = df[df["Behavior"].isin(behaviors_keep)]

    # --- Sort chronologically ---
    df = df.sort_values("Start").reset_index(drop=True)

    return df



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
# Pad bouts shorter than 5 seconds symmetrically around their
# center, and drop bouts shorter than 1 second. If bouts are 
# longer than 5 seconds they are cropped symmetrically.
# ============================================================

def standardize_bouts_to_target(boris_df, signal_start, signal_end,
                                target_dur=5.0, min_dur=1.0):
    """
    Make every bout exactly target_dur seconds when possible.
    - Drop bouts shorter than min_dur
    - Pad shorter bouts symmetrically
    - Crop longer bouts symmetrically
    - Clamp to session boundaries
    """
    boris_df = boris_df[boris_df["Duration"] >= min_dur].copy()

    for idx in boris_df.index:
        start = boris_df.at[idx, "Start"]
        stop = boris_df.at[idx, "Stop"]
        center = (start + stop) / 2.0

        new_start = center - target_dur / 2.0
        new_stop = center + target_dur / 2.0

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
# Core helper: extract per-bout features from an already-fitted bmObject
# =============================================================================

def extract_bout_features(bm, bout_start_sec, bout_stop_sec):
    """
    Given a fitted bmObject (rodentAirflow, estimateAllFeatures already called),
    extract and aggregate all respiratory features for breaths whose inhale onset
    falls within [bout_start_sec, bout_stop_sec].

    Parameters
    ----------
    bm               : fitted bmObject instance
    bout_start_sec   : float, bout start in seconds
    bout_stop_sec    : float, bout stop in seconds

    Returns
    -------
    dict of aggregated features, or None if fewer than 2 breaths found in window
    """

    # Convert bout times to samples
    start_samp = bout_start_sec * bm.srate
    stop_samp  = bout_stop_sec  * bm.srate

    # Find breath indices whose inhale onset falls inside this bout
    inhale_onsets = np.array(bm.inhaleOnsets)
    bout_mask = (inhale_onsets >= start_samp) & (inhale_onsets <= stop_samp)
    bout_inds = np.where(bout_mask)[0]

    if len(bout_inds) < 2:
        # Not enough breaths to compute meaningful features
        return None

    n_breaths = len(bout_inds)

    # ------------------------------------------------------------------ #
    # Breathing rate & IBI
    # Computed from consecutive inhale onsets within the bout
    # ------------------------------------------------------------------ #
    bout_inhale_onsets = inhale_onsets[bout_inds]
    ibis_samp = np.diff(bout_inhale_onsets)             # inter-breath intervals in samples
    ibis_sec  = ibis_samp / bm.srate

    mean_ibi          = np.mean(ibis_sec)
    breathing_rate    = 1.0 / mean_ibi if mean_ibi > 0 else np.nan
    cv_breathing_rate = np.std(ibis_sec) / mean_ibi if mean_ibi > 0 else np.nan

    # ------------------------------------------------------------------ #
    # Inhale durations
    # ------------------------------------------------------------------ #
    inhale_durs = bm.inhaleDurations[0, bout_inds]     # already in seconds
    inhale_durs = inhale_durs[~np.isnan(inhale_durs)]

    mean_inhale_dur = np.nanmean(inhale_durs) if len(inhale_durs) > 0 else np.nan
    cv_inhale_dur   = np.nanstd(inhale_durs) / mean_inhale_dur if mean_inhale_dur and mean_inhale_dur > 0 else np.nan

    # ------------------------------------------------------------------ #
    # Exhale durations
    # ------------------------------------------------------------------ #
    # Exhale onsets are indexed the same as inhale onsets (one exhale per breath cycle)
    exhale_durs = bm.exhaleDurations[0, bout_inds]
    exhale_durs_clean = exhale_durs[~np.isnan(exhale_durs)]

    mean_exhale_dur = np.nanmean(exhale_durs_clean) if len(exhale_durs_clean) > 0 else np.nan
    cv_exhale_dur   = np.nanstd(exhale_durs_clean) / mean_exhale_dur if mean_exhale_dur and mean_exhale_dur > 0 else np.nan

    # ------------------------------------------------------------------ #
    # Inhale/Exhale ratio
    # ------------------------------------------------------------------ #
    ie_ratio = mean_inhale_dur / mean_exhale_dur if (mean_inhale_dur and mean_exhale_dur and mean_exhale_dur > 0) else np.nan

    # ------------------------------------------------------------------ #
    # Peak flows  (rodentAirflow only)
    # ------------------------------------------------------------------ #
    peak_insp_flows = bm.peakInspiratoryFlows[bout_inds]
    peak_exp_flows  = bm.troughExpiratoryFlows[bout_inds]

    mean_peak_insp_flow = np.nanmean(peak_insp_flows)
    mean_peak_exp_flow  = np.nanmean(peak_exp_flows)
    cv_peak_insp_flow   = np.nanstd(peak_insp_flows) / mean_peak_insp_flow if mean_peak_insp_flow and mean_peak_insp_flow != 0 else np.nan

    # ------------------------------------------------------------------ #
    # Breath volumes  (rodentAirflow only)
    # ------------------------------------------------------------------ #
    inhale_vols = bm.inhaleVolumes[0, bout_inds]
    exhale_vols = bm.exhaleVolumes[0, bout_inds]
    inhale_vols_clean = inhale_vols[~np.isnan(inhale_vols)]
    exhale_vols_clean = exhale_vols[~np.isnan(exhale_vols)]

    mean_inhale_vol  = np.nanmean(inhale_vols_clean) if len(inhale_vols_clean) > 0 else np.nan
    mean_exhale_vol  = np.nanmean(exhale_vols_clean) if len(exhale_vols_clean) > 0 else np.nan
    mean_tidal_vol   = (mean_inhale_vol + mean_exhale_vol) if not (np.isnan(mean_inhale_vol) or np.isnan(mean_exhale_vol)) else np.nan
    cv_tidal_vol     = np.nanstd(inhale_vols_clean) / mean_inhale_vol if mean_inhale_vol and mean_inhale_vol > 0 else np.nan
    minute_vent      = breathing_rate * mean_tidal_vol if (not np.isnan(breathing_rate) and not np.isnan(mean_tidal_vol)) else np.nan

    # ------------------------------------------------------------------ #
    # Inhale pause durations & duty cycles
    # ------------------------------------------------------------------ #
    inh_pause_durs = bm.inhalePauseDurations[0, bout_inds]   # already in seconds
    inh_pause_durs_clean = inh_pause_durs[~np.isnan(inh_pause_durs)]

    pct_inhale_pause    = len(inh_pause_durs_clean) / n_breaths
    mean_inh_pause_dur  = np.nanmean(inh_pause_durs_clean) if len(inh_pause_durs_clean) > 0 else 0.0
    cv_inh_pause_dur    = np.nanstd(inh_pause_durs_clean) / mean_inh_pause_dur if mean_inh_pause_dur > 0 else np.nan

    # ------------------------------------------------------------------ #
    # Exhale pause durations & duty cycles
    # ------------------------------------------------------------------ #
    exh_pause_durs = bm.exhalePauseDurations[0, bout_inds]
    exh_pause_durs_clean = exh_pause_durs[~np.isnan(exh_pause_durs)]

    pct_exhale_pause    = len(exh_pause_durs_clean) / n_breaths
    mean_exh_pause_dur  = np.nanmean(exh_pause_durs_clean) if len(exh_pause_durs_clean) > 0 else 0.0
    cv_exh_pause_dur    = np.nanstd(exh_pause_durs_clean) / mean_exh_pause_dur if mean_exh_pause_dur > 0 else np.nan

    # ------------------------------------------------------------------ #
    # Duty cycles  (fraction of mean IBI spent in each phase)
    # ------------------------------------------------------------------ #
    inhale_duty_cycle       = mean_inhale_dur        / mean_ibi if mean_ibi > 0 else np.nan
    exhale_duty_cycle       = mean_exhale_dur        / mean_ibi if mean_ibi > 0 else np.nan
    inh_pause_duty_cycle    = (mean_inh_pause_dur * pct_inhale_pause) / mean_ibi if mean_ibi > 0 else np.nan
    exh_pause_duty_cycle    = (mean_exh_pause_dur * pct_exhale_pause) / mean_ibi if mean_ibi > 0 else np.nan

    # ------------------------------------------------------------------ #
    # Pack into dict
    # ------------------------------------------------------------------ #
    return {
        # bookkeeping
        "n_breaths"                         : n_breaths,

        # rate & timing
        "breathing_rate_hz"                 : breathing_rate,
        "mean_ibi_sec"                      : mean_ibi,
        "cv_breathing_rate"                 : cv_breathing_rate,

        # inhale phase
        "mean_inhale_dur_sec"               : mean_inhale_dur,
        "cv_inhale_dur"                     : cv_inhale_dur,

        # exhale phase
        "mean_exhale_dur_sec"               : mean_exhale_dur,
        "cv_exhale_dur"                     : cv_exhale_dur,

        # inhale/exhale ratio
        "ie_ratio"                          : ie_ratio,

        # peak flows
        "mean_peak_insp_flow"               : mean_peak_insp_flow,
        "mean_peak_exp_flow"                : mean_peak_exp_flow,
        "cv_peak_insp_flow"                 : cv_peak_insp_flow,

        # volumes
        "mean_inhale_vol"                   : mean_inhale_vol,
        "mean_exhale_vol"                   : mean_exhale_vol,
        "mean_tidal_vol"                    : mean_tidal_vol,
        "cv_tidal_vol"                      : cv_tidal_vol,
        "minute_ventilation"                : minute_vent,

        # inhale pause
        "pct_breaths_with_inhale_pause"     : pct_inhale_pause,
        "mean_inhale_pause_dur_sec"         : mean_inh_pause_dur,
        "cv_inhale_pause_dur"               : cv_inh_pause_dur,
        "inhale_pause_duty_cycle"           : inh_pause_duty_cycle,

        # exhale pause
        "pct_breaths_with_exhale_pause"     : pct_exhale_pause,
        "mean_exhale_pause_dur_sec"         : mean_exh_pause_dur,
        "cv_exhale_pause_dur"               : cv_exh_pause_dur,
        "exhale_pause_duty_cycle"           : exh_pause_duty_cycle,

        # duty cycles
        "inhale_duty_cycle"                 : inhale_duty_cycle,
        "exhale_duty_cycle"                 : exhale_duty_cycle,
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
    window_dur=5.0,
    n_windows=100,
    seed=42,
    allow_partial_if_short=False,
):
    """
    Sample up to n_windows non-overlapping fixed-duration windows from a session.

    Parameters
    ----------
    time : np.ndarray
        Session time vector in seconds.
    window_dur : float
        Window duration in seconds.
    n_windows : int
        Maximum number of windows to sample.
    seed : int
        Random seed for reproducibility.
    allow_partial_if_short : bool
        If True and session is shorter than one full window, return one
        clamped session-spanning window. If False, return [].

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

    n_use = min(n_windows, len(possible_starts))

    chosen_starts = rng.choice(
        possible_starts,
        size=n_use,
        replace=False
    )

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