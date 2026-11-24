"""
Respiration processing: raw intranasal pressure -> clean 100 Hz trace -> breaths.

References: Wesson et al. 2008 (Chem Senses); Janke et al. 2022 (iScience).

Usage:
    python -m src.rsp.resp_processing <session.h5> [start_s end_s]
"""
import sys

import h5py
import numpy as np
import neurokit2 as nk
from scipy.signal import butter, filtfilt, resample_poly, find_peaks

RAW_FS = 20000      # Trodes ECU analog sampling rate (Hz)
TARGET_FS = 100     # processed respiration rate (Hz)
MAX_RATE_HZ = 12    # fastest breath/sniff rate accepted (Hz)


def load_raw_resp(h5_path):
    """
    Read the raw respiration trace from an H5.

    Args:
        h5_path: H5 with /analog/*Ain1 (current layout) or /resp (older Aim1 layout)
    Returns:
        1-D float array at RAW_FS
    """
    with h5py.File(h5_path, "r") as f:
        if "analog" in f:
            keys = [k for k in f["analog"] if "Ain1" in k]
            if keys:
                return f["analog"][keys[0]][()]["voltage"].astype(float)
        return f["resp"][()].astype(float).ravel()


def preprocess_resp(resp_raw, raw_fs=RAW_FS, target_fs=TARGET_FS):
    """
    Clean and downsample a raw respiration trace.

    Args:
        resp_raw: raw trace (load_raw_resp output)
        raw_fs: input sampling rate (Hz)
        target_fs: output sampling rate (Hz)
    Returns:
        (resp, t): clean trace and its time in seconds from the first sample
    """
    resp = np.asarray(resp_raw, dtype=float).copy()

    # 1) Median center IOT remove the sensor's DC offset (robust to artifacts)
    resp -= np.median(resp)

    # 2) Clip to +/-10 MAD IOT cap movement/chewing spikes before filtering
    mad = np.median(np.abs(resp))
    if mad > 0:
        resp = np.clip(resp, -10 * mad, 10 * mad)

    # 3) Low-pass at 50 Hz (4th-order Butterworth, zero-phase) IOT anti-alias
    #    before downsampling to 100 Hz
    b, a = butter(4, (target_fs / 2) / (raw_fs / 2), btype="low")
    resp = filtfilt(b, a, resp)

    # 4) Downsample 20 kHz -> 100 Hz IOT reduce data size; breathing is < 20 Hz
    down = int(raw_fs // target_fs)
    resp = resample_poly(resp, up=1, down=down)
    t = np.arange(len(resp_raw))[::down] / raw_fs

    # 5) Band-pass 0.1-20 Hz (2nd-order Butterworth, zero-phase) IOT remove
    #    baseline drift and fast noise while keeping breathing and sniffing
    resp = nk.signal_filter(resp, lowcut=0.1, highcut=20,
                            sampling_rate=target_fs, order=2)
    return resp, t


def detect_breaths(resp, t, start_s, end_s, fs=TARGET_FS, max_rate_hz=MAX_RATE_HZ,
                   min_prominence_frac=None):
    """
    Find breath peaks in one time window.

    Args:
        resp, t: preprocess_resp output
        start_s, end_s: window [start_s, end_s) in seconds
        fs: sampling rate of resp (Hz)
        max_rate_hz: fastest rate counted as breathing
        min_prominence_frac: if set, drop peaks whose prominence is below this
            fraction of the window's median peak prominence (None = off;
            recommended 0.2, matching breathmetrics.SRC_MIN_PROMINENCE_FRAC)
    Returns:
        peak indices into resp (empty if the window is < 1 s)
    """
    # 1) Cut out the window IOT detect breaths separately per phase
    in_win = np.flatnonzero((t >= start_s) & (t < end_s))
    if len(in_win) < fs:
        return np.array([], dtype=int)
    seg = resp[in_win]

    # 2) Subtract the window mean IOT center this window at zero
    seg = seg - seg.mean()

    # 3) Peaks at least 1/max_rate_hz apart (80 ms at 12 Hz) IOT count each
    #    breath once
    peaks, props = find_peaks(seg, distance=int((1.0 / max_rate_hz) * fs), prominence=0)

    # 4) Optional: drop peaks far smaller than a typical breath IOT reject
    #    ripple/noise peaks (~5% of peaks without this, see breathmetrics.py)
    if min_prominence_frac is not None and len(peaks):
        prom = props["prominences"]
        peaks = peaks[prom >= min_prominence_frac * np.median(prom)]
    return in_win[peaks]


def breathing_rate(t, peaks):
    """
    Mean instantaneous breathing rate.

    Args:
        t: time array (s)
        peaks: detect_breaths output
    Returns:
        mean of 1 / inter-breath interval (Hz), NaN if < 2 peaks
    """
    if len(peaks) < 2:
        return np.nan
    ibi = np.diff(t[peaks])
    return float(np.mean(1.0 / ibi))


if __name__ == "__main__":
    h5_path = sys.argv[1]
    resp, t = preprocess_resp(load_raw_resp(h5_path))
    start, end = (float(sys.argv[2]), float(sys.argv[3])) if len(sys.argv) > 3 \
        else (t[0], t[-1] + 1)
    peaks = detect_breaths(resp, t, start, end)
    print(f"{len(peaks)} breaths in {start:.0f}-{end:.0f} s, "
          f"rate {breathing_rate(t, peaks):.2f} Hz")