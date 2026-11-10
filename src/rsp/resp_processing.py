"""
Respiration processing

Citation (R34 hub decision): intranasal pressure recording in freely moving
mice, Wesson et al. 2008 (Chem Senses), and Janke et al. 2022 (iScience).
The hub notes that the exact filter settings are lab choices, not copied from
a paper.

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
MAX_RATE_HZ = 12    # fastest breath/sniff rate we accept (Hz)


def load_raw_resp(h5_path):
    """Return the raw 20 kHz respiration trace from one H5 file."""
    with h5py.File(h5_path, "r") as f:
        # New + BLA layout: /analog/<name>.analog_ECU_Ain1 with a 'voltage' field
        if "analog" in f:
            keys = [k for k in f["analog"] if "Ain1" in k]
            if keys:
                return f["analog"][keys[0]][()]["voltage"].astype(float)
        # Older Aim1 layout: /resp, shape (N,) or (N, 1)
        return f["resp"][()].astype(float).ravel()


def preprocess_resp(resp_raw, raw_fs=RAW_FS, target_fs=TARGET_FS):
    """Raw 20 kHz pressure trace -> clean 100 Hz trace + time (s)."""
    resp = np.asarray(resp_raw, dtype=float).copy()

    # 1) Median center IOT remove the sensor's DC offset (median is not
    #    pulled by large artifacts the way the mean is).
    resp -= np.median(resp)

    # 2) Clip to +/-10 MAD IOT cap movement/chewing spikes so they don't
    #    ring through the filters below. (MAD = median absolute deviation
    #    of the centered signal.)
    mad = np.median(np.abs(resp))
    if mad > 0:
        resp = np.clip(resp, -10 * mad, 10 * mad)

    # 3) Low-pass at 50 Hz (4th-order Butterworth, zero-phase) IOT
    #    anti-alias before downsampling to 100 Hz (50 Hz = new Nyquist).
    b, a = butter(4, (target_fs / 2) / (raw_fs / 2), btype="low")
    resp = filtfilt(b, a, resp)

    # 4) Downsample 20 kHz -> 100 Hz IOT shrink data 200x; breathing
    #    (< 20 Hz) is fully kept at 100 Hz.
    down = int(raw_fs // target_fs)
    resp = resample_poly(resp, up=1, down=down)
    t = np.arange(len(resp_raw))[::down] / raw_fs

    # 5) Band-pass 0.1-20 Hz (2nd-order Butterworth, zero-phase, NeuroKit)
    #    IOT remove slow baseline drift (< 0.1 Hz) and fast noise (> 20 Hz)
    #    while keeping resting breaths (~2-5 Hz) and sniffing (up to ~12 Hz).
    resp = nk.signal_filter(resp, lowcut=0.1, highcut=20,
                            sampling_rate=target_fs, order=2)
    return resp, t


def detect_breaths(resp, t, start_s, end_s, fs=TARGET_FS,
                   max_rate_hz=MAX_RATE_HZ):
    """Breath peak indices (into resp) inside one phase [start_s, end_s)."""
    # 1) Cut out the phase IOT detect breaths separately per phase
    #    (e.g. baseline vs social), as in her notebooks.
    in_win = np.flatnonzero((t >= start_s) & (t < end_s))
    if len(in_win) < fs:  # less than 1 s of data
        return np.array([], dtype=int)
    seg = resp[in_win]

    # 2) Subtract the window mean IOT center this phase at zero.
    seg = seg - seg.mean()

    # 3) Find peaks at least 1/12 s apart (int(8.33) = 8 samples = 80 ms)
    #    IOT count each breath once; anything faster than 12 Hz is treated
    #    as noise, not a breath. No height threshold is used.
    peaks, _ = find_peaks(seg, distance=int((1.0 / max_rate_hz) * fs))
    return in_win[peaks]


def breathing_rate(t, peaks):
    """Mean instantaneous rate (Hz) = mean of 1 / inter-breath interval."""
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
