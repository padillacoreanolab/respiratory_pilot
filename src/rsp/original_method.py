"""The lab's original respiration processing and peak finder (load_resp_simple + phase-wide find_peaks).

Reproduces scripts/compare_original_dan_noto.py exactly: raw 20 kHz pressure, median-centered,
clipped at +/-10 MAD, order-4 50 Hz low-pass (filtfilt), resample_poly to 100 Hz, NeuroKit
order-2 0.1-20 Hz band-pass, then scipy find_peaks with an 80 ms (8-sample) minimum spacing and
no prominence threshold, run separately within each phase (recording boundaries).
"""
import numpy as np, h5py, neurokit2 as nk
from scipy.signal import butter, filtfilt, resample_poly, find_peaks

FS = 100
DISTANCE = 8  # samples at 100 Hz = 80 ms


def original_signal(h5_path):
    """Return (time_s, signal, phase_bounds_s) on the original uniform 100 Hz grid."""
    with h5py.File(h5_path, 'r') as h:
        raw = h['respiration/raw/pressure_adc'][:].astype(float)
        ticks = h['respiration/raw/acquisition_ticks'][:]
        fs = float(h.attrs['clock_rate_hz']); origin = int(h.attrs['recording_tick_origin'])
        bounds = h['candidate_recording_boundaries_s'][:]
    if fs != 20000 or not np.all(np.diff(ticks) == 1) or not np.isfinite(raw).all():
        raise ValueError(f'{h5_path}: clock is not a gap-free 20 kHz sequence; the uniform original time base would be wrong')
    offset = (int(ticks[0])-origin)/fs
    centered = raw-np.median(raw); mad = np.median(abs(centered))
    clipped = np.clip(centered, -10*mad, 10*mad) if mad > 0 else centered
    b, a = butter(4, 50/(fs/2), btype='low')
    y = nk.signal_filter(resample_poly(filtfilt(b, a, clipped), 1, 200), lowcut=.1, highcut=20, sampling_rate=FS, order=2)
    return offset+np.arange(len(y))/FS, np.asarray(y, float), bounds


def original_candidates(t, y, bounds):
    """Phase-wide find_peaks(distance=8); returns (sample index, prominence) for every peak found.

    Prominence is computed only so a reviewer can optionally raise a floor; with floor 0 the
    result is exactly the original detector's output.
    """
    idx, prom = [], []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        m = np.flatnonzero((t >= lo)&(t < hi))
        if not len(m): continue
        x = y[m]-y[m].mean()
        ix, props = find_peaks(x, distance=DISTANCE, prominence=0)
        idx.append(m[ix]); prom.append(props['prominences'])
    return np.concatenate(idx).astype(int), np.concatenate(prom)


def detect_original(t, y, bounds, floor=0.):
    """Original detector plus an optional prominence floor (0 = original)."""
    idx, prom = original_candidates(t, y, bounds)
    return idx[prom >= floor]
