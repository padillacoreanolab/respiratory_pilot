"""Breath-amplitude signal quality and a PROPOSED dropout rule (not yet applied to detection).

Amplitude = 1 s rolling p90-p10 of 0.5 Hz high-passed pressure. A dropout is a stretch where
amplitude stays below `fraction` of a rolling reference (median amplitude over `reference_s`)
for at least `min_s`. The rolling reference tolerates gain changes; it must be longer than
twice the longest expected dropout, or the dropout drags the reference down with it.
"""
import numpy as np
from scipy import signal
from scipy.ndimage import percentile_filter, median_filter


def breath_amplitude(pressure, fs, step=40):
    """Return (sample index, amplitude) at fs/step Hz."""
    hp = signal.sosfiltfilt(signal.butter(2, .5, btype='highpass', fs=fs, output='sos'), np.asarray(pressure, float))
    size = int(fs) | 1
    amp = percentile_filter(hp, 90, size=size, mode='nearest')-percentile_filter(hp, 10, size=size, mode='nearest')
    idx = np.arange(0, len(hp), step)
    return idx, amp[idx]


def propose_dropouts(pressure, fs, fraction=.2, reference_s=120., min_s=1., merge_s=.5, pad_s=.5, step=40):
    """Return (list of (start_sample, stop_sample), amplitude idx, amplitude, reference)."""
    idx, amp = breath_amplitude(pressure, fs, step)
    rate = fs/step
    ref = median_filter(amp, size=int(reference_s*rate) | 1, mode='nearest')
    low = amp < fraction*ref
    edges = np.diff(np.r_[False, low, False].astype(int))
    runs = [[a, b] for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]
    merged = []
    for a, b in runs:  # join runs separated by brief recoveries
        if merged and a-merged[-1][1] <= merge_s*rate: merged[-1][1] = b
        else: merged.append([a, b])
    out = []
    for a, b in merged:
        if (b-a)/rate < min_s: continue
        s0 = max(0, int(idx[a]-pad_s*fs)); s1 = min(len(pressure), int(idx[min(b, len(idx)-1)]+pad_s*fs))
        out.append((s0, s1))
    return out, idx, amp, ref
