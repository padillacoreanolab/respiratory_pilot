"""Synthetic rodent airflow-like signals with exactly known breath landmarks.

Each cycle: inhale lobe  A*sin(pi*t/Ti)            (peak at Ti/2)
            inhale pause zeros for Tpi            (optional)
            exhale lobe  -(A*Ti/Te)*sin(pi*t/Te)  (equal area -> zero net flow per cycle)
            exhale pause zeros for Tpe            (optional)
Truth indices refer to the returned signal; onsets are the first sample of each lobe.
"""
import numpy as np
from scipy.signal import butter, filtfilt


def make_signal(fs, cycles, noise_sd=0.0, drift=None, dc=0.0, seed=0, lead_s=0.3):
    """cycles: list of dicts with keys Ti, Te, A and optional Tpi, Tpe (seconds).
    drift: list of (amplitude, freq_hz) sinusoids added to the signal.
    Returns signal, truth (dict of int arrays + 'pause' info), time_s."""
    rng = np.random.default_rng(seed)
    parts = [np.zeros(int(lead_s * fs))]
    pos = len(parts[0])
    tr = {k: [] for k in ["inhale_onset", "peak", "inhale_pause_onset", "exhale_onset", "trough",
                          "exhale_pause_onset", "amp", "inhale_area", "exhale_area"]}
    for c in cycles:
        Ti, Te, A = c["Ti"], c["Te"], c["A"]
        Tpi, Tpe = c.get("Tpi", 0.0), c.get("Tpe", 0.0)
        ni, ne = max(int(round(Ti * fs)), 3), max(int(round(Te * fs)), 3)
        npi, npe = int(round(Tpi * fs)), int(round(Tpe * fs))
        inh = A * np.sin(np.pi * np.arange(ni) / ni)
        exh = -(A * ni / ne) * np.sin(np.pi * np.arange(ne) / ne)
        tr["inhale_onset"].append(pos)
        tr["peak"].append(pos + int(np.argmax(inh)))
        tr["inhale_pause_onset"].append(pos + ni if npi > 0 else -1)
        tr["exhale_onset"].append(pos + ni + npi)
        tr["trough"].append(pos + ni + npi + int(np.argmin(exh)))
        tr["exhale_pause_onset"].append(pos + ni + npi + ne if npe > 0 else -1)
        tr["amp"].append(A)
        tr["inhale_area"].append(np.abs(inh).sum() / fs * 1000)
        tr["exhale_area"].append(np.abs(exh).sum() / fs * 1000)
        parts += [inh, np.zeros(npi), exh, np.zeros(npe)]
        pos += ni + npi + ne + npe
    parts.append(np.zeros(int(lead_s * fs)))
    x = np.concatenate(parts)
    t = np.arange(len(x)) / fs
    if drift:
        for amp, f in drift:
            x = x + amp * np.sin(2 * np.pi * f * t)
    x = x + dc
    if noise_sd > 0:
        nz = rng.normal(0, 1, len(x))
        b, a = butter(2, min(20.0 / (fs / 2), 0.99))
        nz = filtfilt(b, a, nz)
        x = x + noise_sd * nz / nz.std()
    return x, {k: np.asarray(v) for k, v in tr.items()}, t


def regular(n, rate_hz, A=100.0, ti_frac=0.4, **kw):
    T = 1.0 / rate_hz
    return [dict(Ti=ti_frac * T, Te=(1 - ti_frac) * T, A=A, **kw) for _ in range(n)]


def match(truth, det, tol):
    """Greedy one-to-one matching within tol samples. Returns (tp, fp, fn, errors, matched_pairs)."""
    truth = np.sort(np.asarray(truth, float))
    det = np.sort(np.asarray(det, float)[np.isfinite(det)])
    used = np.zeros(len(det), bool)
    errs, pairs = [], []
    for ti, t in enumerate(truth):
        if not len(det):
            break
        d = np.abs(det - t)
        d[used] = np.inf
        j = int(np.argmin(d))
        if d[j] <= tol:
            used[j] = True
            errs.append(det[j] - t)
            pairs.append((ti, j))
    tp = len(errs)
    return tp, len(det) - tp, len(truth) - tp, np.asarray(errs), pairs
