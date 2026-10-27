"""Adaptive local-prominence breath detection for rodent respiration (airflow/pressure) signals.

Why this exists
---------------
The BreathMetrics extrema step (``breathmetrics.bmObject.findExtrema``; MATLAB
``findRespiratoryExtrema.m``) only accepts a peak above ``mean + std/2`` of the WHOLE
session and a trough below ``mean - std/2``. Breaths that are small relative to the
loudest breathing in the session, or that ride on sub-second drift, never pass, and a
missed trough makes two real breaths look like "two peaks in a row", one of which is
then deleted. BreathMetrics onsets are then defined as crossings of the session mean,
which breaths riding on drift may never cross.

This module replaces those steps:

1. Validity mask - NaNs, recording-time gaps, near-flat sensor dropouts and gross
   artifacts become invalid samples; detection runs separately in each valid segment,
   so no breath or interval is ever built across a gap.
2. Extrema - ``scipy.signal.find_peaks`` with a minimum *local* prominence:
   ``max(prominence_frac * local 1 s p90-p10 range, noise_mult * noise floor)``, a
   minimum peak spacing, and a minimum half-prominence width (spike rejection).
3. Cycles - a trough is the minimum between consecutive accepted peaks. Because both
   peaks passed the prominence test, the trough between them is at least that deep
   below the lower peak (checked explicitly; failures are rejected with a reason).
4. Local baseline - cycle-mean baseline (mean of the signal over each peak-to-peak
   cycle, interpolated). For an airflow signal the integral over a full cycle is zero
   (inhaled = exhaled volume), so the cycle mean estimates the zero-flow level locally.
5. Onsets / pauses - per cycle: crossings of the local baseline (MATLAB semantics: last
   at-or-below-baseline sample before the peak), and pauses = runs within a band of the
   baseline whose half-width scales with THAT cycle's amplitude, lasting >= min_pause_s.

Every time index refers to the input array; ``time_s`` (if given) is never modified.
Nothing here is calibrated: volumes are integrated pressure proxies (BreathMetrics units).

The legacy BreathMetrics path is untouched; select this detector with
``fit_bm_session(..., detector="adaptive_prominence")``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks

DETECTOR_NAME = "adaptive_prominence"
DETECTOR_VERSION = "1.0.0"
MIN_SUPPORTED_FS = 40.0  # below this, 12-15 Hz sniffing has < ~3 samples per cycle


@dataclass
class AdaptiveDetectorConfig:
    """All tunable choices. Defaults were chosen on development data (see
    detector_validation/README.md); they are NOT validated biological thresholds."""

    # --- extrema -------------------------------------------------------------------------
    prominence_frac: float = 0.20        # x local robust range (p90-p10 over local_range_win_s)
    noise_mult: float = 2.0              # x noise floor (peak-to-peak of sensor noise); 2 chosen on
                                         # dev data (detector_validation/outputs_v1/selection_note.md)
    min_peak_distance_s: float = 0.06    # minimum peak spacing
    prominence_wlen_s: float | None = 1.0  # window for prominence bases; None = whole segment
    local_range_win_s: float = 1.0
    local_range_pcts: tuple = (10.0, 90.0)
    min_peak_width_s: float = 0.015      # half-prominence width; narrower = spike. Real rank-cohort
                                         # breaths: 0.1st pct 30 ms; a 14 Hz sinusoid ~24 ms
    smooth_ms: float = 10.0              # same running-mean de-noising as BreathMetrics rodent
    scale_highpass_hz: float | None = 1.0  # high-pass used ONLY to find provisional cycles and to
                                         # measure the local range (drift-robust threshold scale);
                                         # landmarks/features use the unfiltered signal. None = off

    # --- validity mask -------------------------------------------------------------------
    block_s: float = 1.0                 # block size for flat/artifact screening
    block_hop_s: float = 0.25
    flat_rel_range: float = 0.03         # block range < 3% of session median range ...
    flat_acf_max: float = 0.5            # ... AND not periodic (ACF peak 60-500 ms below this) => dropout
    low_amp_rel_range: float = 0.15      # valid but flagged "low_amplitude" below this
    artifact_rel_range: float = 8.0      # block range > 8x session median => artifact
    artifact_pad_s: float = 0.25
    min_segment_s: float = 1.0
    gap_factor: float = 1.5              # time step > gap_factor x median step => gap

    # --- onsets / pauses -----------------------------------------------------------------
    pause_band_frac: float = 0.05        # pause = |y| <= this x the cycle's peak-trough amplitude
    pause_noise_mult: float = 1.5        # ... but at least this x noise floor
    min_pause_s: float = 0.025

    # --- polarity ------------------------------------------------------------------------
    polarity: int = 1                    # +1: inhalation is positive (BreathMetrics convention)

    def to_dict(self):
        return asdict(self)


@dataclass
class BreathDetectionResult:
    fs: float
    config: AdaptiveDetectorConfig
    smoothed: np.ndarray                 # polarity-applied, smoothed input
    baseline: np.ndarray                 # local cycle-mean baseline (NaN where invalid)
    corrected: np.ndarray                # smoothed - baseline (NaN where invalid)
    valid: np.ndarray                    # bool per sample
    invalid_reason: np.ndarray           # '' or reason per sample (object array)
    low_amplitude: np.ndarray            # bool per sample
    segments: list                       # [(start, stop_exclusive), ...]
    noise_ptp: float
    breaths: pd.DataFrame                # one row per accepted inhale peak / cycle
    rejected: pd.DataFrame               # candidate peaks not accepted, with reason
    time_s: np.ndarray | None = None
    notes: list = field(default_factory=list)


# ----------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------
def running_mean(x, n):
    """Centered running mean; edges average the available samples (no zero padding)."""
    x = np.asarray(x, float)
    n = max(int(n), 1)
    s = n // 2
    c = np.concatenate([[0.0], np.cumsum(x)])
    p = np.arange(len(x))
    lo = np.clip(p - s, 0, len(x))
    hi = np.clip(p - s + n, 0, len(x))
    return (c[hi] - c[lo]) / (hi - lo)


def _runs(mask):
    """(start, stop_exclusive) of True runs."""
    m = np.r_[False, np.asarray(mask, bool), False]
    d = np.diff(m.astype(int))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def _acf_peak(seg, fs, lo_s=0.06, hi_s=0.5):
    seg = seg - seg.mean()
    if seg.std() == 0 or len(seg) < int(hi_s * fs) + 2:
        return 0.0
    ac = np.correlate(seg, seg, "full")[len(seg) - 1:]
    ac = ac / ac[0]
    return float(ac[int(lo_s * fs):int(hi_s * fs)].max())


def _block_ranges(x, fs, block_s, hop_s, pcts=(5, 95)):
    w, h = max(int(block_s * fs), 2), max(int(hop_s * fs), 1)
    starts = np.arange(0, max(len(x) - w, 0) + 1, h)
    r = np.full(len(starts), np.nan)
    for k, s in enumerate(starts):
        b = x[s:s + w]
        b = b[np.isfinite(b)]
        if len(b) > w // 2:
            lo, hi = np.percentile(b, pcts)
            r[k] = hi - lo
    return starts, w, r


def local_range(y, fs, win_s=1.0, pcts=(10.0, 90.0)):
    """Per-sample robust range (p_hi - p_lo) in a centered window, from 0.1 s hops."""
    n = len(y)
    w = max(int(win_s * fs), 3)
    hop = max(int(0.1 * fs), 1)
    centers = np.arange(0, n, hop)
    vals = np.empty(len(centers))
    for k, c in enumerate(centers):
        b = y[max(0, c - w // 2):min(n, c + w // 2 + 1)]
        lo, hi = np.percentile(b, pcts)
        vals[k] = hi - lo
    return np.interp(np.arange(n), centers, vals)


# ----------------------------------------------------------------------------------------
# validity mask
# ----------------------------------------------------------------------------------------
def build_validity_mask(x, fs, cfg, time_s=None):
    """Return valid (bool), reason (object), low_amplitude (bool), segments, noise_ptp, notes."""
    n = len(x)
    reason = np.full(n, "", dtype=object)
    notes = []
    finite = np.isfinite(x)
    reason[~finite] = "missing"
    seg_breaks = []
    if time_s is not None:
        t = np.asarray(time_s, float)
        if len(t) != n:
            raise ValueError("time_s must have the same length as the signal")
        dt = np.diff(t)
        if np.any(dt <= 0):
            raise ValueError("time_s must be strictly increasing")
        med = np.median(dt)
        gaps = np.where(dt > cfg.gap_factor * med)[0]
        seg_breaks = list(gaps + 1)
        if len(gaps):
            notes.append(f"{len(gaps)} recording-time gap(s) > {cfg.gap_factor}x median sample step")
        if abs(1.0 / med - fs) / fs > 0.01:
            notes.append(f"time_s median step implies {1 / med:.3f} Hz but fs={fs}")

    xf = np.where(finite, x, np.nan)
    starts, w, r = _block_ranges(xf, fs, cfg.block_s, cfg.block_hop_s)
    med_r = np.nanmedian(r) if np.any(np.isfinite(r)) else np.nan
    low_amp = np.zeros(n, bool)
    flat_blocks = []
    if np.isfinite(med_r) and med_r > 0:
        rel = r / med_r
        for k, s in enumerate(starts):
            if not np.isfinite(rel[k]):
                continue
            sl = slice(s, s + w)
            if rel[k] > cfg.artifact_rel_range:
                pad = int(cfg.artifact_pad_s * fs)
                a0, a1 = max(0, s - pad), min(n, s + w + pad)
                reason[a0:a1] = np.where(reason[a0:a1] == "", "artifact", reason[a0:a1])
            elif rel[k] < cfg.flat_rel_range:
                b = xf[sl]
                b = b[np.isfinite(b)]
                if _acf_peak(b, fs) < cfg.flat_acf_max:
                    reason[sl] = np.where(reason[sl] == "", "flat_or_dropout", reason[sl])
                    flat_blocks.append(sl)
                else:
                    low_amp[sl] = True
            elif rel[k] < cfg.low_amp_rel_range:
                low_amp[sl] = True

    # noise floor: peak-to-peak of 250 ms pieces inside detected dropouts, else the quietest
    # 1% of 250 ms pieces capped at 1% of the median range (a session without dropouts has
    # no flat stretches; its quietest pieces are shallow breathing, not noise)
    q = max(int(0.25 * fs), 2)
    pieces = []
    for sl in flat_blocks:
        b = xf[sl]
        for i in range(0, len(b) - q + 1, q):
            pc = b[i:i + q]
            if np.all(np.isfinite(pc)):
                pieces.append(np.ptp(pc))
    if pieces:
        noise_ptp = float(np.median(pieces))
        notes.append("noise floor from detected dropout segments")
    else:
        m = len(xf) // q
        pp = np.ptp(xf[: m * q].reshape(m, q), axis=1) if m else np.array([0.0])
        pp = pp[np.isfinite(pp)]
        q1 = np.percentile(pp, 1) if len(pp) else 0.0
        noise_ptp = float(min(q1, 0.01 * med_r)) if np.isfinite(med_r) else float(q1)
        notes.append("noise floor from quietest 1% of 250 ms pieces (capped at 1% median range)")

    valid = reason == ""
    # segments: valid runs, split at time gaps, dropping runs shorter than min_segment_s
    cut = np.zeros(n, bool)
    cut[seg_breaks] = True
    segments = []
    for a, b in _runs(valid):
        bounds = [a] + [c for c in seg_breaks if a < c < b] + [b]
        for s0, s1 in zip(bounds[:-1], bounds[1:]):
            if (s1 - s0) >= cfg.min_segment_s * fs:
                segments.append((int(s0), int(s1)))
            else:
                reason[s0:s1] = np.where(reason[s0:s1] == "", "short_segment", reason[s0:s1])
    valid = reason == ""
    return valid, reason, low_amp & valid, segments, noise_ptp, notes


# ----------------------------------------------------------------------------------------
# extrema within one segment
# ----------------------------------------------------------------------------------------
def _highpass(seg, fs, hz):
    if hz is None:
        return seg
    b, a = butter(2, hz / (fs / 2.0), btype="high")
    if len(seg) <= 3 * max(len(a), len(b)) * 3:
        return seg - np.mean(seg)
    return filtfilt(b, a, seg)


def _detect_peaks(y, fs, cfg, noise_ptp, scale_signal=None):
    """Accepted peak indices (segment-local) + rejected candidate table rows.
    scale_signal: signal on which the local range is measured (default: y itself)."""
    lr = local_range(y if scale_signal is None else scale_signal, fs, cfg.local_range_win_s, cfg.local_range_pcts)
    wlen = None if cfg.prominence_wlen_s is None else max(int(cfg.prominence_wlen_s * fs), 3)
    floor = cfg.noise_mult * noise_ptp
    cand, pr = find_peaks(y, prominence=0, wlen=wlen, width=0, rel_height=0.5)
    need = np.maximum(cfg.prominence_frac * lr[cand], floor)
    prom, width = pr["prominences"], pr["widths"] / fs
    ok_prom = prom >= need
    ok_width = width >= cfg.min_peak_width_s
    keep = ok_prom & ok_width
    # minimum spacing among the survivors; the taller peak wins (scipy find_peaks `distance`)
    dist = max(int(round(cfg.min_peak_distance_s * fs)), 1)
    surv = cand[keep]
    if len(surv):
        only_surv = np.full(len(y), -np.inf)
        only_surv[surv] = y[surv]
        sel, _ = find_peaks(only_surv, distance=dist)
        sel = np.intersect1d(sel, surv)
    else:
        sel = surv
    sel_set = set(sel.tolist())
    rows = []
    for c, p, wd, nd, op, ow in zip(cand, prom, width, need, ok_prom, ok_width):
        if op and ow and c in sel_set:
            continue
        if p < floor * 0.5:      # sub-noise wiggles: not worth recording
            continue
        why = "low_prominence" if not op else ("spike_width" if not ow else "too_close")
        rows.append(dict(idx=int(c), prominence=float(p), required=float(nd), width_s=float(wd), reason=why))
    return np.asarray(sel, int), rows, lr


def _troughs_between(y, pk):
    return np.array([a + int(np.argmin(y[a:b])) for a, b in zip(pk[:-1], pk[1:])], int)


def _validate_cycles(y, pk, need_at, rows):
    """Enforce: every trough between consecutive accepted peaks is at least the required
    prominence below the LOWER peak; otherwise drop the lower peak (reason shallow_trough)."""
    pk = list(pk)
    changed = True
    while changed and len(pk) > 1:
        changed = False
        tr = _troughs_between(y, np.asarray(pk))
        for i, t in enumerate(tr):
            a, b = pk[i], pk[i + 1]
            low = a if y[a] <= y[b] else b
            if y[low] - y[t] < need_at(low) - 1e-9:
                rows.append(dict(idx=int(low), prominence=float(y[low] - y[t]), required=float(need_at(low)),
                                 width_s=np.nan, reason="shallow_trough"))
                pk.remove(low)
                changed = True
                break
    return np.asarray(pk, int)


def _cycle_mean_baseline(x, pk, n):
    """Mean of x over each peak-to-peak cycle, placed at the cycle centre, interpolated."""
    if len(pk) < 2:
        return np.full(n, np.nanmean(x))
    c = np.concatenate([[0.0], np.cumsum(x)])
    a, b = pk[:-1], pk[1:]
    means = (c[b] - c[a]) / (b - a)
    centers = (a + b) / 2.0
    return np.interp(np.arange(n), centers, means)


# ----------------------------------------------------------------------------------------
# onsets / pauses in one segment
# ----------------------------------------------------------------------------------------
def _pause_or_crossing(y, lo, hi, band, min_pause, rising):
    """Search y[lo:hi] (lo < hi) for the phase transition.

    rising=True  : trough -> next peak. Onset = first sample after the LAST at-or-below-zero
                   sample (MATLAB: find(resp > thr == 0, 1, 'last')). A pause is a run with
                   |y| <= band that contains the last at-or-below-zero sample, >= min_pause.
    rising=False : peak -> trough. Onset = first sample after the LAST above-zero sample.
    Returns (onset, pause_onset or nan) as absolute indices, or (nan, nan) if no crossing."""
    seg = y[lo:hi]
    if len(seg) < 2:
        return np.nan, np.nan
    below = seg <= 0 if rising else seg > 0
    idx = np.where(below)[0]
    if len(idx) == 0:
        return np.nan, np.nan
    z = int(idx[-1])                      # last sample still on the "before" side
    if z == len(seg) - 1:
        return np.nan, np.nan             # never crosses before the extremum
    onset = z + 1
    inband = np.abs(seg) <= band
    pause = np.nan
    if inband[z] or inband[min(z + 1, len(seg) - 1)]:
        k0 = z if inband[z] else z + 1
        s = k0
        while s > 0 and inband[s - 1]:
            s -= 1
        e = k0
        while e < len(seg) - 1 and inband[e + 1]:
            e += 1
        if (e - s + 1) >= min_pause:
            pause = lo + s
            onset = e + 1
    return lo + onset, pause


# ----------------------------------------------------------------------------------------
# main entry point
# ----------------------------------------------------------------------------------------
def detect_breaths(signal, fs, time_s=None, config: AdaptiveDetectorConfig | None = None):
    """Detect breaths. Returns a BreathDetectionResult; indices refer to ``signal``."""
    cfg = config or AdaptiveDetectorConfig()
    fs = float(fs)
    if fs < MIN_SUPPORTED_FS:
        raise ValueError(f"adaptive_prominence requires fs >= {MIN_SUPPORTED_FS} Hz (got {fs})")
    if cfg.polarity not in (1, -1):
        raise ValueError("polarity must be +1 or -1")
    x = cfg.polarity * np.asarray(signal, float).ravel()
    n = len(x)
    notes = []
    if fs < 100:
        notes.append(f"fs={fs} Hz: timing resolution {1000 / fs:.0f} ms per sample")

    valid, reason, low_amp, segments, noise_ptp, mask_notes = build_validity_mask(x, fs, cfg, time_s)
    notes += mask_notes
    # odd window so the running mean is centred (an even window, as in BreathMetrics' N=4 at
    # 400 Hz, shifts every landmark by half a sample)
    smooth_n = max(int(np.floor(fs / 1000.0 * cfg.smooth_ms)), 1)
    smooth_n += 1 - smooth_n % 2
    xs = np.full(n, np.nan)
    base = np.full(n, np.nan)
    rows_b, rows_r = [], []
    min_pause = max(int(round(cfg.min_pause_s * fs)), 2)

    for sid, (s0, s1) in enumerate(segments):
        seg = running_mean(x[s0:s1], smooth_n)
        xs[s0:s1] = seg
        # pass A: provisional cycles on a high-passed copy (drift removed) -> cycle-mean
        # baseline computed on the UNFILTERED signal
        hp = _highpass(seg, fs, cfg.scale_highpass_hz)
        pkA, _, _ = _detect_peaks(hp, fs, cfg, noise_ptp)
        bl = _cycle_mean_baseline(seg, pkA, len(seg))
        y = seg - bl
        # pass B: final peaks on the baseline-corrected unfiltered signal; threshold scale from
        # the high-passed copy so residual drift cannot inflate it
        pk, rej, lr = _detect_peaks(y, fs, cfg, noise_ptp, scale_signal=hp)
        need_at = lambda i: max(cfg.prominence_frac * lr[i], cfg.noise_mult * noise_ptp)  # noqa: E731
        pk = _validate_cycles(y, pk, need_at, rej)
        # refine the baseline with the final peaks, keep the same peak set
        if len(pk) >= 2:
            bl = _cycle_mean_baseline(seg, pk, len(seg))
            y = seg - bl
        base[s0:s1] = bl
        for r in rej:
            r.update(segment=sid, idx=r["idx"] + s0)
            rows_r.append(r)
        if len(pk) == 0:
            continue
        tr = _troughs_between(y, pk)            # tr[i] follows pk[i]
        med_ibi = int(np.median(np.diff(pk))) if len(pk) > 1 else int(0.15 * fs)
        for i, p in enumerate(pk):
            prev_t = tr[i - 1] if i > 0 else None
            next_t = tr[i] if i < len(tr) else None
            ref_amp = y[p] - (y[prev_t] if prev_t is not None else (y[next_t] if next_t is not None else 0.0))
            band = max(cfg.pause_band_frac * abs(ref_amp), cfg.pause_noise_mult * noise_ptp)
            lo = prev_t if prev_t is not None else max(0, p - med_ibi)
            inh_on, exh_pause_prev = _pause_or_crossing(y, lo, p + 1, band, min_pause, rising=True)
            if next_t is not None:
                band_e = max(cfg.pause_band_frac * abs(y[p] - y[next_t]), cfg.pause_noise_mult * noise_ptp)
                exh_on, inh_pause = _pause_or_crossing(y, p, next_t + 1, band_e, min_pause, rising=False)
            else:
                exh_on, inh_pause = np.nan, np.nan
            rows_b.append(dict(segment=sid, peak=s0 + p, trough=np.nan if next_t is None else s0 + next_t,
                               inhale_onset=np.nan if np.isnan(inh_on) else s0 + inh_on,
                               exhale_onset=np.nan if np.isnan(exh_on) else s0 + exh_on,
                               inhale_pause_onset=np.nan if np.isnan(inh_pause) else s0 + inh_pause,
                               exhale_pause_onset_before=np.nan if np.isnan(exh_pause_prev) else s0 + exh_pause_prev,
                               first_in_segment=i == 0, last_in_segment=i == len(pk) - 1,
                               peak_value=float(y[p]),
                               trough_value=np.nan if next_t is None else float(y[next_t]),
                               cycle_amplitude=float(ref_amp),   # peak minus preceding (else following) trough
                               local_range=float(lr[p]),
                               low_amplitude=bool(low_amp[s0 + p])))

    corrected = xs - base
    breaths = pd.DataFrame(rows_b)
    if len(breaths):
        breaths = _finish_breath_table(breaths, corrected, fs)
    rejected = pd.DataFrame(rows_r, columns=["idx", "prominence", "required", "width_s", "reason", "segment"])
    if time_s is not None and len(breaths):
        t = np.asarray(time_s, float)
        breaths["peak_time_s"] = t[breaths.peak.astype(int)]
        breaths["inhale_onset_time_s"] = np.where(breaths.inhale_onset.notna(),
                                                  t[breaths.inhale_onset.fillna(0).astype(int)], np.nan)
    return BreathDetectionResult(fs=fs, config=cfg, smoothed=xs, baseline=base, corrected=corrected,
                                 valid=valid, invalid_reason=reason, low_amplitude=low_amp,
                                 segments=segments, noise_ptp=noise_ptp, breaths=breaths,
                                 rejected=rejected, time_s=None if time_s is None else np.asarray(time_s),
                                 notes=notes)


def _finish_breath_table(b, y, fs):
    """Offsets, durations, pauses and volume proxies, using BreathMetrics definitions,
    never linking breaths across segments."""
    b = b.sort_values("peak").reset_index(drop=True)
    nxt_same = (b.segment.shift(-1) == b.segment)
    next_inh_on = b.inhale_onset.shift(-1).where(nxt_same)
    # exhale pause of breath i = the pause found before breath i+1's inhale (same segment)
    b["exhale_pause_onset"] = b.exhale_pause_onset_before.shift(-1).where(nxt_same)
    b = b.drop(columns="exhale_pause_onset_before")
    b["inhale_offset"] = np.where(b.inhale_pause_onset.notna(), b.inhale_pause_onset - 1, b.exhale_onset - 1)
    b["exhale_offset"] = np.where(b.exhale_pause_onset.notna(), b.exhale_pause_onset - 1, next_inh_on - 1)
    b["next_inhale_onset"] = next_inh_on
    b["complete_cycle"] = b[["inhale_onset", "exhale_onset", "inhale_offset", "exhale_offset"]].notna().all(axis=1)
    b["inhale_duration_s"] = (b.inhale_offset - b.inhale_onset) / fs
    b["exhale_duration_s"] = (b.exhale_offset - b.exhale_onset) / fs
    b["inhale_pause_duration_s"] = np.where(b.inhale_pause_onset.notna(), (b.exhale_onset - b.inhale_pause_onset) / fs, np.nan)
    b["exhale_pause_duration_s"] = np.where(b.exhale_pause_onset.notna(), (next_inh_on - b.exhale_pause_onset) / fs, np.nan)

    def vol(a, z):
        if not (np.isfinite(a) and np.isfinite(z)) or z < a:
            return np.nan
        return float(np.sum(np.abs(y[int(a):int(z)]))) / fs * 1000.0   # same scaling as BreathMetrics

    b["inhale_volume_proxy"] = [vol(a, z) for a, z in zip(b.inhale_onset, b.inhale_offset)]
    b["exhale_volume_proxy"] = [vol(a, z) for a, z in zip(b.exhale_onset, b.exhale_offset)]
    b["ibi_s"] = (next_inh_on - b.inhale_onset) / fs
    return b


# ----------------------------------------------------------------------------------------
# feature status: what has been checked, and how far it can be trusted
# ----------------------------------------------------------------------------------------
FEATURE_STATUS = {
    # breath timing: synthetic ground truth + real-data internal checks
    "n_breaths": "supported", "breathing_rate_hz": "supported", "mean_ibi_sec": "supported",
    "cv_ibi": "supported",
    # phase durations depend on the local-baseline (zero-flow) estimate
    "mean_inhale_dur_sec": "provisional", "cv_inhale_dur": "provisional",
    "mean_exhale_dur_sec": "provisional", "cv_exhale_dur": "provisional", "ie_ratio": "provisional",
    "inhale_duty_cycle": "provisional", "exhale_duty_cycle": "provisional",
    # amplitudes are uncalibrated pressure (ADC/volts after filtering), relative to local baseline
    "mean_peak_insp_flow": "provisional", "mean_peak_exp_flow": "provisional",
    "cv_peak_insp_flow": "provisional",
    # integrated pressure, NOT volume; equal-area assumption and sensor calibration unverified
    "mean_inhale_vol": "proxy_provisional", "mean_exhale_vol": "proxy_provisional",
    "mean_tidal_vol_proxy": "proxy_provisional", "cv_inhale_vol": "proxy_provisional",
    "ventilation_proxy": "proxy_provisional",
    # pauses: synthetic tests pass, but real-signal pause detection is unvalidated
    "pct_breaths_with_inhale_pause": "provisional", "mean_inhale_pause_dur_sec": "provisional",
    "cv_inhale_pause_dur": "provisional", "inhale_pause_duty_cycle": "provisional",
    "pct_breaths_with_exhale_pause": "provisional", "mean_exhale_pause_dur_sec": "provisional",
    "cv_exhale_pause_dur": "provisional", "exhale_pause_duty_cycle": "provisional",
}


def feature_status_table():
    return pd.DataFrame(sorted(FEATURE_STATUS.items()), columns=["feature", "status"])


def features_with_status(statuses=("supported",)):
    return [f for f, s in FEATURE_STATUS.items() if s in statuses]


def session_quality(result: BreathDetectionResult):
    """Whole-session summary in ABSOLUTE signal units, for comparison ACROSS sessions.

    All within-session rules (dropout, artifact, prominence) are relative to the session's
    own typical amplitude, so a recording that is broken from start to finish looks
    'normal' to them. Compare median_cycle_amplitude across sessions recorded with the same
    hardware: e.g. valence 5_13 (documented drop) is ~10-20 vs ~1500-3000 elsewhere."""
    b = result.breaths
    return dict(median_cycle_amplitude=float(np.nanmedian(b.cycle_amplitude)) if len(b) else np.nan,
                median_local_range=float(np.nanmedian(b.local_range)) if len(b) else np.nan,
                noise_ptp=result.noise_ptp, invalid_frac=float(np.mean(~result.valid)),
                low_amplitude_breath_frac=float(b.low_amplitude.mean()) if len(b) else np.nan,
                n_breaths=len(b))


def polarity_diagnostic(result: BreathDetectionResult):
    """Weak, data-driven polarity evidence. In resting rodents expiration is usually longer
    than inspiration and pauses mostly follow expiration. Returns medians; not proof."""
    b = result.breaths
    if not len(b):
        return {}
    return dict(median_positive_lobe_s=float(np.nanmedian(b.inhale_duration_s)),
                median_negative_lobe_s=float(np.nanmedian(b.exhale_duration_s)),
                frac_pause_after_positive_lobe=float(b.inhale_pause_onset.notna().mean()),
                frac_pause_after_negative_lobe=float(b.exhale_pause_onset.notna().mean()))
