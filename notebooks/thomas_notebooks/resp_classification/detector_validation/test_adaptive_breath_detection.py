"""Tests for adaptive_breath_detection.py, the detector switch, and the legacy fixes.

Runs under pytest, or directly:  python test_adaptive_breath_detection.py
(biopipeline-env has no pytest; the __main__ block is a minimal runner.)
"""
import os
import sys
import traceback
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
warnings.filterwarnings("ignore")

from adaptive_breath_detection import (  # noqa: E402
    AdaptiveDetectorConfig, detect_breaths, running_mean)
from breathmetrics import bmObject  # noqa: E402
from synthetic import make_signal, match, regular  # noqa: E402

FS = 400
TOL = int(0.02 * FS)  # 20 ms matching tolerance


def _recall_precision(truth, det, tol=TOL):
    tp, fp, fn, err, _ = match(truth, det, tol)
    return tp / max(tp + fn, 1), tp / max(tp + fp, 1), err


def _legacy(x, fs=FS):
    bm = bmObject(x, fs, "rodentAirflow")
    bm.estimateAllFeatures()
    return bm


def _adaptive(x, fs=FS, **kw):
    bm = bmObject(x, fs, "rodentAirflow")
    bm.estimateAllFeatures(detector="adaptive_prominence", **kw)
    return bm


# ------------------------------------------------------------------ detection basics
def test_clean_known_peaks_and_offbyone():
    x, tr, _ = make_signal(FS, regular(200, 6, A=100))
    b = detect_breaths(x, FS).breaths
    rec, prec, err = _recall_precision(tr["peak"], b.peak)
    assert rec == 1.0 and prec == 1.0
    assert np.all(np.abs(err) <= 1), "peaks must be within one sample of the true maximum"
    _, _, err_on = _recall_precision(tr["inhale_onset"], b.inhale_onset)
    assert np.all(np.abs(err_on) <= 1)


def test_amplitude_change_reproduces_legacy_failure_and_adaptive_fix():
    cyc = regular(240, 6, A=100)
    for i in range(80, 160):
        cyc[i] = dict(cyc[i], A=15)            # 15% amplitude stretch
    x, tr, _ = make_signal(FS, cyc, noise_sd=1.0)
    quiet = tr["peak"][80:160]
    lo, hi = quiet[0] - 30, quiet[-1] + 30

    def in_quiet(p):
        p = np.asarray(p, float)
        return p[(p > lo) & (p < hi)]

    r_leg, _, _ = _recall_precision(quiet, in_quiet(_legacy(x).inhalePeaks))
    r_ada, p_ada, _ = _recall_precision(quiet, in_quiet(detect_breaths(x, FS).breaths.peak))
    assert r_leg < 0.2, f"legacy should miss quiet breaths (bug reproduction); recall={r_leg}"
    assert r_ada >= 0.98 and p_ada >= 0.98


def test_drift_and_dc_offset():
    x, tr, _ = make_signal(FS, regular(240, 6, A=100), noise_sd=1.0, drift=[(150, 0.3), (60, 1.1)], dc=5000)
    b = detect_breaths(x, FS).breaths
    rec, prec, _ = _recall_precision(tr["peak"], b.peak)
    assert rec >= 0.98 and prec >= 0.98
    _, _, e = _recall_precision(tr["inhale_onset"], b.inhale_onset, tol=int(0.03 * FS))
    assert np.median(np.abs(e)) <= 2, "median onset error <= 5 ms under drift"


def test_frequency_change_and_fast_sniffing():
    cyc = regular(120, 3, A=100) + regular(240, 12, A=60) + regular(120, 4, A=100)
    x, tr, _ = make_signal(FS, cyc, noise_sd=1.0)
    rec, prec, _ = _recall_precision(tr["peak"], detect_breaths(x, FS).breaths.peak)
    assert rec >= 0.98 and prec >= 0.98


def test_irregular_cycles():
    rng = np.random.default_rng(1)
    cyc = [dict(Ti=t * 0.4, Te=t * 0.6, A=a) for t, a in zip(rng.uniform(0.08, 0.5, 300), rng.uniform(30, 150, 300))]
    x, tr, _ = make_signal(FS, cyc, noise_sd=1.0)
    rec, prec, _ = _recall_precision(tr["peak"], detect_breaths(x, FS).breaths.peak)
    assert rec >= 0.97 and prec >= 0.97


# ------------------------------------------------------------------ pauses
def test_genuine_pauses_detected():
    cyc = [dict(Ti=.06, Te=.1, A=100, Tpe=.08 if i % 2 else 0, Tpi=.04 if i % 3 == 0 else 0) for i in range(150)]
    x, tr, _ = make_signal(FS, cyc, noise_sd=1.0)
    b = detect_breaths(x, FS).breaths
    for k in ["exhale_pause_onset", "inhale_pause_onset"]:
        truth = tr[k][tr[k] >= 0]
        rec, prec, _ = _recall_precision(truth, b[k])
        assert rec >= 0.9 and prec >= 0.9, (k, rec, prec)


def test_amplitude_change_is_not_a_pause():
    cyc = regular(240, 6, A=100)
    for i in range(80, 160):
        cyc[i] = dict(cyc[i], A=10 + (i % 7) * 5)   # varying, small amplitudes, no pauses
    x, _, _ = make_signal(FS, cyc, noise_sd=0.5)
    b = detect_breaths(x, FS).breaths
    frac = (b.inhale_pause_onset.notna() | b.exhale_pause_onset.notna()).mean()
    assert frac <= 0.05, f"false pause rate {frac:.2f}"


# ------------------------------------------------------------------ artifacts / invalid data
def test_isolated_spikes_rejected():
    x, tr, _ = make_signal(FS, regular(200, 6, A=100), noise_sd=1.0)
    rng = np.random.default_rng(3)
    spikes = rng.choice(np.arange(200, len(x) - 200), 15, replace=False)
    for s in spikes:
        x[s - 1:s + 2] += 400                     # 7.5 ms wide spike
    b = detect_breaths(x, FS)
    # spikes must not create extra breaths; a spike riding on a breath crest may move that
    # breath's peak (known limitation), so allow at most one timing-induced mismatch
    tp, fp, fn, _, _ = match(tr["peak"], b.breaths.peak, TOL)
    assert fp <= 1 and fn <= 1, (tp, fp, fn)
    assert (b.rejected.reason == "spike_width").sum() >= 10


def test_movement_artifact_masked():
    x, tr, _ = make_signal(FS, regular(300, 6, A=100), noise_sd=1.0)
    a0 = int(20 * FS)
    x[a0:a0 + int(1.5 * FS)] += 5000 * np.sin(np.linspace(0, 3 * np.pi, int(1.5 * FS)))
    r = detect_breaths(x, FS)
    assert np.all(~r.valid[a0:a0 + int(1.5 * FS)])
    assert not np.any((r.breaths.peak >= a0) & (r.breaths.peak < a0 + int(1.5 * FS)))
    rec, _, _ = _recall_precision(tr["peak"][(tr["peak"] < a0 - FS) | (tr["peak"] > a0 + 3 * FS)], r.breaths.peak)
    assert rec >= 0.98


def test_dropout_masked_but_low_amplitude_breathing_kept():
    cyc = regular(360, 6, A=100)
    for i in range(240, 300):
        cyc[i] = dict(cyc[i], A=8)              # 8% amplitude, genuine breathing
    x, tr, _ = make_signal(FS, cyc, noise_sd=0.5, seed=4)
    d0, d1 = int(15 * FS), int(20 * FS)        # sensor dropout: noise only
    x[d0:d1] = np.random.default_rng(5).normal(0, 0.3, d1 - d0)
    r = detect_breaths(x, FS)
    assert np.mean(~r.valid[d0 + FS // 2:d1 - FS // 2]) > 0.9, "dropout should be invalid"
    assert not np.any((r.breaths.peak > d0 + FS // 2) & (r.breaths.peak < d1 - FS // 2))
    lowamp = tr["peak"][240:300]
    assert np.mean(r.valid[lowamp]) == 1.0, "low-amplitude breathing must stay valid"
    rec, _, _ = _recall_precision(lowamp, r.breaths.peak)
    assert rec >= 0.95
    assert r.low_amplitude[lowamp].mean() > 0.9, "and be flagged low_amplitude"


def test_nan_gap_never_bridged_and_time_preserved():
    x, tr, t = make_signal(FS, regular(240, 6, A=100), noise_sd=1.0)
    t = t + 1234.5                               # recording clock does not start at 0
    g0, g1 = int(18 * FS), int(19 * FS)
    x[g0:g1] = np.nan
    bm = _adaptive(x, time_s=t)
    b = bm.breathTable
    assert b.segment.nunique() == 2
    # no interval spans the gap
    finite = b.ibi_s.notna()
    spans = (b.peak[finite] < g0) & ((b.inhale_onset[finite] + b.ibi_s[finite] * FS) >= g0)
    assert not spans.any()
    # times are looked up, never shifted
    assert np.allclose(b.peak_time_s, t[b.peak.astype(int)])


def test_time_gap_splits_segments():
    x, tr, t = make_signal(FS, regular(240, 6, A=100), noise_sd=1.0)
    t = t.copy()
    t[int(20 * FS):] += 5.0                       # 5 s hole in the recording clock
    r = detect_breaths(x, FS, time_s=t)
    assert len(r.segments) == 2 and r.segments[0][1] == int(20 * FS)
    assert any("gap" in n for n in r.notes)


def test_partial_breaths_at_boundaries():
    x, tr, _ = make_signal(FS, regular(100, 5, A=100), noise_sd=1.0, lead_s=0.0)
    x = x[int(0.05 * FS):]                         # start mid-inhale
    b = detect_breaths(x, FS).breaths
    assert b.first_in_segment.iloc[0] and b.last_in_segment.iloc[-1]
    assert np.isnan(b.exhale_offset.iloc[-1]), "last breath has no following inhale"
    assert not b.complete_cycle.iloc[-1]


# ------------------------------------------------------------------ sampling rates
def test_sampling_rates():
    for fs in [50, 100, 200, 400, 1000]:
        x, tr, _ = make_signal(fs, regular(150, 6, A=100), noise_sd=0.5)
        b = detect_breaths(x, fs).breaths
        rec, prec, err = _recall_precision(tr["peak"], b.peak, tol=max(int(0.02 * fs), 1))
        assert rec >= 0.98 and prec >= 0.98, fs
        assert np.median(np.abs(err)) <= 1, fs              # within one sample at every rate


def test_unsupported_rate_raises():
    for bad in [20, 30]:
        try:
            detect_breaths(np.zeros(1000), bad)
        except ValueError:
            continue
        raise AssertionError(f"fs={bad} should raise")


def test_legacy_low_rate_clear_error():
    x, _, _ = make_signal(100, regular(100, 5), noise_sd=0.5)
    try:
        _legacy(x, 100)
    except ValueError as e:
        assert "zero-length window" in str(e)
        return
    raise AssertionError("legacy at 100 Hz should raise a clear ValueError")


# ------------------------------------------------------------------ legacy fixes (regressions)
def test_legacy_fix1_peaks_on_true_maximum():
    x, tr, _ = make_signal(FS, regular(200, 5, A=100))
    bm = _legacy(x)
    xb = bm.baselineCorrectedRespiration
    off = [p - 5 + np.argmax(xb[p - 5:p + 6]) - p for p in bm.inhalePeaks if 5 < p < len(x) - 6]
    assert set(off) == {0}


def test_legacy_fix2_first_peak_kept():
    t = np.arange(FS * 60) / FS
    bm = _legacy(100 * np.sin(2 * np.pi * 5 * t))
    assert bm.inhalePeaks[0] / FS < 0.06


def test_legacy_fix3_keeps_larger_of_two_peaks():
    t = np.arange(FS * 120) / FS
    y = 100 * np.sin(2 * np.pi * 5 * t)
    k = int(60 * FS)
    seg = np.arange(int(0.2 * FS)) / FS
    g = lambda c, a, w: a * np.exp(-((seg - c) / w) ** 2)  # noqa: E731
    y[k:k + len(seg)] = 40 + g(0.06, 60, 0.012) + g(0.13, 160, 0.012) - g(0.0, 140, 0.02) - g(0.2, 140, 0.02)
    p = np.asarray(_legacy(y).inhalePeaks)
    kept = (p[(p > k) & (p < k + len(seg))] - k) / FS
    assert len(kept) == 1 and abs(kept[0] - 0.13) < 0.005


def test_legacy_fix5_running_mean_edges():
    bm = bmObject(np.zeros(10), FS, "rodentAirflow")
    r = np.random.default_rng(0).normal(size=3000)
    out = bm.running_mean(r + 1000, 400)
    assert np.all(np.abs(out[:5] - 1000) < 1), "edges must not be zero-filled"
    assert np.allclose(running_mean(r, 400)[200:-200], bm.running_mean(r, 400)[200:-200])


# ------------------------------------------------------------------ pipeline integration
def test_fit_bm_session_legacy_identical_to_direct_call():
    import resp_helper_functions as rh
    x, _, _ = make_signal(FS, regular(240, 6, A=100), noise_sd=1.0)
    a = rh.fit_bm_session(x, FS)
    b = _legacy(x)
    assert np.array_equal(a.inhaleOnsets, b.inhaleOnsets)
    assert np.allclose(a.inhaleVolumes, b.inhaleVolumes, equal_nan=True)
    assert getattr(a, "detector", "legacy") == "legacy" and not hasattr(a, "breathSegmentIds")


def test_window_features_consistent_with_session_table():
    import resp_helper_functions as rh
    x, _, _ = make_signal(FS, regular(400, 6, A=100), noise_sd=1.0)
    bm = rh.fit_bm_session(x, FS, detector="adaptive_prominence")
    b = bm.breathTable
    for s in [5.0, 20.0, 41.0]:
        f = rh.extract_bout_features(bm, s, s + 3.0, min_breaths=1)
        inw = b[(b.inhale_onset >= s * FS) & (b.inhale_onset <= (s + 3) * FS)]
        on = inw.inhale_onset.to_numpy()
        assert f["n_breaths"] == len(inw)
        assert np.isclose(f["breathing_rate_hz"], 1 / np.mean(np.diff(on) / FS))
        assert np.isclose(f["mean_inhale_dur_sec"], np.nanmean(inw.inhale_duration_s))


def test_window_ibi_never_spans_gap():
    import resp_helper_functions as rh
    x, _, _ = make_signal(FS, regular(240, 6, A=100), noise_sd=1.0)
    x[int(20 * FS):int(21 * FS)] = np.nan
    bm = rh.fit_bm_session(x, FS, detector="adaptive_prominence")
    f = rh.extract_bout_features(bm, 18.5, 22.5, min_breaths=1)
    assert f["mean_ibi_sec"] < 0.2, "a 1 s gap must not enter the mean IBI (true IBI 0.167 s)"
    q = rh.window_qc(bm, 18.5, 22.5)
    assert q["qc_n_segments"] == 2 and q["qc_invalid_frac"] > 0.2


def test_default_detector_switch():
    import resp_helper_functions as rh
    x, _, _ = make_signal(FS, regular(150, 6, A=100), noise_sd=1.0)
    old = rh.DEFAULT_DETECTOR
    try:
        rh.DEFAULT_DETECTOR = "adaptive_prominence"
        assert hasattr(rh.fit_bm_session(x, FS), "breathSegmentIds")
    finally:
        rh.DEFAULT_DETECTOR = old
    assert not hasattr(rh.fit_bm_session(x, FS), "breathSegmentIds")


def test_polarity_flag():
    x, tr, _ = make_signal(FS, regular(150, 6, A=100), noise_sd=1.0)
    b = detect_breaths(-x, FS, config=AdaptiveDetectorConfig(polarity=-1)).breaths
    rec, prec, _ = _recall_precision(tr["peak"], b.peak)
    assert rec >= 0.98 and prec >= 0.98


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {name}\n{traceback.format_exc()}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
