"""Parameter sweep on DEVELOPMENT data only (odd minutes of the 7 rank recordings).

Selection uses detector-intrinsic criteria, never classifier performance:
  amp_recall   : real signal scaled to 25% in 20 s of every dev minute; fraction of the
                 unscaled detections still found (tol 20 ms)      -> want >= 0.97
  spike_fp     : 3-sample spikes (amplitude = local 1 s range) injected 20x per dev minute;
                 fraction that create a NEW detection within 10 ms -> want <= 0.05
  drift_stable : 0.5 Hz sinusoid (amplitude = session median 1 s range) added; fraction of
                 base detections kept AND fraction of drift-run detections that are in base
                 (min of the two)                                  -> want >= 0.97
  doubling     : fraction of IBIs 1.7-2.3x the median of the 6 neighbouring IBIs (misses)
  halving      : fraction of IBIs < 0.6x that median (splits / false peaks)
  pileup       : fraction of IBIs within 5 ms of the minimum peak distance (constraint binding)
Pick: among configs meeting the three thresholds, minimise doubling + halving.

  python sweep_dev.py <out_dir>
"""
import itertools
import os
import sys
from dataclasses import replace
from multiprocessing import Pool

import numpy as np
import pandas as pd

from real_common import FS, RANK_TRIALS, all_recordings, dev_mask, load_signal, match_idx
from adaptive_breath_detection import AdaptiveDetectorConfig, detect_breaths, local_range

TOL = int(0.02 * FS)


def configs():
    base = AdaptiveDetectorConfig()
    out = []
    for frac, dist in itertools.product([0.10, 0.15, 0.20, 0.25, 0.30], [0.04, 0.05, 0.06, 0.08]):
        out.append(replace(base, prominence_frac=frac, min_peak_distance_s=dist))
    for nm in [2.0, 5.0]:
        out.append(replace(base, noise_mult=nm))
    for wl in [0.5, 2.0, None]:
        out.append(replace(base, prominence_wlen_s=wl))
    for hp in [0.5, None]:
        out.append(replace(base, scale_highpass_hz=hp))
    for lw in [0.5, 2.0]:
        out.append(replace(base, local_range_win_s=lw))
    return out


def ibi_flags(peaks, seg_ids, mask):
    peaks = np.asarray(peaks)
    ibi = np.diff(peaks).astype(float)
    same = np.diff(seg_ids) == 0
    ibi[~same] = np.nan
    ratio = np.full(len(ibi), np.nan)
    for j in range(len(ibi)):
        nb = np.r_[ibi[max(0, j - 3):j], ibi[j + 1:j + 4]]
        nb = nb[np.isfinite(nb)]
        if len(nb) >= 3 and np.isfinite(ibi[j]):
            ratio[j] = ibi[j] / np.median(nb)
    inside = mask[peaks[:-1]] & np.isfinite(ratio)
    r = ratio[inside]
    return ibi[inside], r


def run(args):
    key, cfgs = args
    sig, t = load_signal(key, all_recordings()[key])
    n = len(sig)
    dev = dev_mask(n)
    rng = np.random.default_rng(abs(hash(key)) % 2**32)
    lr = local_range(sig, FS)
    w = FS
    med_range = np.median([np.ptp(sig[i:i + w]) for i in range(0, n - w, w)])
    # perturbations (built once, same for every config)
    scaled = sig.copy()
    zones = np.zeros(n, bool)
    for m0 in range(int(60 * FS), n, int(120 * FS)):
        a, b = m0 + int(20 * FS), min(m0 + int(40 * FS), n)
        scaled[a:b] *= 0.25
        zones[a:b] = True
    spiked = sig.copy()
    dev_idx = np.where(dev)[0]
    spikes = rng.choice(dev_idx[(dev_idx > 2) & (dev_idx < n - 3)], int(20 * dev.sum() / (60 * FS)), replace=False)
    for s in spikes:
        spiked[s - 1:s + 2] += lr[s]
    drifted = sig + med_range * np.sin(2 * np.pi * 0.5 * np.arange(n) / FS)
    rows = []
    for ci, cfg in enumerate(cfgs):
        r0 = detect_breaths(sig, FS, time_s=t, config=cfg)
        p0 = r0.breaths.peak.to_numpy().astype(int)
        p0_dev = p0[dev[p0]]
        # amplitude scaling: base detections inside scaled zones
        ps = detect_breaths(scaled, FS, time_s=t, config=cfg).breaths.peak.to_numpy()
        base_z = p0[zones[p0]]
        amp_recall = match_idx(base_z, ps, TOL).mean() if len(base_z) else np.nan
        # spikes: new detections near spikes
        pk = detect_breaths(spiked, FS, time_s=t, config=cfg).breaths.peak.to_numpy()
        new = pk[~match_idx(pk, p0, TOL)]
        spike_fp = match_idx(spikes, new, int(0.01 * FS)).mean()
        # drift
        pdr = detect_breaths(drifted, FS, time_s=t, config=cfg).breaths.peak.to_numpy()
        pdr_dev = pdr[dev[pdr.astype(int)]]
        drift_stable = min(match_idx(p0_dev, pdr_dev, TOL).mean(), match_idx(pdr_dev, p0_dev, TOL).mean())
        ibi, ratio = ibi_flags(p0, r0.breaths.segment.to_numpy(), dev)
        rows.append(dict(trial=key, config=ci, n_dev_breaths=len(p0_dev),
                         dev_rate_hz=len(p0_dev) / (dev & r0.valid).sum() * FS,
                         amp_recall=amp_recall, spike_fp=spike_fp, drift_stable=drift_stable,
                         doubling=np.mean((ratio > 1.7) & (ratio < 2.3)), halving=np.mean(ratio < 0.6),
                         pileup=np.mean(np.abs(ibi / FS - cfg.min_peak_distance_s) < 0.005),
                         invalid_frac_dev=np.mean(~r0.valid[dev])))
    print(key, "done", flush=True)
    return rows


if __name__ == "__main__":
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    cfgs = configs()
    pd.DataFrame([dict(config=i, **c.to_dict()) for i, c in enumerate(cfgs)]).to_csv(os.path.join(out, "sweep_configs.csv"), index=False)
    with Pool(4) as pool:
        res = pool.map(run, [(k, cfgs) for k in RANK_TRIALS])
    df = pd.DataFrame([r for rr in res for r in rr])
    df.to_csv(os.path.join(out, "sweep_dev_per_recording.csv"), index=False)
    agg = df.groupby("config").agg(amp_recall=("amp_recall", "mean"), spike_fp=("spike_fp", "mean"),
                                   drift_stable=("drift_stable", "mean"), doubling=("doubling", "mean"),
                                   halving=("halving", "mean"), pileup=("pileup", "mean"),
                                   dev_rate_hz=("dev_rate_hz", "mean"), worst_amp_recall=("amp_recall", "min"),
                                   worst_drift=("drift_stable", "min"))
    agg = pd.read_csv(os.path.join(out, "sweep_configs.csv")).set_index("config")[
        ["prominence_frac", "min_peak_distance_s", "noise_mult", "prominence_wlen_s", "scale_highpass_hz",
         "local_range_win_s"]].join(agg)
    ok = (agg.amp_recall >= 0.97) & (agg.spike_fp <= 0.05) & (agg.drift_stable >= 0.97)
    agg["eligible"] = ok
    agg["score"] = agg.doubling + agg.halving
    agg.to_csv(os.path.join(out, "sweep_dev_summary.csv"))
    pd.set_option("display.width", 250)
    print(agg.round(4).to_string())
    best = agg[ok].score.idxmin() if ok.any() else None
    print("\nselected config:", best)
    if best is not None:
        print(agg.loc[best])
