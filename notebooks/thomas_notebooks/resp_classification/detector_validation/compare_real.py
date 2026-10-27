"""Legacy vs adaptive on real recordings (rank: EVALUATION minutes only = even minutes;
valence: all minutes, never used for tuning).

  python compare_real.py <out_dir> <config_json or 'default'>

Writes to <out_dir>: detection_per_recording.csv, detection_by_amplitude.csv,
acf_windows.csv, features_rank_windows.csv, features_valence_windows.csv,
feature_change_summary.csv, polarity.csv, per-recording npz (for figures/labeling).
"""
import json
import os
import sys
import traceback
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from real_common import FS, HERE, all_recordings, dev_mask, load_signal, match_idx
from adaptive_breath_detection import AdaptiveDetectorConfig, polarity_diagnostic
import resp_helper_functions as rh

TOL = int(0.025 * FS)
FEATURES = ["n_breaths", "breathing_rate_hz", "mean_ibi_sec", "cv_ibi", "mean_inhale_dur_sec", "cv_inhale_dur",
            "mean_exhale_dur_sec", "cv_exhale_dur", "ie_ratio", "mean_peak_insp_flow", "mean_peak_exp_flow",
            "cv_peak_insp_flow", "mean_inhale_vol", "mean_exhale_vol", "mean_tidal_vol_proxy", "cv_inhale_vol",
            "ventilation_proxy", "pct_breaths_with_inhale_pause", "mean_inhale_pause_dur_sec",
            "cv_inhale_pause_dur", "inhale_pause_duty_cycle", "pct_breaths_with_exhale_pause",
            "mean_exhale_pause_dur_sec", "cv_exhale_pause_dur", "exhale_pause_duty_cycle", "inhale_duty_cycle",
            "exhale_duty_cycle"]
RANK_WINDOWS = os.path.join(HERE, "..", "..", "seq-sent", "detector_audit", "rank_features_orig.csv")


def acf_rate(seg):
    seg = seg - seg.mean()
    if not np.all(np.isfinite(seg)) or seg.std() == 0:
        return np.nan
    ac = np.correlate(seg, seg, "full")[len(seg) - 1:]
    ac = ac / ac[0]
    lo, hi = int(0.06 * FS), int(1.0 * FS)
    pk, _ = find_peaks(ac[lo:hi], height=0.3)
    return FS / (pk[0] + lo) if len(pk) else np.nan


def ratio_flags(onsets, seg):
    on = np.asarray(onsets, float)
    ok = np.isfinite(on)
    on, seg = on[ok], np.asarray(seg)[ok]
    ibi = np.diff(on)
    ibi[np.diff(seg) != 0] = np.nan
    r = np.full(len(ibi), np.nan)
    for j in range(len(ibi)):
        nb = np.r_[ibi[max(0, j - 3):j], ibi[j + 1:j + 4]]
        nb = nb[np.isfinite(nb)]
        if len(nb) >= 3 and np.isfinite(ibi[j]):
            r[j] = ibi[j] / np.median(nb)
    return on[:-1], r


def nk_peaks(sig):
    try:
        import neurokit2 as nk
        _, info = nk.rsp_peaks(sig, sampling_rate=FS, method="khodadad2018")
        return np.asarray(info["RSP_Peaks"], float)
    except Exception:  # noqa: BLE001
        return None


def run(args):
    key, rec, cfg_dict, out = args
    try:
        cfg = AdaptiveDetectorConfig(**cfg_dict) if cfg_dict else AdaptiveDetectorConfig()
        sig, t = load_signal(key, rec)
        n = len(sig)
        evalmask = ~dev_mask(n) if rec["cohort"] == "rank" else np.ones(n, bool)
        bl = rh.fit_bm_session(sig, FS, detector="legacy")
        ba = rh.fit_bm_session(sig, FS, detector="adaptive_prominence", detector_config=cfg, time_s=t)
        res = ba.adaptiveResult
        valid = res.valid
        pa = ba.inhalePeaks.astype(int)
        pl = np.asarray(bl.inhalePeaks, int) if bl is not None else np.array([], int)
        nkp = nk_peaks(sig)
        # amplitude stratum at each peak: local robust range relative to session median
        lr_rel = res.breaths.local_range.to_numpy() / np.nanmedian(res.breaths.local_range)
        from adaptive_breath_detection import local_range
        lr_all = local_range(np.where(np.isfinite(res.smoothed), res.smoothed, 0), FS)
        lr_all_rel = lr_all / np.nanmedian(res.breaths.local_range)
        live_eval = evalmask & valid
        a_in, l_in = pa[live_eval[pa]], pl[live_eval[pl]] if len(pl) else pl
        a_match = match_idx(a_in, l_in, TOL)
        l_match = match_idx(l_in, a_in, TOL)
        bins = [0, 0.25, 0.5, 1.0, np.inf]
        names = ["<0.25x", "0.25-0.5x", "0.5-1x", ">1x"]
        amp_rows = []
        for lo, hi, nm in zip(bins[:-1], bins[1:], names):
            sa = (lr_all_rel[a_in] >= lo) & (lr_all_rel[a_in] < hi)
            sl = (lr_all_rel[l_in] >= lo) & (lr_all_rel[l_in] < hi) if len(l_in) else np.array([], bool)
            amp_rows.append(dict(key=key, stratum=nm, adaptive=int(sa.sum()), legacy=int(sl.sum()),
                                 adaptive_only=int((sa & ~a_match).sum()), legacy_only=int((sl & ~l_match).sum())))
        on_a, r_a = ratio_flags(ba.inhaleOnsets, ba.breathSegmentIds)
        on_l, r_l = ratio_flags(bl.inhaleOnsets, np.zeros(len(bl.inhaleOnsets))) if bl is not None else (np.array([]), np.array([]))
        ev_a = live_eval[on_a.astype(int)] & np.isfinite(r_a)
        ev_l = live_eval[on_l.astype(int)] & np.isfinite(r_l) if len(on_l) else np.array([], bool)
        det = dict(key=key, cohort=rec["cohort"], subject=rec["subject"], label=rec["label"],
                   eval_live_s=live_eval.sum() / FS, invalid_frac=float(np.mean(~valid)),
                   invalid_reasons=json.dumps(pd.Series(res.invalid_reason[~valid]).value_counts().to_dict()),
                   n_adaptive=len(a_in), n_legacy=len(l_in),
                   adaptive_only=int((~a_match).sum()), legacy_only=int((~l_match).sum()),
                   adaptive_rate_hz=len(a_in) / (live_eval.sum() / FS), legacy_rate_hz=len(l_in) / (live_eval.sum() / FS),
                   doubling_adaptive=float(np.mean((r_a[ev_a] > 1.7) & (r_a[ev_a] < 2.3))),
                   doubling_legacy=float(np.mean((r_l[ev_l] > 1.7) & (r_l[ev_l] < 2.3))) if len(ev_l) else np.nan,
                   halving_adaptive=float(np.mean(r_a[ev_a] < 0.6)),
                   halving_legacy=float(np.mean(r_l[ev_l] < 0.6)) if len(ev_l) else np.nan,
                   noise_ptp=res.noise_ptp, legacy_failed=bl is None)
        if nkp is not None:
            nk_in = nkp[live_eval[nkp.astype(int)]]
            det.update(n_neurokit=len(nk_in), adaptive_matched_by_nk=float(match_idx(a_in, nk_in, TOL).mean()),
                       legacy_matched_by_nk=float(match_idx(l_in, nk_in, TOL).mean()) if len(l_in) else np.nan,
                       nk_matched_by_adaptive=float(match_idx(nk_in, a_in, TOL).mean()),
                       nk_matched_by_legacy=float(match_idx(nk_in, l_in, TOL).mean()) if len(l_in) else np.nan)
        pol = dict(key=key, **polarity_diagnostic(res))
        # windows
        if rec["cohort"] == "rank":
            w = pd.read_csv(RANK_WINDOWS)
            w = w[w.Trial == key][["Start", "Stop"]]
            wins = list(zip(w.Start, w.Stop))
        else:
            wins = rh.sample_random_nonoverlapping_windows(time=t, signal=sig, window_dur=3.0, n_windows=100, seed=42)
        frows, arows = [], []
        for wi, (s, e) in enumerate(wins):
            row = dict(key=key, window=wi, Start=s, Stop=e, eval_window=bool(evalmask[int(s * FS):int(e * FS)].all()))
            for nm, bm in [("legacy", bl), ("adaptive", ba)]:
                f = rh.extract_bout_features(bm, s, e, min_breaths=1) if bm is not None else None
                for c in FEATURES:
                    row[f"{c}__{nm}"] = np.nan if f is None else f[c]
            row.update({k2: v for k2, v in rh.window_qc(ba, s, e).items()})
            frows.append(row)
            seg = sig[int(s * FS):int(e * FS)]
            arows.append(dict(key=key, window=wi, eval_window=row["eval_window"], acf_rate=acf_rate(seg),
                              rate_legacy=row["breathing_rate_hz__legacy"], rate_adaptive=row["breathing_rate_hz__adaptive"],
                              qc_invalid_frac=row["qc_invalid_frac"]))
        np.savez_compressed(os.path.join(out, "recordings", f"{key}.npz"), sig=sig, smoothed=res.smoothed,
                            baseline=res.baseline, valid=valid, reason=res.invalid_reason.astype(str),
                            peaks_a=pa, troughs_a=ba.exhaleTroughs, onsets_a=ba.inhaleOnsets,
                            exh_onsets_a=ba.exhaleOnsets, peaks_l=pl,
                            legacy_x=(bl.baselineCorrectedRespiration if bl is not None else np.array([])),
                            legacy_thr=(np.mean(bl.baselineCorrectedRespiration) + np.std(bl.baselineCorrectedRespiration) / 2)
                            if bl is not None else np.nan,
                            evalmask=evalmask, nk=nkp if nkp is not None else np.array([]))
        print(key, "done", flush=True)
        return dict(det=[det], amp=amp_rows, feat=frows, acf=arows, pol=[pol], err=[])
    except Exception:  # noqa: BLE001
        return dict(det=[], amp=[], feat=[], acf=[], pol=[], err=[dict(key=key, error=traceback.format_exc())])


if __name__ == "__main__":
    out = sys.argv[1]
    cfg_arg = sys.argv[2] if len(sys.argv) > 2 else "default"
    cfg_dict = None if cfg_arg == "default" else json.load(open(cfg_arg))
    os.makedirs(os.path.join(out, "recordings"), exist_ok=True)
    recs = all_recordings()
    jobs = [(k, r, cfg_dict, out) for k, r in recs.items()]
    with Pool(4) as pool:
        res = pool.map(run, jobs)
    for part in ["det", "amp", "feat", "acf", "pol", "err"]:
        df = pd.DataFrame([x for r in res for x in r[part]])
        name = dict(det="detection_per_recording", amp="detection_by_amplitude", feat="features_windows",
                    acf="acf_windows", pol="polarity", err="errors")[part]
        df.to_csv(os.path.join(out, f"{name}.csv"), index=False)
    print("done")
