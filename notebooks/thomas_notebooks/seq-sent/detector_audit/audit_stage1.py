"""Stage 1 of the breath-detector audit (rank cohort, 7 recordings).

Per recording:
  1. Reproduce the exact rank pipeline (load_rank_resp_signal_interactions @400 Hz ->
     fit_bm_session) and check window features against the saved pickle.
  2. Run a reference local-prominence peak detector on the SAME BreathMetrics input
     signal (bm.baselineCorrectedRespiration), at several prominence settings.
  3. Validation tests on peaks the reference finds but BreathMetrics does not:
     gap test, SNR, width, autocorrelation rate, semi-synthetic amplitude scaling.
  4. Per-window refit: BreathMetrics fit on each 3 s window alone.
  5. Corrected features: same BreathMetrics pipeline, only findExtrema swapped for the
     prominence detector, summarized over the same windows.

Run with the biopipeline-env interpreter:
  python audit_stage1.py <rank_features_orig.csv> <out_dir>
"""
import os
import sys
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

warnings.filterwarnings("ignore")
HELPERS = r"C:/Users/thoma/Code/ResearchCode/respiratory_pilot/notebooks/thomas_notebooks/resp_classification"
sys.path.insert(0, HELPERS)
from breathmetrics import bmObject  # noqa: E402
from resp_helper_functions import (  # noqa: E402
    extract_bout_features,
    fit_bm_session,
    load_rank_resp_signal_interactions,
)

DATA_DIR = (r"C:/Users/thoma/UF Dropbox/Thomas Heeps/Padilla-Coreano Lab/2025/ECG_cohort1/Aim1/"
            r"AIM1/Day1_new/CM interactions/cm_h5/baseline_cagemate_interactions_h5_outputs")
TRIAL_FILES = {  # copied from rank-featureextraction-3sec.ipynb cell 12
    "CM_1_1_d1_2": "CM_s1_1_d1_2_20250623_111352_merged.h5",
    "CM_1_2_sub1_1": "CM_s1_2_sub1_1_20250623_133932_merged.h5",
    "CM_2_3_d2_4": "CM_s2_3_d2_4_20250623_151153_merged.h5",
    "CM_2_4_sub2_3": "CM_s2_4_sub2_3_20250623_143348_merged.h5",
    "CM_3_5_d3_6": "CM_s3_5_d3_6_20250623_160001_merged.h5",
    "CM_4_7_d4_8": "CM_s4_7_d4_8_20250623_193718_merged.h5",
    "CM_4_8_sub4_7": "CM_s4_8_sub4_7_20250623_182649_merged.h5",
}
FS = 400
MATCH_TOL = int(0.025 * FS)          # a reference peak is "found" if a BM peak is within 25 ms
MIN_DIST = int(0.06 * FS)            # 60 ms minimum peak spacing (~16 Hz ceiling)
FRACS = [0.10, 0.20, 0.30, 0.40]     # reference prominence settings (fraction of local span)
MAIN_FRAC = 0.20
SEMI_SCALES = [1.0, 0.5, 0.35, 0.25, 0.15]
SEMI_TRIALS = {"CM_4_8_sub4_7": (200.0, 260.0), "CM_1_1_d1_2": (380.0, 440.0)}
FEATURE_COLS = [
    "n_breaths", "breathing_rate_hz", "mean_ibi_sec", "cv_ibi", "mean_inhale_dur_sec",
    "cv_inhale_dur", "mean_exhale_dur_sec", "cv_exhale_dur", "ie_ratio", "mean_peak_insp_flow",
    "mean_peak_exp_flow", "cv_peak_insp_flow", "mean_inhale_vol", "mean_exhale_vol",
    "mean_tidal_vol_proxy", "cv_inhale_vol", "ventilation_proxy", "pct_breaths_with_inhale_pause",
    "mean_inhale_pause_dur_sec", "cv_inhale_pause_dur", "inhale_pause_duty_cycle",
    "pct_breaths_with_exhale_pause", "mean_exhale_pause_dur_sec", "cv_exhale_pause_dur",
    "exhale_pause_duty_cycle", "inhale_duty_cycle", "exhale_duty_cycle",
]


# ----------------------------------------------------------------------------- reference detector
def noise_floor(x, fs=FS):
    """Peak-to-peak of the quietest 1% of 250 ms segments (flat stretches / sensor floor),
    capped at 1% of the median 1 s peak-to-peak: a session without dropouts has no flat
    stretches, and its quietest segments are shallow breathing, not noise."""
    w = int(0.25 * fs)
    n = len(x) // w
    q1 = np.percentile(np.ptp(x[: n * w].reshape(n, w), axis=1), 1)
    w1 = int(fs)
    n1 = len(x) // w1
    typical = np.median(np.ptp(x[: n1 * w1].reshape(n1, w1), axis=1))
    return float(min(q1, 0.01 * typical))


def local_span(x, fs=FS):
    """90th-10th percentile range in each 1 s block, broadcast to samples."""
    w = int(fs)
    n = int(np.ceil(len(x) / w))
    span = np.array([np.subtract(*np.percentile(x[i * w:(i + 1) * w], [90, 10])) for i in range(n)])
    return np.repeat(span, w)[: len(x)]


def ref_peaks(x, frac, noise, fs=FS):
    pk, pr = find_peaks(x, distance=MIN_DIST, prominence=0, width=0, rel_height=0.5)
    need = np.maximum(frac * local_span(x, fs)[pk], 3 * noise)
    keep = pr["prominences"] >= need
    return pk[keep], pr["prominences"][keep], pr["widths"][keep] / fs


def live_mask_for(x, fs=FS):
    """Seconds with real signal: 1 s peak-to-peak >= 10% of the session median (drops sensor dropouts)."""
    w = int(fs)
    n = int(np.ceil(len(x) / w))
    ptp = np.array([np.ptp(x[i * w:(i + 1) * w]) for i in range(n)])
    return np.repeat(ptp >= 0.1 * np.median(ptp), w)[: len(x)]


def matched(ref, bm_pk, tol=MATCH_TOL):
    if len(bm_pk) == 0:
        return np.zeros(len(ref), bool)
    bm_pk = np.sort(bm_pk)
    i = np.clip(np.searchsorted(bm_pk, ref), 1, len(bm_pk) - 1)
    d = np.minimum(np.abs(bm_pk[i] - ref), np.abs(bm_pk[i - 1] - ref))
    return d <= tol


# ----------------------------------------------------------------------------- corrected BM
class ProminenceBM(bmObject):
    """BreathMetrics with two steps replaced; onsets, durations, volumes and pauses are unchanged BM code.
    1. Baseline: after BM's own 60 s sliding baseline, subtract a cycle midline (interpolated through
       the midpoints between consecutive peaks and troughs). BM defines onsets as crossings of the
       trace mean, so a breath riding on sub-second drift has no crossing without this.
    2. findExtrema: prominence peaks; trough = minimum between consecutive peaks (always alternating)."""

    frac = MAIN_FRAC

    def correctRespirationToBaseline(self, method="simple", zScore=0, verbose=0):
        super().correctRespirationToBaseline(method=method, zScore=zScore, verbose=verbose)
        x = self.baselineCorrectedRespiration
        pk, _, _ = ref_peaks(x, self.frac, noise_floor(x, self.srate), self.srate)
        tr = np.array([a + np.argmin(x[a:b]) for a, b in zip(pk[:-1], pk[1:])], int)
        ext = np.sort(np.r_[pk, tr])
        knots_t = (ext[:-1] + ext[1:]) / 2
        knots_v = (x[ext[:-1]] + x[ext[1:]]) / 2
        base = np.interp(np.arange(len(x)), knots_t, knots_v)
        x = x - base
        self.baselineCorrectedRespiration = x - np.mean(x)
        # reuse these exact extrema so every cycle crosses the midline
        self._pk, self._tr = [int(p) for p in pk[:-1]], [int(v) for v in tr]

    def findExtrema(self, customDecisionThreshold=0, swSizes=[]):
        return self._pk, self._tr


def fit(cls, signal):
    bm = cls(signal, FS, "rodentAirflow")
    bm.estimateAllFeatures()
    return bm


def acf_rate(seg, fs=FS):
    """Dominant breathing rate from the autocorrelation: first ACF peak (lag 60 ms-1 s) above 0.3."""
    seg = seg - seg.mean()
    if np.std(seg) == 0:
        return np.nan
    ac = np.correlate(seg, seg, "full")[len(seg) - 1:]
    ac = ac / ac[0]
    lo, hi = int(0.06 * fs), int(1.0 * fs)
    pk, _ = find_peaks(ac[lo:hi], height=0.3)
    return fs / (pk[0] + lo) if len(pk) else np.nan


# ----------------------------------------------------------------------------- per recording
def run_trial(args):
    trial, windows, out_dir = args
    path = os.path.join(DATA_DIR, TRIAL_FILES[trial])
    signal, _, fs, _ = load_rank_resp_signal_interactions(path, target_rate=FS, prefer_raw=True)
    assert fs == FS
    bm = fit_bm_session(signal, fs, data_type="rodentAirflow")
    x = bm.baselineCorrectedRespiration
    bm_pk = np.asarray(bm.inhalePeaks, int)
    thr = float(np.mean(x) + np.std(x) / 2)
    noise = noise_floor(x)
    live = live_mask_for(x)

    # 1. reproduction check against the saved pickle
    repro = []
    for _, w in windows.iterrows():
        f = extract_bout_features(bm, w.Start, w.Stop)
        repro.append(dict(WindowID=w.WindowID, n_breaths_repro=np.nan if f is None else f["n_breaths"],
                          rate_repro=np.nan if f is None else f["breathing_rate_hz"]))

    # 2. reference detector at several settings
    sens = []
    for frac in FRACS:
        ref, prom, wid = ref_peaks(x, frac, noise)
        lv = live[ref]
        found = matched(ref, bm_pk)
        # also: how many BM peaks have no reference peak (BM-only)
        bm_only = ~matched(bm_pk, ref)
        sens.append(dict(trial=trial, frac=frac, n_ref_live=int(lv.sum()), n_missed_live=int((~found & lv).sum()),
                         n_bm_live=int(live[bm_pk].sum()), n_bm_only_live=int((bm_only & live[bm_pk]).sum())))
    ref, prom, wid = ref_peaks(x, MAIN_FRAC, noise)
    found = matched(ref, bm_pk)
    lv = live[ref]

    # 3a. gap test: for each BM interval, how many reference-only peaks fall inside, and how long is it
    #     compared with neighbouring BM intervals?
    bm_s = np.sort(bm_pk)
    ibis = np.diff(bm_s)
    missed = ref[~found & lv]
    k = np.bincount(np.clip(np.searchsorted(bm_s, missed) - 1, 0, len(ibis) - 1), minlength=len(ibis))
    gap_rows = []
    for j in range(3, len(ibis) - 3):
        if not live[bm_s[j]] or not live[bm_s[j + 1]]:
            continue
        neigh = np.r_[ibis[j - 3:j], ibis[j + 1:j + 4]]
        # neighbours should themselves be "clean" (no missed peaks inside) for a fair local reference
        clean = np.r_[k[j - 3:j], k[j + 1:j + 4]] == 0
        if clean.sum() < 3:
            continue
        R = ibis[j] / np.median(neigh[clean])
        inside = missed[(missed > bm_s[j]) & (missed < bm_s[j + 1])]
        f = (inside[0] - bm_s[j]) / ibis[j] if k[j] == 1 else np.nan
        gap_rows.append(dict(trial=trial, k=int(k[j]), R=R, f=f, a=int(bm_s[j]), b=int(bm_s[j + 1])))

    # 3b. SNR / width of missed vs found peaks
    peak_rows = pd.DataFrame(dict(trial=trial, idx=ref, prom=prom, width_s=wid, found=found, live=lv,
                                  above_thr=x[ref] > thr, snr=prom / noise))

    # 3c. independent rate check per saved window; 4. per-window refit
    win_rows = []
    for _, w in windows.iterrows():
        s, e = int(round(w.Start * FS)), int(round(w.Stop * FS))
        in_ref = ref[(ref >= s) & (ref < e)]
        rr = dict(trial=trial, WindowID=w.WindowID, acf_rate=acf_rate(x[s:e]),
                  n_ref=len(in_ref), n_bm_session=int(((bm_pk >= s) & (bm_pk < e)).sum()),
                  live_frac=float(live[s:e].mean()))
        rr["ref_recall_session"] = matched(in_ref, bm_pk).mean() if len(in_ref) else np.nan
        try:
            bw = fit(bmObject, signal[s:e])
            wp = np.asarray(bw.inhalePeaks, int) + s
            rr["n_bm_window_refit"] = len(wp)
            rr["ref_recall_window_refit"] = matched(in_ref, wp).mean() if len(in_ref) else np.nan
            rr["window_refit_extra"] = int((~matched(wp, in_ref)).sum()) if len(in_ref) else np.nan
        except Exception as ex:  # noqa: BLE001
            rr["n_bm_window_refit"] = np.nan
            rr["ref_recall_window_refit"] = 0.0
            rr["window_refit_error"] = type(ex).__name__
        win_rows.append(rr)

    # 5. corrected features on the same windows
    bmc = fit(ProminenceBM, signal)
    corr = []
    for _, w in windows.iterrows():
        f = extract_bout_features(bmc, w.Start, w.Stop, min_breaths=1)
        row = dict(trial=trial, WindowID=w.WindowID)
        row.update({c: (np.nan if f is None else f[c]) for c in FEATURE_COLS})
        corr.append(row)

    # 3d. semi-synthetic amplitude scaling on real signal
    semi = []
    if trial in SEMI_TRIALS:
        a, b = (int(t * FS) for t in SEMI_TRIALS[trial])
        truth_bm = bm_pk[(bm_pk >= a) & (bm_pk < b)]
        truth_ref = ref[(ref >= a) & (ref < b)]
        for sc in SEMI_SCALES:
            sig2 = signal.copy()
            sig2[a:b] *= sc
            try:
                b2 = fit_bm_session(sig2, FS)
                p2 = np.asarray(b2.inhalePeaks, int)
                x2 = b2.baselineCorrectedRespiration
                r2, _, _ = ref_peaks(x2, MAIN_FRAC, noise_floor(x2))
                semi.append(dict(trial=trial, scale=sc,
                                 bm_recall_of_bm_truth=matched(truth_bm, p2).mean(),
                                 bm_recall_of_ref_truth=matched(truth_ref, p2).mean(),
                                 ref_recall_of_ref_truth=matched(truth_ref, r2).mean(),
                                 n_truth_bm=len(truth_bm), n_truth_ref=len(truth_ref)))
            except Exception as ex:  # noqa: BLE001
                semi.append(dict(trial=trial, scale=sc, error=type(ex).__name__))

    np.savez_compressed(os.path.join(out_dir, f"{trial}.npz"), x=x, bm_pk=bm_pk, ref=ref, found=found,
                        live=live, thr=thr, noise=noise, bmc_pk=np.asarray(bmc.inhalePeaks, int))
    print(f"{trial}: done  noise_ptp={noise:.1f} thr={thr:.0f}", flush=True)
    return dict(repro=pd.DataFrame(repro).assign(trial=trial), sens=pd.DataFrame(sens), gap=pd.DataFrame(gap_rows),
                peaks=peak_rows, win=pd.DataFrame(win_rows), corr=pd.DataFrame(corr), semi=pd.DataFrame(semi),
                meta=pd.DataFrame([dict(trial=trial, thr=thr, noise=noise, std=float(np.std(x)),
                                        dur_s=len(x) / FS, live_frac=float(live.mean()))]))


if __name__ == "__main__":
    orig_csv, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(orig_csv)
    jobs = [(t, df[df.Trial == t][["WindowID", "Start", "Stop"]], out_dir) for t in TRIAL_FILES]
    with Pool(4) as pool:
        results = pool.map(run_trial, jobs)
    for key in results[0]:
        pd.concat([r[key] for r in results], ignore_index=True).to_csv(os.path.join(out_dir, f"{key}.csv"), index=False)
    print("stage 1 complete")
