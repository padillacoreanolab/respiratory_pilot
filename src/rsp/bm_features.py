"""
BreathMetrics per-window / per-bout features from a full-session bmObject fit.

Workflow: fit_bm_session(signal, fs) once per recording -> windows
(tile_windows / sample_random_nonoverlapping_windows, or BORIS bouts) ->
extract_features_from_windows / extract_bout_features.
"""
import numpy as np
import matplotlib.pyplot as plt

from .breathmetrics import bmObject
# ==================================================================================================================
# Breathmetrics Helper Functions
# ==================================================================================================================


# =============================================================================
# Core helper: extract per-window features from an already-fitted bmObject
# =============================================================================

def get_bout_breath_indices(
    bm,
    bout_start_sec,
    bout_stop_sec,
    strict_within_window=False,
):
    """
    Return breath indices whose inhale onsets fall in [bout_start_sec, bout_stop_sec]
    (or fully inside the window when strict_within_window=True).
    """
    start_samp = bout_start_sec * bm.srate
    stop_samp = bout_stop_sec * bm.srate
    inhale_onsets = np.asarray(bm.inhaleOnsets, dtype=float)

    if strict_within_window:
        exhale_offsets = np.asarray(bm.exhaleOffsets[0], dtype=float)
        bout_mask = (
            (inhale_onsets >= start_samp) &
            (exhale_offsets <= stop_samp) &
            (~np.isnan(exhale_offsets))
        )
    else:
        bout_mask = (inhale_onsets >= start_samp) & (inhale_onsets <= stop_samp)

    return np.where(bout_mask)[0]


def plot_dropped_bout_diagnostic(
    signal,
    time,
    fs,
    bm,
    bout_start,
    bout_stop,
    feat_start,
    feat_stop,
    *,
    min_breaths_required=2,
    behavior=np.nan,
    trial="",
    pad_sec=0.5,
    ax=None,
    show=True,
):
    """
    Plot respiration around a dropped bout with BM inhale/exhale onsets overlaid.

    Gold shading = standardized bout window; blue shading = feature-extraction window.
    Green/red markers = inhale/exhale onsets counted inside the feature window.
    """
    n_breaths = len(get_bout_breath_indices(bm, feat_start, feat_stop))
    t0 = max(time[0], min(feat_start, bout_start) - pad_sec)
    t1 = min(time[-1], max(feat_stop, bout_stop) + pad_sec)
    mask = (time >= t0) & (time <= t1)
    t_view = time[mask]
    sig_view = signal[mask]

    if ax is None:
        _, ax = plt.subplots(figsize=(14, 3))

    ax.plot(t_view, sig_view, color="0.35", lw=0.8, label="resp")

    inhale_t = np.asarray(bm.inhaleOnsets, dtype=float) / fs
    exhale_t = np.asarray(bm.exhaleOnsets, dtype=float) / fs
    feat_inds = get_bout_breath_indices(bm, feat_start, feat_stop)

    inhale_in = inhale_t[(inhale_t >= t0) & (inhale_t <= t1)]
    exhale_in = exhale_t[(exhale_t >= t0) & (exhale_t <= t1)]
    counted_inhale = inhale_t[feat_inds] if len(feat_inds) else np.array([])
    counted_exhale = exhale_t[feat_inds] if len(feat_inds) else np.array([])

    other_inhale = np.setdiff1d(inhale_in, counted_inhale)
    other_exhale = np.setdiff1d(exhale_in, counted_exhale)

    if other_inhale.size:
        ax.scatter(
            other_inhale,
            np.interp(other_inhale, t_view, sig_view),
            s=14, c="0.75", marker="o", label="inhale onset (outside feat window)",
        )
    if other_exhale.size:
        ax.scatter(
            other_exhale,
            np.interp(other_exhale, t_view, sig_view),
            s=14, c="0.85", marker="x", label="exhale onset (outside feat window)",
        )
    if counted_inhale.size:
        ax.scatter(
            counted_inhale,
            np.interp(counted_inhale, t_view, sig_view),
            s=22, c="tab:green", marker="o", label="inhale onset (counted)",
        )
    if counted_exhale.size:
        ax.scatter(
            counted_exhale,
            np.interp(counted_exhale, t_view, sig_view),
            s=22, c="tab:red", marker="x", label="exhale onset (counted)",
        )

    ax.axvspan(bout_start, bout_stop, color="gold", alpha=0.2, label="bout window")
    ax.axvspan(feat_start, feat_stop, color="tab:blue", alpha=0.12, label="feature window")
    ax.set_title(
        f"DROPPED | {trial} | {behavior} | "
        f"n_breaths={n_breaths} (need >={min_breaths_required}) | "
        f"bout {bout_start:.2f}-{bout_stop:.2f}s | feat {feat_start:.2f}-{feat_stop:.2f}s"
    )
    ax.set_xlabel("Time (s)")
    ax.legend(loc="upper right", fontsize=8)
    if show:
        plt.tight_layout()
        plt.show()
    return ax


def plot_recording_bout_overview(
    signal,
    time,
    trial,
    kept_bouts,
    dropped_bouts,
    *,
    show=True,
    figsize=(16, 4),
    max_plot_points=80000,
):
    """
    Full-session respiration overview with kept (green) and dropped (red) bout windows.

    Parameters
    ----------
    kept_bouts, dropped_bouts : list[dict]
        Each dict must have Start and Stop (seconds). Behavior is optional (used in title only).
    """
    from matplotlib.patches import Patch

    t = np.asarray(time, dtype=float)
    s = np.asarray(signal, dtype=float)
    if len(t) > max_plot_points:
        step = int(np.ceil(len(t) / max_plot_points))
        t_plot = t[::step]
        s_plot = s[::step]
    else:
        t_plot, s_plot = t, s

    fig, ax = plt.subplots(figsize=figsize)

    for bout in kept_bouts:
        ax.axvspan(
            bout["Start"], bout["Stop"],
            color="tab:green", alpha=0.22, zorder=1,
        )
    for bout in dropped_bouts:
        ax.axvspan(
            bout["Start"], bout["Stop"],
            color="tab:red", alpha=0.30, zorder=2,
        )

    ax.plot(t_plot, s_plot, color="0.25", lw=0.4, zorder=3, label="resp")
    ax.set_xlim(t[0], t[-1])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Resp")
    ax.set_title(
        f"{trial} | kept={len(kept_bouts)} (green) | dropped={len(dropped_bouts)} (red)"
    )
    ax.legend(
        handles=[
            Patch(facecolor="tab:green", alpha=0.22, label="kept bout"),
            Patch(facecolor="tab:red", alpha=0.30, label="dropped bout"),
            Patch(facecolor="0.25", label="resp"),
        ],
        loc="upper right",
        fontsize=8,
    )
    if show:
        plt.tight_layout()
        plt.show()
    return ax


def extract_bout_features(
    bm,
    bout_start_sec,
    bout_stop_sec,
    min_breaths=4,
    strict_within_window=False,
):
    """
    Given a fitted bmObject (with estimateAllFeatures already called on the FULL
    recording), aggregate respiratory features for breaths belonging to a time
    window [bout_start_sec, bout_stop_sec].

    Why we still fit BreathMetrics on the full recording first:
    -----------------------------------------------------------
    This helper does NOT detect breaths from scratch. It only:
      1) selects breaths already detected by BreathMetrics
      2) summarizes those breath-level measurements inside this window

    That full-session fit is important because BreathMetrics estimates:
      - inhale/exhale onsets
      - pauses
      - offsets
      - durations
      - volumes
    using context from neighboring breaths and surrounding signal. Those
    estimates are usually more reliable on the full recording than on tiny
    clipped windows.

    Parameters
    ----------
    bm : fitted bmObject
        BreathMetrics object already fit on the full session.
    bout_start_sec : float
        Window start time in seconds.
    bout_stop_sec : float
        Window stop time in seconds.
    min_breaths : int, default=4
        Minimum number of breaths required to keep this window.

        CHANGED:
        The old code used 2 breaths. That is enough to compute some quantities,
        but it is weak for variability-style features (CVs), pause fractions,
        and volume summaries. Using 4 makes the window features more stable.
    strict_within_window : bool, default=False
        If False:
            include a breath if its inhale onset falls inside the window.
        If True:
            require the full breath to fit within the window, using inhale onset
            and exhale offset.

        This is optional because the original behavior is reasonable for many ML
        pipelines, but strict mode is useful as a sensitivity check.

    Returns
    -------
    dict or None
        Dictionary of aggregated features, or None if the window does not
        contain enough valid breaths.
    """

    inhale_onsets = np.asarray(bm.inhaleOnsets, dtype=float)
    bout_inds = get_bout_breath_indices(
        bm, bout_start_sec, bout_stop_sec,
        strict_within_window=strict_within_window,
    )

    # CHANGED:
    # old code required only 2 breaths. We now default to 4 because:
    # - 2 breaths gives only 1 IBI
    # - CV metrics become unstable
    # - pause percentages become very coarse
    if len(bout_inds) < min_breaths:
        return None

    # n_breaths:
    #   number of breaths contributing to this window summary
    # Naming reason:
    #   simple bookkeeping feature that also helps you later inspect whether
    #   some windows are based on too little data.
    n_breaths = len(bout_inds)

    # -------------------------------------------------------------------------
    # 1) RATE / INTER-BREATH TIMING
    # -------------------------------------------------------------------------
    # bout_inhale_onsets:
    #   inhale start times for only the breaths selected in this window
    bout_inhale_onsets = inhale_onsets[bout_inds]

    # ibis_sec:
    #   inter-breath intervals in seconds, computed between consecutive inhale
    #   onsets
    # Naming reason:
    #   IBI = inter-breath interval, a standard timing measure.
    ibis_sec = np.diff(bout_inhale_onsets) / bm.srate

    # If somehow we have no interval after filtering, stop here
    if len(ibis_sec) == 0:
        return None

    # mean_ibi_sec:
    #   mean time between breath starts in seconds
    # Naming reason:
    #   explicit "_sec" suffix makes the units clear.
    mean_ibi_sec = np.nanmean(ibis_sec)

    # breathing_rate_hz:
    #   breaths per second, derived from 1 / mean IBI
    # Naming reason:
    #   "_hz" here means events per second.
    breathing_rate_hz = 1.0 / mean_ibi_sec if mean_ibi_sec > 0 else np.nan

    # cv_ibi:
    #   variability of inter-breath timing
    #
    # CHANGED:
    # old code called this "cv_breathing_rate", but mathematically it was really
    # CV of the breath intervals, not CV of rate itself. This new name is more
    # honest and easier to interpret.
    cv_ibi = np.nan
    if len(ibis_sec) >= 2 and mean_ibi_sec > 0:
        cv_ibi = np.nanstd(ibis_sec) / mean_ibi_sec

    # -------------------------------------------------------------------------
    # 2) INHALE / EXHALE DURATIONS
    # -------------------------------------------------------------------------
    # These durations were already computed by BM from onset/offset landmarks on
    # the full session, and BM stores them in seconds.
    inhale_durs_sec = np.asarray(bm.inhaleDurations[0, bout_inds], dtype=float)
    exhale_durs_sec = np.asarray(bm.exhaleDurations[0, bout_inds], dtype=float)

    inhale_durs_sec = inhale_durs_sec[~np.isnan(inhale_durs_sec)]
    exhale_durs_sec = exhale_durs_sec[~np.isnan(exhale_durs_sec)]

    # mean_inhale_dur_sec / mean_exhale_dur_sec:
    #   average duration of inhale/exhale phases in this window
    mean_inhale_dur_sec = np.nanmean(inhale_durs_sec) if inhale_durs_sec.size else np.nan
    mean_exhale_dur_sec = np.nanmean(exhale_durs_sec) if exhale_durs_sec.size else np.nan

    # CVs only computed when there are enough values to make variability at
    # least somewhat meaningful.
    cv_inhale_dur = np.nan
    if inhale_durs_sec.size >= 2 and mean_inhale_dur_sec > 0:
        cv_inhale_dur = np.nanstd(inhale_durs_sec) / mean_inhale_dur_sec

    cv_exhale_dur = np.nan
    if exhale_durs_sec.size >= 2 and mean_exhale_dur_sec > 0:
        cv_exhale_dur = np.nanstd(exhale_durs_sec) / mean_exhale_dur_sec

    # ie_ratio:
    #   inhale-to-exhale duration ratio
    # Naming reason:
    #   short for inhale/exhale ratio; common compact respiratory shorthand.
    ie_ratio = (
        mean_inhale_dur_sec / mean_exhale_dur_sec
        if (
            not np.isnan(mean_inhale_dur_sec) and
            not np.isnan(mean_exhale_dur_sec) and
            mean_exhale_dur_sec > 0
        )
        else np.nan
    )

    # -------------------------------------------------------------------------
    # 3) PEAK FLOW FEATURES
    # -------------------------------------------------------------------------
    # peakInspiratoryFlows / troughExpiratoryFlows come from BM's detected
    # extrema on the full signal.
    peak_insp_flows = np.asarray(bm.peakInspiratoryFlows[bout_inds], dtype=float)
    peak_exp_flows  = np.asarray(bm.troughExpiratoryFlows[bout_inds], dtype=float)

    # mean_peak_insp_flow / mean_peak_exp_flow:
    #   average inspiratory and expiratory extrema amplitude
    # Naming reason:
    #   kept close to original naming for compatibility with your existing
    #   pipeline and plots.
    mean_peak_insp_flow = np.nanmean(peak_insp_flows) if peak_insp_flows.size else np.nan
    mean_peak_exp_flow  = np.nanmean(peak_exp_flows) if peak_exp_flows.size else np.nan

    valid_peak_insp_flows = peak_insp_flows[~np.isnan(peak_insp_flows)]

    cv_peak_insp_flow = np.nan
    if valid_peak_insp_flows.size >= 2 and np.nanmean(valid_peak_insp_flows) != 0:
        cv_peak_insp_flow = np.nanstd(valid_peak_insp_flows) / np.nanmean(valid_peak_insp_flows)

    # -------------------------------------------------------------------------
    # 4) VOLUME FEATURES
    # -------------------------------------------------------------------------
    inhale_vols = np.asarray(bm.inhaleVolumes[0, bout_inds], dtype=float)
    exhale_vols = np.asarray(bm.exhaleVolumes[0, bout_inds], dtype=float)

    inhale_vols = inhale_vols[~np.isnan(inhale_vols)]
    exhale_vols = exhale_vols[~np.isnan(exhale_vols)]

    mean_inhale_vol = np.nanmean(inhale_vols) if inhale_vols.size else np.nan
    mean_exhale_vol = np.nanmean(exhale_vols) if exhale_vols.size else np.nan

    # CHANGED:
    # old name: mean_tidal_vol
    #
    # Reason for rename:
    # this is not a perfectly paired per-breath tidal volume computation.
    # It is mean inhale volume + mean exhale volume, so "proxy" is more honest.
    mean_tidal_vol_proxy = (
        mean_inhale_vol + mean_exhale_vol
        if (not np.isnan(mean_inhale_vol) and not np.isnan(mean_exhale_vol))
        else np.nan
    )

    # CHANGED:
    # old code called this cv_tidal_vol, but it actually used inhale volumes
    # only. Renaming it to cv_inhale_vol matches the math.
    cv_inhale_vol = np.nan
    if inhale_vols.size >= 2 and mean_inhale_vol > 0:
        cv_inhale_vol = np.nanstd(inhale_vols) / mean_inhale_vol

    # CHANGED:
    # old name: minute_ventilation
    #
    # Reason for rename:
    # breathing_rate_hz is breaths/second, so multiplying by volume gives a
    # per-second proxy, not a literal per-minute quantity. This is still useful
    # for ML, but the old name overclaimed physiological precision.
    ventilation_proxy = (
        breathing_rate_hz * mean_tidal_vol_proxy
        if (not np.isnan(breathing_rate_hz) and not np.isnan(mean_tidal_vol_proxy))
        else np.nan
    )

    # -------------------------------------------------------------------------
    # 5) PAUSE FEATURES
    # -------------------------------------------------------------------------
    inh_pause_durs_sec = np.asarray(bm.inhalePauseDurations[0, bout_inds], dtype=float)
    exh_pause_durs_sec = np.asarray(bm.exhalePauseDurations[0, bout_inds], dtype=float)

    inh_pause_durs_sec = inh_pause_durs_sec[~np.isnan(inh_pause_durs_sec)]
    exh_pause_durs_sec = exh_pause_durs_sec[~np.isnan(exh_pause_durs_sec)]

    # pct_breaths_with_inhale_pause / pct_breaths_with_exhale_pause:
    #   fraction of breaths in this window that contain that pause type
    pct_breaths_with_inhale_pause = len(inh_pause_durs_sec) / n_breaths
    pct_breaths_with_exhale_pause = len(exh_pause_durs_sec) / n_breaths

    # If no pauses exist, using 0.0 is often more interpretable than NaN for
    # the mean pause duration itself.
    mean_inhale_pause_dur_sec = np.nanmean(inh_pause_durs_sec) if inh_pause_durs_sec.size else 0.0
    mean_exhale_pause_dur_sec = np.nanmean(exh_pause_durs_sec) if exh_pause_durs_sec.size else 0.0

    cv_inhale_pause_dur = np.nan
    if inh_pause_durs_sec.size >= 2 and mean_inhale_pause_dur_sec > 0:
        cv_inhale_pause_dur = np.nanstd(inh_pause_durs_sec) / mean_inhale_pause_dur_sec

    cv_exhale_pause_dur = np.nan
    if exh_pause_durs_sec.size >= 2 and mean_exhale_pause_dur_sec > 0:
        cv_exhale_pause_dur = np.nanstd(exh_pause_durs_sec) / mean_exhale_pause_dur_sec

    # -------------------------------------------------------------------------
    # 6) DUTY CYCLES
    # -------------------------------------------------------------------------
    # Duty cycle = fraction of the average breath cycle spent in a phase.
    inhale_duty_cycle = mean_inhale_dur_sec / mean_ibi_sec if mean_ibi_sec > 0 else np.nan
    exhale_duty_cycle = mean_exhale_dur_sec / mean_ibi_sec if mean_ibi_sec > 0 else np.nan

    # Pause duty cycles scale the mean pause duration by the fraction of breaths
    # that actually contained the pause, which matches the logic used in the BM
    # secondary-feature code.
    inhale_pause_duty_cycle = (
        (mean_inhale_pause_dur_sec * pct_breaths_with_inhale_pause) / mean_ibi_sec
        if mean_ibi_sec > 0 else np.nan
    )
    exhale_pause_duty_cycle = (
        (mean_exhale_pause_dur_sec * pct_breaths_with_exhale_pause) / mean_ibi_sec
        if mean_ibi_sec > 0 else np.nan
    )

    # -------------------------------------------------------------------------
    # Return a flat dictionary for downstream dataframe assembly
    # -------------------------------------------------------------------------
    return {
        # ---------------------------------------------------------------------
        # bookkeeping
        # ---------------------------------------------------------------------
        "n_breaths": n_breaths,

        # ---------------------------------------------------------------------
        # rate / timing
        # ---------------------------------------------------------------------
        "breathing_rate_hz": breathing_rate_hz,
        "mean_ibi_sec": mean_ibi_sec,
        "cv_ibi": cv_ibi,

        # ---------------------------------------------------------------------
        # inhale phase
        # ---------------------------------------------------------------------
        "mean_inhale_dur_sec": mean_inhale_dur_sec,
        "cv_inhale_dur": cv_inhale_dur,

        # ---------------------------------------------------------------------
        # exhale phase
        # ---------------------------------------------------------------------
        "mean_exhale_dur_sec": mean_exhale_dur_sec,
        "cv_exhale_dur": cv_exhale_dur,

        # ---------------------------------------------------------------------
        # inhale/exhale timing relationship
        # ---------------------------------------------------------------------
        "ie_ratio": ie_ratio,

        # ---------------------------------------------------------------------
        # peak flow summaries
        # ---------------------------------------------------------------------
        "mean_peak_insp_flow": mean_peak_insp_flow,
        "mean_peak_exp_flow": mean_peak_exp_flow,
        "cv_peak_insp_flow": cv_peak_insp_flow,

        # ---------------------------------------------------------------------
        # volume summaries
        # ---------------------------------------------------------------------
        "mean_inhale_vol": mean_inhale_vol,
        "mean_exhale_vol": mean_exhale_vol,
        "mean_tidal_vol_proxy": mean_tidal_vol_proxy,
        "cv_inhale_vol": cv_inhale_vol,
        "ventilation_proxy": ventilation_proxy,

        # ---------------------------------------------------------------------
        # inhale pause summaries
        # ---------------------------------------------------------------------
        "pct_breaths_with_inhale_pause": pct_breaths_with_inhale_pause,
        "mean_inhale_pause_dur_sec": mean_inhale_pause_dur_sec,
        "cv_inhale_pause_dur": cv_inhale_pause_dur,
        "inhale_pause_duty_cycle": inhale_pause_duty_cycle,

        # ---------------------------------------------------------------------
        # exhale pause summaries
        # ---------------------------------------------------------------------
        "pct_breaths_with_exhale_pause": pct_breaths_with_exhale_pause,
        "mean_exhale_pause_dur_sec": mean_exhale_pause_dur_sec,
        "cv_exhale_pause_dur": cv_exhale_pause_dur,
        "exhale_pause_duty_cycle": exhale_pause_duty_cycle,

        # ---------------------------------------------------------------------
        # overall phase fractions
        # ---------------------------------------------------------------------
        "inhale_duty_cycle": inhale_duty_cycle,
        "exhale_duty_cycle": exhale_duty_cycle,
    }


# =============================================================================
# Session-level fitter
# =============================================================================

def fit_bm_session(signal, srate, data_type="rodentAirflow", peak_method="src"):
    """
    Fits a bmObject on a full session signal.

    Parameters
    ----------
    signal      : 1-D np.ndarray, cleaned respiration (e.g. resp_processing.preprocess_resp)
    srate       : float, sampling rate in Hz
    data_type   : str, passed to bmObject (default 'rodentAirflow')
    peak_method : 'src' (resp_processing.detect_breaths) or 'legacy' (original
                  BreathMetrics search; reproduces feature files made before 2026-11-24)

    Returns
    -------
    bm : fitted bmObject, or None on failure
    """
    try:
        bm = bmObject(signal, srate, data_type)
        bm.estimateAllFeatures(peak_method=peak_method)
        return bm
    except Exception as e:
        print(f"  bmObject fitting failed: {e}")
        return None


# =============================================================================
# Metadata (non-feature) columns in the feature matrices
# =============================================================================

METADATA_COLS = [
    "BoutID", "Behavior", "Start", "Stop", "Duration", "Label",
    "PreBoutRespRate", "Trial", "Subject", "Condition", "Rank",
    "Type", "resp_rate_delta", "breathing_rate_z",
]

# Window feature columns that are measured from breath timing. Every OTHER feature
# column (durations, duty cycles, I:E, flows, volumes, pauses) is estimated; see
# breathmetrics.UNVERIFIED_FEATURES for the reasons.
SUPPORTED_WINDOW_FEATURES = ["n_breaths", "breathing_rate_hz", "mean_ibi_sec", "cv_ibi"]


# =============================================================================
# Creates Windows Throughout Sessions
# =============================================================================

def tile_windows(time, window_sec=2.0, step_sec=None, signal=None, dead_frac=0.02):
    """
    Consecutive (start, stop) windows covering the whole recording.

    Parameters
    ----------
    time       : session time vector (s)
    window_sec : window length (s)
    step_sec   : start-to-start spacing (s); None = window_sec (no overlap),
                 smaller = overlapping windows
    signal     : optional respiration aligned to time; if given, drop windows whose
                 std or range is below dead_frac x the session's (dead signal; same
                 gate as sample_random_nonoverlapping_windows)

    Returns
    -------
    list of (start, stop) tuples, in time order
    """
    step = window_sec if step_sec is None else step_sec
    t0, t1 = float(time[0]), float(time[-1])
    if t1 - t0 < window_sec:
        return []
    # + 1e-9 IOT keep the last window when the length is an exact multiple
    n = int(np.floor((t1 - t0 - window_sec) / step + 1e-9)) + 1
    windows = [(t0 + i * step, t0 + i * step + window_sec) for i in range(n)]
    if signal is None:
        return windows

    signal = np.asarray(signal, dtype=float)
    min_std = max(1e-6, dead_frac * np.nanstd(signal))
    min_ptp = max(1e-6, dead_frac * (np.nanmax(signal) - np.nanmin(signal)))
    kept = []
    for s, e in windows:
        seg = signal[(time >= s) & (time < e)]
        if seg.size and np.nanstd(seg) >= min_std and np.nanmax(seg) - np.nanmin(seg) >= min_ptp:
            kept.append((s, e))
    return kept

def sample_random_nonoverlapping_windows(
    time,
    signal=None,
    window_dur=5.0,
    n_windows=100,
    seed=42,
    allow_partial_if_short=False,
    dead_std_frac=0.02,
    dead_ptp_frac=0.02,
    dead_abs_std_floor=1e-6,
    dead_abs_ptp_floor=1e-6,
):
    """
    Sample up to n_windows non-overlapping fixed-duration windows from a session.
    Optionally rejects windows that appear "dead" (near-flat signal).
    Sampling is spread-aware: it first draws across temporal strata, then fills
    any remaining slots with a distance-weighted random draw to reduce clumping.

    Parameters
    ----------
    time : np.ndarray
        Session time vector in seconds.
    signal : np.ndarray or None
        Respiration signal aligned to `time`. If provided, windows whose
        within-window variability is too low are excluded before random draw.
    window_dur : float
        Window duration in seconds.
    n_windows : int
        Maximum number of windows to sample.
    seed : int
        Random seed for reproducibility.
    allow_partial_if_short : bool
        If True and session is shorter than one full window, return one
        clamped session-spanning window. If False, return [].
    dead_std_frac : float
        Fraction of global signal std used as the minimum window std threshold.
    dead_ptp_frac : float
        Fraction of global signal peak-to-peak range used as minimum window
        range threshold.
    dead_abs_std_floor : float
        Absolute lower bound for window std threshold.
    dead_abs_ptp_floor : float
        Absolute lower bound for window range threshold.

    Returns
    -------
    windows : list of (start, stop) tuples
    """
    rng = np.random.default_rng(seed)

    t_start = float(time[0])
    t_end   = float(time[-1])
    session_dur = t_end - t_start

    if session_dur < window_dur:
        if allow_partial_if_short:
            return [(t_start, t_end)]
        return []

    # candidate starts for full non-overlapping windows
    possible_starts = np.arange(t_start, t_end - window_dur, window_dur)

    if len(possible_starts) == 0:
        return []

    # If signal is provided, drop candidate windows with very low variation.
    if signal is not None:
        signal = np.asarray(signal)
        if len(signal) != len(time):
            raise ValueError("signal and time must have the same length")

        global_std = float(np.nanstd(signal))
        global_ptp = float(np.nanmax(signal) - np.nanmin(signal))
        min_std = max(dead_abs_std_floor, dead_std_frac * global_std)
        min_ptp = max(dead_abs_ptp_floor, dead_ptp_frac * global_ptp)

        valid_starts = []
        for s in possible_starts:
            e = s + window_dur
            mask = (time >= s) & (time < e)
            seg = signal[mask]
            if seg.size == 0:
                continue

            seg_std = float(np.nanstd(seg))
            seg_ptp = float(np.nanmax(seg) - np.nanmin(seg))
            if np.isfinite(seg_std) and np.isfinite(seg_ptp):
                if (seg_std >= min_std) and (seg_ptp >= min_ptp):
                    valid_starts.append(float(s))

        possible_starts = np.array(valid_starts, dtype=float)
        if len(possible_starts) == 0:
            return []

    n_use = min(n_windows, len(possible_starts))

    # Spread-aware random sampling:
    # 1) stratified random draw over session timeline
    # 2) if some strata are empty, fill remaining slots with a
    #    distance-weighted random draw to avoid local clustering
    starts_sorted = np.sort(possible_starts.astype(float))
    session_last_start = t_end - window_dur
    bin_edges = np.linspace(t_start, session_last_start, n_use + 1)

    chosen_starts = []
    chosen_set = set()
    for i in range(n_use):
        left = bin_edges[i]
        right = bin_edges[i + 1]
        if i == n_use - 1:
            in_bin = starts_sorted[(starts_sorted >= left) & (starts_sorted <= right)]
        else:
            in_bin = starts_sorted[(starts_sorted >= left) & (starts_sorted < right)]
        if in_bin.size == 0:
            continue
        s = float(rng.choice(in_bin))
        chosen_starts.append(s)
        chosen_set.add(s)

    if len(chosen_starts) < n_use:
        remaining = starts_sorted[~np.isin(starts_sorted, np.array(chosen_starts, dtype=float))]
        while len(chosen_starts) < n_use and remaining.size > 0:
            if len(chosen_starts) == 0:
                idx = int(rng.integers(0, remaining.size))
            else:
                chosen_arr = np.array(chosen_starts, dtype=float)
                # Favor candidates farther from already selected windows.
                min_dists = np.min(np.abs(remaining[:, None] - chosen_arr[None, :]), axis=1)
                weights = min_dists + 1e-12
                probs = weights / weights.sum()
                idx = int(rng.choice(np.arange(remaining.size), p=probs))

            s = float(remaining[idx])
            chosen_starts.append(s)
            chosen_set.add(s)
            remaining = remaining[remaining != s]

    windows = sorted((float(s), float(s + window_dur)) for s in chosen_starts)
    return windows


def extract_features_from_windows(
    bm,
    windows,
    min_breaths_per_window=2,
):
    """
    Extract respiratory features for a list of (start, stop) windows
    from an already-fitted bmObject.

    Parameters
    ----------
    bm : fitted bmObject
    windows : list of (start, stop)
    min_breaths_per_window : int
        Minimum breaths required to keep a window.

    Returns
    -------
    rows : list of dict
        Each dict contains Start, Stop, Duration, and extracted features.
    """
    rows = []

    for win_id, (start, stop) in enumerate(windows):
        feats = extract_bout_features(bm, start, stop)

        if feats is None:
            continue
        if feats["n_breaths"] < min_breaths_per_window:
            continue

        row = {
            "WindowID": win_id,
            "Start": start,
            "Stop": stop,
            "Duration": stop - start,
        }
        row.update(feats)
        rows.append(row)

    return rows
