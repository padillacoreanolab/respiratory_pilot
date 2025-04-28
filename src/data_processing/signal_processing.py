import neurokit2 as nk
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt

def mice_rsp_clean(rsp_signal, sampling_rate):
    """
    Custom `rsp_clean()` function with a specific filtering method.

    Parameters
    ----------
    rsp_signal : array-like
        The raw respiratory signal to clean.
    sampling_rate : int
        The sampling frequency of the signal in Hz.

    Returns
    -------
    rsp_cleaned : array-like
        The cleaned respiratory signal.
    """
    # Apply the custom filtering logic
    rsp_cleaned = nk.signal_filter(
        rsp_signal,
        lowcut=0.1,   # Low cutoff frequency (Hz)
        highcut=20,   # High cutoff frequency (Hz)
        method="butterworth",  # Filtering method
        sampling_rate=sampling_rate,
        order=2  # Filter order
    )
    return rsp_cleaned

def peakfinder(x0, sel=None, thresh=None, extrema=1, include_endpoints=True, interpolate=False):
    """
    Python version of the MATLAB peakfinder function.

    Parameters:
        x0: array-like
            A real vector where peaks will be identified.
        sel: float, optional
            The selectivity threshold. Peaks must be this much above surrounding data.
            Default is (max(x0) - min(x0)) / 4.
        thresh: float, optional
            Peaks must be larger (or smaller for minima) than this value.
        extrema: int, optional
            1 to find maxima, -1 to find minima. Default is 1.
        include_endpoints: bool, optional
            Include endpoints as possible extrema. Default is True.
        interpolate: bool, optional
            Perform quadratic interpolation around each extrema. Default is False.

    Returns:
        peak_inds: numpy.ndarray
            Indices of the identified peaks.
        peak_mags: numpy.ndarray
            Magnitudes of the identified peaks.
    """
    x0 = np.asarray(x0, dtype=float)
    
    if x0.ndim != 1:
        raise ValueError("Input data must be a 1D array.")
    if not np.isreal(x0).all():
        warnings.warn("Absolute value of data will be used.", RuntimeWarning)
        x0 = np.abs(x0)
    
    if sel is None:
        sel = (np.max(x0) - np.min(x0)) / 4
    if thresh is None:
        thresh = np.nan
    if extrema not in [1, -1]:
        raise ValueError("extrema must be 1 (maxima) or -1 (minima).")
    
    x0 = extrema * x0  # Adjust for finding maxima/minima
    thresh *= extrema
    
    # Compute first derivative and find zero crossings
    dx0 = np.diff(x0)
    dx0[dx0 == 0] = -np.finfo(float).eps  # Handle repeated values
    ind = np.where(dx0[:-1] * dx0[1:] < 0)[0] + 1

    # Include endpoints if needed
    if include_endpoints:
        ind = np.concatenate(([0], ind, [len(x0) - 1]))

    x = x0[ind]
    min_mag = np.min(x)
    left_min = min_mag

    peak_locs = []
    peak_mags = []
    found_peak = False
    temp_mag = min_mag
    
    # Peak finding loop
    for i in range(len(x) - 1):
        if found_peak:
            temp_mag = min_mag
            found_peak = False
        
        if x[i] > temp_mag and x[i] > left_min + sel:
            temp_loc = i
            temp_mag = x[i]
        
        if i + 1 < len(x) and x[i + 1] < left_min:
            left_min = x[i + 1]

        if x[i] > left_min + sel:
            found_peak = True
            peak_locs.append(ind[temp_loc])
            peak_mags.append(temp_mag)

    # Interpolate if needed
    if interpolate and len(peak_locs) > 0:
        peak_locs = np.array(peak_locs, dtype=float)
        peak_mags = np.array(peak_mags, dtype=float)
        for i in range(len(peak_locs)):
            if 1 <= peak_locs[i] < len(x0) - 1:
                x1, x2, x3 = x0[int(peak_locs[i]) - 1:int(peak_locs[i]) + 2]
                denom = 2 * (x1 - 2 * x2 + x3)
                if denom != 0:
                    peak_locs[i] += (x1 - x3) / denom
                    peak_mags[i] += ((x1 - x3) * (x1 - x3)) / (8 * denom)
    
    # Convert to numpy arrays
    peak_locs = np.array(peak_locs, dtype=float)
    peak_mags = np.array(peak_mags, dtype=float)

    # Apply threshold
    if not np.isnan(thresh):
        mask = peak_mags > thresh
        peak_locs = peak_locs[mask]
        peak_mags = peak_mags[mask]

    return peak_locs, peak_mags


def find_high_frequency_regions(resp_df, fs, window_size=5, threshold_percentile=30):
    """
    Identifies regions in the respiratory signal where the breathing frequency is higher.

    Parameters:
    -----------
    resp_df : DataFrame
        DataFrame containing 'Timestamp (s)' and 'Respiration Value'.
    fs : float
        Sampling frequency in Hz.
    window_size : int, optional
        Number of peaks to use for the moving average of inter-peak distances (default: 5).
    threshold_percentile : float, optional
        Percentile value to determine high-frequency regions (default: 30 means areas with the lowest 30% of inter-peak intervals).
    show_plot : bool, optional
        Whether to visualize the detected high-frequency regions (default: True).

    Returns:
    --------
    high_freq_regions : list of tuples
        A list of (start_time, end_time) tuples representing high-frequency breathing regions.
    """

    # Extract respiration signal
    time = resp_df["Timestamp (s)"].values
    resp_values = resp_df["Respiration Value"].values

    # Detect peaks using peakfinder
    peak_inds, _ = peakfinder(resp_values)

    # Get peak times
    peak_times = time[peak_inds.astype(int)]

    # Compute inter-peak intervals (IPI)
    inter_peak_intervals = np.diff(peak_times)  # Time differences between peaks

    # Smooth using a moving average
    if len(inter_peak_intervals) >= window_size:
        smoothed_ipi = np.convolve(inter_peak_intervals, np.ones(window_size) / window_size, mode="valid")
    else:
        smoothed_ipi = inter_peak_intervals  # No smoothing if not enough peaks

    # Find the threshold based on the percentile
    threshold = np.percentile(smoothed_ipi, threshold_percentile)

    # Identify high-frequency regions (where IPI is below the threshold)
    high_freq_regions = []
    start_time = None

    for i, interval in enumerate(smoothed_ipi):
        if interval < threshold:
            if start_time is None:
                start_time = peak_times[i]  # Start of high-frequency region
        else:
            if start_time is not None:
                end_time = peak_times[i]  # End of high-frequency region
                high_freq_regions.append((start_time, end_time))
                start_time = None

    # Add the last region if it extends to the end
    if start_time is not None:
        high_freq_regions.append((start_time, peak_times[-1]))

    return high_freq_regions


def compute_frequency_change(resp_df, fs, pre_event_window=2, post_event_window=2):
    """
    Computes % change in respiratory frequency (Hz) before, during, and after each behavioral event.

    Parameters:
    -----------
    resp_df : DataFrame
        DataFrame containing 'Timestamp (s)', 'Respiration Value', 'Behavioral Event', and 'peak'.
    fs : float
        Sampling frequency in Hz.
    pre_event_window : int, optional
        Time (seconds) to calculate baseline frequency before the event (default: 2).
    post_event_window : int, optional
        Time (seconds) to calculate frequency after the event (default: 2).

    Returns:
    --------
    behavior_freq_changes : DataFrame
        DataFrame containing % frequency change before, during, and after each behavioral event.
    """

    # Extract peak times
    peak_times = resp_df.loc[resp_df["peak"] == 1, "Timestamp (s)"].values

    # Compute inter-peak intervals (IPI) in seconds
    inter_peak_intervals = np.diff(peak_times)

    # Compute breathing frequency (Hz) from IPI
    breathing_freq = 1 / inter_peak_intervals

    # Associate frequencies with time (shifted to match peaks)
    freq_times = peak_times[1:]  # Ignore first peak since IPI starts at 2nd peak

    # Prepare output DataFrame
    results = []

    # Iterate through unique behavioral events
    for behavior in resp_df["Behavioral Event"].dropna().unique():
        behavior_mask = resp_df["Behavioral Event"] == behavior
        behavior_times = resp_df.loc[behavior_mask, "Timestamp (s)"].values

        for start_time in behavior_times:
            end_time = start_time + 1  # Assuming behavioral events last at least 1s

            # Define time ranges
            pre_start, pre_end = start_time - pre_event_window, start_time
            post_start, post_end = end_time, end_time + post_event_window

            # Get frequency values in each period
            pre_freqs = breathing_freq[(freq_times >= pre_start) & (freq_times < pre_end)]
            during_freqs = breathing_freq[(freq_times >= start_time) & (freq_times <= end_time)]
            post_freqs = breathing_freq[(freq_times >= post_start) & (freq_times <= post_end)]

            # Compute average frequencies
            avg_pre = np.mean(pre_freqs) if len(pre_freqs) > 0 else np.nan
            avg_during = np.mean(during_freqs) if len(during_freqs) > 0 else np.nan
            avg_post = np.mean(post_freqs) if len(post_freqs) > 0 else np.nan

            # Compute % change
            change_pre_during = ((avg_during - avg_pre) / avg_pre * 100) if avg_pre else np.nan
            change_during_post = ((avg_post - avg_during) / avg_during * 100) if avg_during else np.nan

            # Store result
            results.append({
                "Behavior": behavior,
                "Start Time": start_time,
                "End Time": end_time,
                "Freq Before (Hz)": avg_pre,
                "Freq During (Hz)": avg_during,
                "Freq After (Hz)": avg_post,
                "% Change Before-During": change_pre_during,
                "% Change During-After": change_during_post
            })

    # Convert to DataFrame
    behavior_freq_changes = pd.DataFrame(results)
    
    return behavior_freq_changes
