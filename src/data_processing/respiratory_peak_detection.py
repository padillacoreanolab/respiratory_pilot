import neurokit2 as nk
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt

def peakfinder(x0, sel=None, thresh=None, extrema=1, include_endpoints=True, interpolate=False):
    """
Python version of the MATLAB peakfinder function.                                                                       
Purpose: Identifies peaks (maxima or minima) in a signal.
Parameters:
x0: The signal in which peaks will be detected.

sel: A threshold value used to define what counts as a peak.

thresh: A minimum magnitude threshold.

extrema: Whether to find maxima (1) or minima (-1).

include_endpoints: Whether to include the endpoints as potential peaks.

interpolate: Whether to interpolate the peaks to improve accuracy.

How it works:
First, it computes the derivative of the signal and finds zero crossings, which indicates possible peak locations.
Then, it searches for local maxima (or minima) that exceed a certain threshold.

If interpolation is enabled, it refines the peak location by fitting a quadratic curve around the detected peaks.
Output: Returns the indices and magnitudes of the detected peaks.
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
