import neurokit2 as nk
import pandas as pd
import numpy as np
import warnings

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

