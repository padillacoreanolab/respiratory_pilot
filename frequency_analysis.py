import neurokit2 as nk
import numpy as np
import pandas as pd

"""
def detect_sniffing(resp_signal, sampling_rate, baseline_duration, threshold_factor):
    """
"""
    Detect significant increases in sniffing frequency from respiratory data.

    Parameters:
        resp_signal (array): The raw or cleaned respiratory signal.
        sampling_rate (float): The sampling frequency of the signal in Hz.
        baseline_duration (int): The duration (in seconds) to calculate baseline sniffing frequency.
        threshold_factor (float): Factor above baseline to classify as a significant increase.

    Returns:
        DataFrame: Times and frequencies of significant sniffing increases.
    """
"""
    # Step 1: Clean the respiratory signal
    cleaned_resp = nk.rsp_clean(resp_signal, sampling_rate=sampling_rate)
    
    # Step 2: Detect breathing peaks
    peaks, _ = nk.signal_findpeaks(cleaned_resp, sampling_rate=sampling_rate)
    peak_times = peaks / sampling_rate  # Convert peak indices to times
    
    # Step 3: Calculate instantaneous sniffing frequency
    peak_intervals = np.diff(peak_times)  # Time intervals between successive peaks
    sniffing_freq = 1 / peak_intervals  # Frequency in Hz
    sniffing_times = peak_times[:-1]  # Associated times for frequencies
    
    # Step 4: Calculate baseline sniffing frequency
    baseline_freq = np.mean(sniffing_freq[sniffing_times <= baseline_duration])
    threshold = baseline_freq * threshold_factor
    
    # Step 5: Identify significant increases
    significant_indices = np.where(sniffing_freq > threshold)[0]
    significant_times = sniffing_times[significant_indices]
    significant_frequencies = sniffing_freq[significant_indices]
    
    # Step 6: Create a DataFrame with results
    results = pd.DataFrame({
        "Time (s)": significant_times,
        "Frequency (Hz)": significant_frequencies
    })
    
    return results
"""

def compute_sniffing_frequency(resp_signal, sampling_rate, window_size):
    """
    Computes the sniffing frequency over sliding windows.
    
    Parameters:
        resp_signal (array): Respiratory signal data.
        sampling_rate (float): Sampling rate in Hz.
        window_size (float): Window size in seconds.
        
    Returns:
        list: Sniffing frequency for each window.
    """
    # Convert window size to number of samples
    window_samples = int(window_size * sampling_rate)
    
    # Clean the respiratory signal
    cleaned_resp = nk.rsp_clean(resp_signal, sampling_rate)
    
    # Detect peaks
    peaks_dict = nk.signal_findpeaks(cleaned_resp)
    peak_indices = peaks_dict['Peaks']  # Extract peak indices
    print(peak_indices)
    print(peaks_dict['Peaks'])

    
    peak_times = peak_indices / sampling_rate  # Convert indices to time
    
    # Calculate sniffing frequency for each window
    sniffing_frequencies = []
    for start in range(0, len(resp_signal) - window_samples, window_samples):
        # Define window start and end times
        start_time = start / sampling_rate
        end_time = (start + window_samples) / sampling_rate
        
        # Get peaks within the current window
        window_peaks = peak_times[(peak_times >= start_time) & (peak_times < end_time)]
        
        # Calculate frequency if enough peaks are detected
        if len(window_peaks) < 2:
            sniffing_frequencies.append(0)  # No frequency detected
        else:
            intervals = np.diff(window_peaks)  # Intervals between peaks
            sniffing_frequencies.append(np.mean(1 / intervals))  # Frequency in Hz
    
    return sniffing_frequencies


resp_signal = np.sin(np.linspace(0, 10 * np.pi, 1000)) + 0.1 * np.random.randn(1000)  # Simulated noisy signal
sampling_rate = 100  # 100 Hz

print(compute_sniffing_frequency(resp_signal=resp_signal, sampling_rate=sampling_rate, window_size=20))