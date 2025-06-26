import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rsp.data_processing.signal_processing import peakfinder
from scipy.signal import find_peaks, welch
import warnings

class Rsp_Kit:

	def __init__(self, resp_data, fs, size, start_time=0, end_time=None, behavior_data=None):
		"""
		Initialize the Rsp_Kit object with respiratory data and optional behavioral data.
		Respiratory data is expected to be a 1D numpy array.
		Attributes:
		-----------
		resp_data : np.ndarray
			1D array of respiratory signal data.
		fs : float
			Sampling frequency in Hz.
		start_time : float
			Start time of the recording in seconds.
		size : float
			Total duration of the recording in samples (not seconds).
		end_time : float, optional
			End time of the recording in seconds. If not provided, defaults to start_time + size
		behavior_data : pd.DataFrame, optional
			DataFrame containing behavioral events with columns 'Behavior', 'Start (s)', and 'Stop
			 (s)'.
		Parameters:
		-----------
		resp_data : np.ndarray
			1D array of respiratory signal data.
		fs : float
			Sampling frequency in Hz.
		start_time : float
			Start time of the recording in seconds.
		size : float
			Total duration of the recording in samples (not seconds).
			For example, if fs is 100 Hz and size is 100,000
			it means the recording lasts for 1000 second or 16.6 minutes.
		end_time : float, optional
			End time of the recording in seconds. If not provided, defaults to start_time + size.
		behavior_data : pd.DataFrame, optional
			DataFrame containing behavioral events with columns 'Behavior', 'Start (s)', and 'Stop (s)'.
			If not provided, no behavioral data will be included in the DataFrame.
		"""

		# Attributes used to help create dataframe in methods
		self.fs = fs
		self.start_time = start_time
		self.size = size
		self.resp_data = resp_data
		self.behavior_data = behavior_data

		self.df = None
		self.time = None
		self.timestamps = None
		self.behavioral_maps = None
		self.peaks = None
		self.high_frequency_areas = None
		
		# Creates Initial Dataframe with no behavior data
		self.generate_timestamps()
		self._frequencies()

		# If behavior data is provided, map behaviors to timestamps
		if behavior_data is not None:
			self._create_df()
			self._map_behaviors() 
		else:
			self._create_df()


	# Creating Time array for Dataframe, or for plot.
	def generate_timestamps(self, seconds = None):
		"""
		Generates timestamps for the respiratory data based on the sampling frequency and start time.
		Parameters:
		-----------
		seconds : int, optional
			If provided, generates timestamps for the specified number of seconds.
			If None, uses the size attribute to determine the duration.
		"""
		if seconds is not None:
			# If seconds is provided, calculate the number of samples based on fs
			n_samples = int(seconds * self.fs)
			self.size = n_samples / self.fs
		else:	
			# Creating total time array
			n_samples = len(self.resp_data)
			self.time = np.arange(n_samples) / self.fs + self.start_time

		# Creating time array using duration
		total_duration = self.size/self.fs
		self.timestamps = np.arange(0, total_duration, 1/self.fs)  # Create time axis

	# Tuple of timestamps with high frequencies (relevant plotting areas)
	def _frequencies(self):
		pass

	# Returns np array of behaviors mapped to correct timestamps
	def _map_behaviors(self):
		""" 
		Maps behavioral events to the respiratory data timestamps.
		Iterates through the behavior data and assigns the behavior to the corresponding timestamps in the respiratory data.
		"""
		if self.behavior_data is None:
			return None

		behav_data = self.behavior_data[['Behavior', 'Start (s)', 'Stop (s)']]
		for _, row in behav_data.iterrows():
			start_time = row["Start (s)"]
			stop_time = row["Stop (s)"]
			behavior = row["Behavior"]

			# Mask: Find respiration timestamps that fall within the behavior window
			mask = (self.df["Timestamps"] >= start_time) & (self.df["Timestamps"] <= stop_time)

			# Assign the behavior to matching respiration timestamps
			self.df.loc[mask, "Behavioral Event"] = behavior

			# Reset start_time
			start_time = 0


	# Use data_processing/frequency_analysis peakfinder
	# returns np array of peaks
	def _find_peaks(self, x0, sel=None, thresh=None, extrema=1, include_endpoints=True, interpolate=False):
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


	def update_peaks(self, sel=None, thresh=None, extrema=1, include_endpoints=True, interpolate=False):
		"""
		Updates self.peaks with indices of peaks found in the respiration data.
		Parameters:
		-----------
		sel : float, optional
			Selectivity threshold for peak detection. If None, defaults to (max - min) / 4.
		thresh : float, optional
			Minimum threshold for detected peaks. If None, no threshold is applied.
		extrema : int, optional
			1 to find maxima, -1 to find minima. Default is 1 (maxima).
		include_endpoints : bool, optional
			Whether to include endpoints as possible extrema. Default is True.
		interpolate : bool, optional
			Whether to perform quadratic interpolation around each extrema. Default is False. | Should be probably always be False? double check this.
		"""
		peak_inds, _ = self._find_peaks(
			self.resp_data,
			sel=sel,
			thresh=thresh,
			extrema=extrema,
			include_endpoints=include_endpoints,
			interpolate=interpolate
		)
		self.peaks = np.full(len(self.resp_data), np.nan)
		self.peaks[peak_inds.astype(int)] = self.resp_data[peak_inds.astype(int)]


	# Creates the dataframe using all previous data
	def _create_df(self):
		"""
		Creates a DataFrame with timestamps, respiration data, behavioral events, and peaks.
		"""
		
		# resp_data should always be present so we base the length of the necessary NaN padding on it
		max_len = len(self.resp_data)

		# Pad missing data with NaN
		self.behavior_data= self.behavior_data if self.behavior_data is not None else [np.nan] * max_len
		if self.peaks is None:
			self.update_peaks()


		# Create the DataFrame
		self.df = pd.DataFrame({
			'Timestamps': self.timestamps,
			'Respiration Value': self.resp_data,  # Always present
			'Behavioral Event': self.behavioral_maps,
			'Peaks': self.peaks
		})



	# final_resp_plot code
	def plot(self, start_time=0, duration=10, figsize=(12, 6), show_peaks=False, 
					peak_sel=None, peak_thresh=None, behavior_data=False, show_high_freq_regions=False, 
					high_freq_window=5, high_freq_threshold_percentile=30, title="Respiratory Signal"):
		"""
		Plot respiratory data for a given duration with optional peak detection, behavior event overlays,
		and high-frequency breathing region identification.

		Parameters:
		-----------
		resp_df : DataFrame
			DataFrame containing 'Timestamp', 'Respiration Value', and optionally 'Behavioral Event'.
		fs : float
			Sampling frequency in Hz.
		start_time : float
			Start time of the recording.
		duration : int, optional
			Duration of time (in seconds) to display in the plot (default: 10).
		figsize : tuple, optional
			Figure size (width, height) in inches.
		show_peaks : bool, optional
			Whether to detect and plot peaks in the respiratory signal (default: False).
		peak_sel : float, optional
			Selectivity threshold for peak detection. If None, defaults to (max - min) / 4.
		peak_thresh : float, optional
			Minimum threshold for detected peaks.
		behavioral_data : bool, optional
			Whether to overlay behavioral event data (default: False).
		show_high_freq_regions : bool, optional
			Whether to identify and highlight high-frequency breathing regions (default: False).
		high_freq_window : int, optional
			Number of peaks to use for the moving average of inter-peak distances (default: 5).
		high_freq_threshold_percentile : float, optional
			Percentile value to determine high-frequency regions (default: 30 means areas with the lowest 30% of inter-peak intervals).
		"""

		# Filter data within the desired time range
		time_end = start_time + duration
		mask = (self.df["Timestamps"] >= start_time) & (self.df["Timestamps"] <= time_end)
		plot_data = self.df[mask]

		if plot_data.empty:
			print("No data available in the specified time range.")
			return

		# Extract time and respiration values
		time = plot_data["Timestamps"].values
		resp_values = plot_data["Respiration Value"].values

		# if peak_sel or peak_thresh are not provided, set defaults based on the data
		if peak_sel is None:
			peak_sel = (np.max(resp_values) - np.min(resp_values)) / 4
		if peak_thresh is None:
			peak_thresh = np.percentile(resp_values, 90) * 0.2  # Or use mean + std


		# Create the plot
		plt.figure(figsize=figsize)
		plt.plot(time, resp_values, 'b-', linewidth=1, label="Respiration Signal")

		# Detect and plot peaks if enabled
		peak_times = []
		if show_peaks:
			peak_inds, peak_mags = peakfinder(resp_values, sel=peak_sel, thresh=peak_thresh)
			peak_times = time[peak_inds.astype(int)]
			plt.scatter(peak_times, peak_mags, color='red', label="Detected Peaks", zorder=3)

		# Identify and highlight high-frequency regions if enabled
		if show_high_freq_regions and len(peak_times) > 1:
			inter_peak_intervals = np.diff(peak_times)

			# Smooth the intervals using a moving average
			if len(inter_peak_intervals) >= high_freq_window:
				smoothed_ipi = np.convolve(inter_peak_intervals, np.ones(high_freq_window) / high_freq_window, mode="valid")
			else:
				smoothed_ipi = inter_peak_intervals  # No smoothing if not enough peaks

			# Determine threshold based on percentile
			threshold = np.percentile(smoothed_ipi, high_freq_threshold_percentile)

			# Find regions where inter-peak interval is below the threshold
			high_freq_regions = []
			start_region = None
			for i, interval in enumerate(smoothed_ipi):
				if interval < threshold:
					if start_region is None:
						start_region = peak_times[i]  # Start of high-frequency region
				else:
					if start_region is not None:
						high_freq_regions.append((start_region, peak_times[i]))  # End of region
						start_region = None

			if start_region is not None:
				high_freq_regions.append((start_region, peak_times[-1]))  # Last region

			# Plot high-frequency regions
			first_label = True
			for start, end in high_freq_regions:
				mask = (time >= start) & (time <= end)
				plt.fill_between(time[mask], resp_values.min(), resp_values.max(), color='red', alpha=0.3, 
								label="High-Frequency Region" if first_label else None)
				first_label = False  # Only add label once to avoid duplicate legends

		# Add behavior event overlays if behavioral_data is True
		if behavior_data and "Behavior" in self.df.columns:
			behavior_colors = {
				"Facial": "blue",
				"Anogenital": "green",
				"Flank": "red",
				"sniffing object": "purple",
			}
			for behavior in plot_data["Behavior"].dropna().unique():
				if behavior in behavior_colors:
					behavior_mask = plot_data["Behavior"] == behavior
					plt.fill_between(time[behavior_mask], resp_values.min(), resp_values.max(), 
									color=behavior_colors[behavior], alpha=0.3, label=behavior)

		# Add labels and dynamic title
		plt.xlabel('Time (seconds)')
		plt.ylabel('Amplitude')

		if start_time == 0:
			title_suffix = f"First {duration} Seconds"
		else:
			title_suffix = f"{start_time}-{start_time + duration} Seconds"

		plt.title(f"{title} - {title_suffix}")


		# Add grid and legend
		plt.grid(False, alpha=0.3)
		plt.legend()

		# Adjust layout to prevent label cutoff
		plt.tight_layout()
		plt.show()

	def save_plot(self, filename):
		"""Save the last plot to a file."""
		plt.savefig(filename)