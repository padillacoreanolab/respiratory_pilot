import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from data_processing.signal_processing import peakfinder

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, welch


class Rsp_Kit:

	def __init__(self, resp_data, fs, start_time, size, end_time=None, behavior_data=None):
		"""
		self.df = dataframe with respiratory signal, timestamp
		"""

		# Attributes used to help create dataframe in methods
		self.fs = fs
		self.start_time = start_time
		self.size = size
		self.resp_data = resp_data
		self.behavior_data = behavior_data

		self.df = None
		self.time = None
		self.time_stamps = None
		self.behavioral_maps = None
		self.peaks = None
		self.high_frequency_areas = None
		
		# Creates Initial Dataframe with no behavior data
		self._timestamps()
		self._peaks()
		self._frequencies()
		self._create_df()

		# If behavior data is provided, map behaviors to timestamps
		if behavior_data is not None:
			self._map_behaviors()
			self._create_df()

	# Creating Time array for Dataframe, or for plot.
	def _timestamps(self, seconds = None):
		# Creating total time array
		n_samples = len(self.resp_data)
		self.time = np.arange(n_samples) / self.fs + self.start_time

		# Creating time array using duration
		total_duration = self.size/self.fs
		self._timestamps = np.arange(0, total_duration, 1/self.fs)  # Create time axis

	# Tuple of timestamps with high frequencies (relevant plotting areas)
	def _frequencies(self):
		pass

	# Returns np array of behaviors mapped to correct timestamps
	def _map_behaviors(self, behavior_data=None):
		if behavior_data is not None:
			pass

		behavior_data = behavior_data[['Behavior', 'Start (s)', 'Stop (s)']]
		for _, row in behavior_data.iterrows():
			start_time = row["Start (s)"]
			stop_time = row["Stop (s)"]
			behavior = row["Behavior"]

			# Mask: Find respiration timestamps that fall within the behavior window
			mask = (self.df["Timestamp (s)"] >= start_time) & (self.df["Timestamp (s)"] <= stop_time)

			# Assign the behavior to matching respiration timestamps
			self.df.loc[mask, "Behavioral Event"] = behavior

			# Reset start_time
			start_time = 0


	# Use data_processing/frequency_analysis peakfinder
	# returns np array of peaks
	def _peaks(self):
		"""Detect peaks in signal."""
		pass

	# Creates the dataframe using all previous data
	def _create_df(self):
		# resp_data should always be present so we base the length of the necessary NaN padding on it
		max_len = len(self.resp_data)

		# Pad missing data with NaN
		self.behavior_data= self.behavioral_data if self.behavior_data is not None else [np.nan] * max_len
		self.peaks = self.peaks if self.peaks is not None else [np.nan] * max_len

		# Create the DataFrame
		self.df = pd.DataFrame({
			'Timestamps': self._timestamps,
			'Respiration Value': self.resp_data,  # Always present
			'Behavioral Event': self.behavioral_maps,
			'Peaks': self.peaks
		})



	# final_resp_plot code
	def plot(self):
		pass

	def save_plot(self, filename):
		"""Save the last plot to a file."""
		plt.savefig(filename)