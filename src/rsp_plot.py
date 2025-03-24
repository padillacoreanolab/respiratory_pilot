import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from data_processing.signal_processing import peakfinder

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, welch


class rsp_kit:

	def __init__(self, resp_data, fs, start_time, end_time, behavior_data=None):
		"""
		self.df = dataframe with respiratory signal, timestamp
		"""

		# Attributes used to help create dataframe in methods
		self.fs = fs
		self.start_time = start_time
		self.resp_data = resp_data
		self.behavior_data = behavior_data

		self.df = None
		self.time = None
		self.time_stamps = None
		self.behavioral_maps = None
		self.peaks = None
		self.high_frequency_areas = None


		# Automatically calls methods at obj initialization
		self._timestamps()
		self._frequencies()
		self._map_behaviors()
		self._peaks()
		self._create_df()

	# Creating Time array for Dataframe, or for plot.
	def _timestamps(self, seconds = None):
		n_samples = len(self.resp_data)
		self.time = np.arrange(n_samples) / self.fs + self.start_time

	# Tuple of timestamps with high frequencies (relevant plotting areas)
	def _frequencies(self):
		pass
	
	# Returns np array of behaviors mapped to correct timestamps
	def _map_behaviors(self):
		pass
	
	# Use data_processing/frequency_analysis peakfinder
	# returns np array of peaks
	def _peaks(self,):
		"""Detect peaks in signal."""
		pass
	
	# Creates the dataframe using all previous data
	def _create_df(self):
		self.df = pd.DataFrame({
			'Timestamps': self.time_stamps,
			'Respiration Value': self.resp_data,
			'Behavioral Event': self.behavioral_maps,
			'Peaks': self.peaks
		})

	# final_resp_plot code
	def plot(self):
		pass

	def save_plot(self, filename):
		"""Save the last plot to a file."""
		plt.savefig(filename)