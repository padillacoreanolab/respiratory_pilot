import matplotlib.pyplot as plt

def resp_plot(cleaned_signal, time, peaks, troughs, onset=None, end=None):
	# Create the plot
	plt.figure(figsize=(10, 6))

	# Plot cleaned signal
	plt.plot(time, cleaned_signal, color="blue", label="Cleaned Signal")

	# Plot inhalation peaks
	plt.scatter(time[peaks == 1], cleaned_signal[peaks == 1], 
				color="red", label="Inhalation Peaks")

	# Plot exhalation troughs
	plt.scatter(time[troughs == 1], cleaned_signal[troughs == 1], 
				color="orange", label="Exhalation Troughs")
	
	# Vertical line at onset
	if onset != None:
		plt.axvline(x=onset, color="green", linestyle="--", label="Time = onset")

	# Vertical line at end
	if end != None:
		plt.axvline(x=end, color="red", linestyle="--", label="Time = end")

	# Add labels, legend, and title
	plt.title(f"Respiratory Signal")
	plt.xlabel("Time (s)")
	plt.ylabel("Amplitude")
	plt.legend()
	plt.grid(True)
	plt.tight_layout()
	plt.show()