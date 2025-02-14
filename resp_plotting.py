import matplotlib.pyplot as plt
import numpy as np

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
	

def plot_respiratory_data_by_second(resp_data, fs, start_time, num_seconds=10, figsize=(15, 12)):
    """
    Plot respiratory data with each second in a separate subplot.
    
    Parameters:
    -----------
    resp_data : array-like
        The respiratory data array
    fs : float
        Sampling frequency in Hz
    start_time : float
        Start time of the recording
    num_seconds : int, optional
        Number of seconds to plot (default: 10)
    figsize : tuple, optional
        Figure size (width, height) in inches
    """
    # Create figure and subplots
    fig, axes = plt.subplots(5, 2, figsize=figsize)
    axes = axes.flatten()
    
    # Calculate samples per second
    samples_per_second = int(fs)
    
    for i in range(num_seconds):
        # Calculate start and end indices for this second
        start_idx = i * samples_per_second
        end_idx = (i + 1) * samples_per_second
        
        # Create time array for this second
        time = np.arange(samples_per_second) / fs + start_time + i
        
        # Plot data for this second
        axes[i].plot(time, resp_data[start_idx:end_idx], 'b-', linewidth=1)
        
        # Add labels and title for each subplot
        axes[i].set_xlabel('Time (seconds)')
        axes[i].set_ylabel('Amplitude')
        axes[i].set_title(f'Second {i+1}')
        
        # Add grid
        axes[i].grid(True, alpha=0.3)
        
        # Set consistent y-axis limits across all subplots
        y_min = np.min(resp_data[:num_seconds * samples_per_second])
        y_max = np.max(resp_data[:num_seconds * samples_per_second])
        y_margin = (y_max - y_min) * 0.1  # Add 10% margin
        axes[i].set_ylim(y_min - y_margin, y_max + y_margin)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
		

def resp_behavior_plot(resp_df, fs, start_time, duration=10, figsize=(12, 6)):
    """
    Plot respiratory data for a given duration with behavior event overlays.
    
    Parameters:
    -----------
    resp_df : DataFrame
        DataFrame containing 'Timestamp (s)', 'Respiration Value', and 'Behavioral Event'.
    fs : float
        Sampling frequency in Hz.
    start_time : float
        Start time of the recording.
    duration : int, optional
        Duration of time (in seconds) to display in the plot (default: 10).
    figsize : tuple, optional
        Figure size (width, height) in inches.
    """
    # Calculate the number of samples to display
    n_samples = int(duration * fs)

    # Filter data within the desired time range
    time_end = start_time + duration
    mask = (resp_df["Timestamp (s)"] >= start_time) & (resp_df["Timestamp (s)"] <= time_end)
    plot_data = resp_df[mask]

    if plot_data.empty:
        print("No data available in the specified time range.")
        return

    # Extract time and respiration values
    time = plot_data["Timestamp (s)"]
    resp_values = plot_data["Respiration Value"]

    # Define behavior colors
    behavior_colors = {
        "Facial": "blue",
        "Anogenital": "green",
        "Flank": "red"
    }

    # Create the plot
    plt.figure(figsize=figsize)
    plt.plot(time, resp_values, 'b-', linewidth=1, label="Respiration Signal")

    # Add behavior event overlays
    for behavior in plot_data["Behavioral Event"].dropna().unique():
        if behavior in behavior_colors:
            behavior_mask = plot_data["Behavioral Event"] == behavior
            plt.fill_between(time[behavior_mask], resp_values.min(), resp_values.max(), 
                             color=behavior_colors[behavior], alpha=0.3, label=behavior)

    # Add labels and title
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.title(f'Respiratory Data - First {duration} Seconds')

    # Add grid and legend
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    plt.show()
