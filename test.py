import wfdb
import numpy as np
import matplotlib.pyplot as plt


def process_file(filename, segment_size = 100):

    # Load the record
    record = wfdb.rdrecord(filename)
    annotation = wfdb.rdann(filename, 'atr')  # Load annotations
    
    signal =  record.p_signal
    fs = annotation.fs

    samples, channels = signal.shape
    samples_idxs = np.arange(samples)

    totalsegments = len(signal)//segment_size

    energies = np.zeros((totalsegments, channels))  # Initialize energies array

    for channel in range(channels):  # Loop through each channel
        for segment_index in range(totalsegments):  # Loop through each segment
            start = segment_index * segment_size
            end = start + segment_size
            segment = signal[start:end, channel]  # Extract the segment for this channel
            energies[segment_index, channel] = np.sum(segment**2)  # Calculate energy
    plot_signal_and_energy(signal, energies, segment_size)

def plot_signal_and_energy(signal, energies, segment_size):
    plt.subplot(211)
    plt.plot(signal)
    plt.subplot(212)
    plt.plot(energies)
    plt.show()

process_file("mit-bih-arrhythmia-database-1.0.0/100")
