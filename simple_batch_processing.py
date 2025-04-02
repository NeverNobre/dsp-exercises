import os
import numpy as np
import matplotlib.pyplot as plt
import sys
import wfdb

def find_files_with_extension(directory, extension):
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(extension):
                file_paths.append(os.path.join(root, file))
    return file_paths

def process_file(filename, segment_size = 100):

    # Load the record
    record = wfdb.rdrecord(filename.replace(extension, ''))
    annotation = wfdb.rdann(filename.replace(extension, ''), 'atr')  # Load annotations
    
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

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <directory> <extension>")
        sys.exit(1)
    
    directory = sys.argv[1]
    extension = sys.argv[2]
    
    file_paths = find_files_with_extension(directory, extension)

    if not file_paths:
        print(f"No files with extension '{extension}' found in directory '{directory}'.")
    else:
        for file_path in file_paths:
            process_file(file_path)
