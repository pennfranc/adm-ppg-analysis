import math
import numpy as np
from scipy import signal, interpolate
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import r2_score, mean_squared_error

from ADM import ADM
from data_loader import load_pickle, unpack_data

def compute_mutual_information(Sxx, hr, num_hr_bins, num_power_bins):
    
    result = np.zeros(Sxx.shape[0])

    # compute distribution over heart rates
    hr_dist, hr_bin_edges = np.histogram(hr, bins=num_hr_bins)
    hr_dist = hr_dist / sum(hr_dist)

    # iterate over frequencies
    for freq_idx, powers in enumerate(Sxx):
        
        # compute power distribution at given frequency
        power_dist, power_bin_edges = np.histogram(powers, bins=num_power_bins)
        if len(power_dist) == 1: continue
        power_dist = power_dist / sum(power_dist)

        # iterate over heart rate bins
        for i in range(num_hr_bins):
            
            # retrieve powers in current heart rate bin
            hr_bin_edge_left = hr_bin_edges[i]
            hr_bin_edge_right = hr_bin_edges[i + 1]
            powers_at_given_hr = powers[(hr >= hr_bin_edge_left) & (hr < hr_bin_edge_right)]
            
            # compute conditional probability of powers given heart rate bin
            cond_power_dist, _ = np.histogram(powers_at_given_hr, bins=power_bin_edges)
            if sum(cond_power_dist) == 0: continue
            cond_power_dist = cond_power_dist / sum(cond_power_dist)

            # iterate over power bins
            for j in range(num_power_bins):
                
                if cond_power_dist[j] == 0:
                    continue
                
                result[freq_idx] += hr_dist[i] * cond_power_dist[j] * np.log2(cond_power_dist[j] / power_dist[j])

    return result


def reconstruct_from_spikes(spikes, length, spike_value):
    reconstructed_signal = np.zeros(length)
    spikes_sampling_rate_indices = (spikes * 64).astype(int)
    reconstructed_signal[spikes_sampling_rate_indices] = spike_value
    reconstructed_signal = gaussian_filter1d(reconstructed_signal, 10)
    return reconstructed_signal

def to_spikes_and_back(
    input_signal,
    fs,
    ADM_step_factor # step threshold (up and down) as multiple of mean amplitude
):
    ### run ppg through ADM
    ADM_step = ADM_step_factor * np.mean(abs(input_signal))
    up_spikes, down_spikes = ADM(
        input_signal,
        up_threshold=ADM_step,
        down_threshold=ADM_step,
        sampling_rate=fs,
        refractory_period=0
    )
    num_spikes = len(up_spikes) + len(down_spikes)


    ### reconstruct original signal from ADM-generated spike train using gaussian kernel
    reconstructed_signal = (
        reconstruct_from_spikes(up_spikes, len(input_signal), 1) +
        reconstruct_from_spikes(down_spikes, len(input_signal), -1)
    )

    return reconstructed_signal, num_spikes

def create_spectrogram(input_signal, fs, nperseg, noverlap, fmin, fmax, clip_percentile=99):

    # create spectrogram
    f, t, Sxx = signal.spectrogram(input_signal, fs, nperseg=nperseg, noverlap=noverlap)

    # keep only frequencies of interest
    freq_slice = np.where((f >= fmin) & (f <= fmax))
    f = f[freq_slice]
    Sxx = Sxx[freq_slice,:][0]

    # clip
    if clip_percentile is not None:
        Sxx = np.clip(Sxx, 0, np.percentile(Sxx.flatten(), clip_percentile))

    return f, t, Sxx
    

def mutual_info_pipeline(
    input_signal,
    hr,
    fs=64,
    nperseg=1000,
    noverlap=None,
    fmin=0, fmax=10,
    num_hr_bins=100,
    num_power_bins=6,
    ADM_step_factor=1, 
):

    # create spectrogram
    f, t, Sxx = create_spectrogram(input_signal, fs, nperseg, noverlap, fmin, fmax)

    # interpolate relevant heart rate measurements
    hr_timestamps = np.arange(0, len(hr) * 2, 2)
    hr_interpolation = interpolate.interp1d(hr_timestamps, hr)
    hr_at_relevant_timestamps = hr_interpolation(t)

    # compute mutual information
    mutual_information = compute_mutual_information(Sxx, hr_at_relevant_timestamps, num_hr_bins, num_power_bins)

    return f, mutual_information
