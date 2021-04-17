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
        if sum(power_dist) == 0: continue
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
    

#TODO: outsource signal recreation to another function and
def mutual_info_pipeline(
    dataset,
    fs=64,
    nperseg=1000,
    noverlap=None,
    fmin=0, fmax=10,
    num_hr_bins=100,
    num_power_bins=6,
    ADM_step_factor=1, # step threshold (up and down) as multiple of mean amplitude
):

    ### create ppg spectrogram
    ppg, acc, hr, activity, _= unpack_data(dataset)
    hr_timestamps = np.arange(0, len(hr) * 2, 2)
    hr_interpolation = interpolate.interp1d(hr_timestamps, hr)
    f_ppg, t_ppg, Sxx_ppg = create_spectrogram(ppg, fs, nperseg, noverlap, fmin, fmax)

    # interpolate relevant heart rate measurements
    hr_at_ppg_timestamps = hr_interpolation(t_ppg)


    ### run ppg through ADM
    ADM_step = ADM_step_factor * np.mean(abs(ppg))
    up_spikes, down_spikes = ADM(
        ppg,
        up_threshold=ADM_step,
        down_threshold=ADM_step,
        sampling_rate=fs,
        refractory_period=0
    )
    num_spikes = len(up_spikes) + len(down_spikes)


    ### reconstruct original signal from ADM-generated spike train using gaussian kernel
    reconstructed_signal = (
        reconstruct_from_spikes(up_spikes, len(ppg), 1) +
        reconstruct_from_spikes(down_spikes, len(ppg), -1)
    )


    ### create rec signal spectrogram
    f_rec, t_rec, Sxx_rec = create_spectrogram(reconstructed_signal, fs, nperseg, noverlap, fmin, fmax)

    # interpolate relevant heart rate measurements
    hr_at_rec_timestamps = hr_interpolation(t_rec)


    ### compute mutual information
    mutual_information_ppg = compute_mutual_information(Sxx_ppg, hr_at_ppg_timestamps, num_hr_bins, num_power_bins)
    mutual_information_rec = compute_mutual_information(Sxx_rec, hr_at_rec_timestamps, num_hr_bins, num_power_bins)

    return f_ppg, f_rec, mutual_information_ppg, mutual_information_rec, num_spikes
