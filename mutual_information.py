import math
import numpy as np
from scipy import signal, interpolate
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt

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

def first_order_low_pass(
    data,
    cutoff_freq,
    order=1,
    fs=64
):
    nyq = 0.5 * fs
    normal_cutoff = cutoff_freq / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    y = signal.lfilter(b, a, data)
    return y

def to_spikes_and_back(
    input_signal,
    fs,
    ADM_step_factor # step threshold (up and down) as multiple of mean amplitude
):
    # run ppg through ADM
    ADM_step = ADM_step_factor * np.mean(abs(input_signal))
    up_spikes, down_spikes = ADM(
        input_signal,
        up_threshold=ADM_step,
        down_threshold=ADM_step,
        sampling_rate=fs,
        refractory_period=0
    )
    num_spikes = len(up_spikes) + len(down_spikes)


    # reconstruct original signal from ADM-generated spike train using gaussian kernel
    reconstructed_signal = (
        reconstruct_from_spikes(up_spikes, len(input_signal), 1) +
        reconstruct_from_spikes(down_spikes, len(input_signal), -1)
    )

    return reconstructed_signal, num_spikes


def create_spectrogram(input_signal, fs, nperseg, noverlap, fmin, fmax, clip_percentile=None):

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


def interpolate_hr(hr, t):
    hr_timestamps = np.arange(4, len(hr) * 2 + 4, 2)
    hr_interpolation = interpolate.interp1d(hr_timestamps, hr)
    hr_at_relevant_timestamps = hr_interpolation(t)
    return hr_at_relevant_timestamps


def compute_regression_score(X, y):
    reg = LinearRegression().fit(X, y)
    return reg.score(X, y)


def compute_cv_regression_score(X, y):
    reg = LinearRegression().fit(X, y)
    cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
    return cross_val_score(reg, X, y, cv=cv).mean()

def create_features_and_labels(
    input_signals,
    hrs,
    fs=64,
    nperseg=512,
    noverlap=384,
    fmin=0, fmax=10
):
    # create spectrogram
    f, t, Sxx = create_spectrogram(input_signals[0], fs, nperseg, noverlap, fmin, fmax)

    # interpolate relevant heart rate measurements
    hr_at_relevant_timestamps = interpolate_hr(hrs[0], t)

    return Sxx, hr_at_relevant_timestamps, f


def create_features_and_labels_activity_dependent(
    input_signals,
    hrs,
    activities_list,
    chosen_activity,
    fs=64,
    nperseg=512,
    noverlap=384,
    fmin=0, fmax=10
):
    # iterate through ppg signals and heart rates
    Sxx_list = []
    hr_at_rel_ts_list = []
    for input_signal, hr, activities in zip(input_signals, hrs, activities_list):
        
        # create all data
        f, t, Sxx = create_spectrogram(input_signal, fs, nperseg, noverlap, fmin, fmax)
        hr_at_rel_ts = interpolate_hr(hr, t)

        # for every data point, find its activity label
        activities_at_rel_ts = activities[(t * 4).astype(int)]

        # filter out all data points that do not correspond to the currently chosen activity label
        Sxx = Sxx[:, activities_at_rel_ts == chosen_activity]
        hr_at_rel_ts = hr_at_rel_ts[activities_at_rel_ts == chosen_activity]
        
        # save filtered data points
        Sxx_list.append(Sxx.copy())
        hr_at_rel_ts_list.append(hr_at_rel_ts)

    # aggregate data points from different subjects
    Sxx = np.concatenate(Sxx_list, axis=1)
    hr_at_rel_ts = np.concatenate(hr_at_rel_ts_list)

    return Sxx, hr_at_rel_ts, f
        


def score_pipeline(
    Sxx,
    hr_at_relevant_timestamps,
    num_hr_bins=10,
    num_power_bins=6,
    evaluation_method='mutual_info',
    n_neighbors=10
):

    # compute linear regression score
    if evaluation_method == 'mutual_info_sklearn':
        mutual_info = mutual_info_regression(np.transpose(Sxx), hr_at_relevant_timestamps, n_neighbors=n_neighbors)
        score = mutual_info.mean()
    elif evaluation_method == 'mutual_info':
        mutual_info = compute_mutual_information(Sxx, hr_at_relevant_timestamps, num_hr_bins, num_power_bins)
        score = mutual_info.mean()
    elif evaluation_method == 'regression_insample':
        score = compute_regression_score(np.transpose(Sxx), hr_at_relevant_timestamps)
        mutual_info = None
    elif evaluation_method == 'regression_cv':
        score = compute_cv_regression_score(np.transpose(Sxx), hr_at_relevant_timestamps)
        mutual_info = None

    return score, mutual_info


def scoring_loop(
    ppgs,
    hrs,
    activities_list=None,
    chosen_activity=1,
    step_factor_list=[],
    fmin=0.5,
    fmax=4,
    fs_ppg=64,
    nperseg=512,
    noverlap=384,
    evaluation_method='mutual_info',
    num_power_bins=6,
    num_hr_bins=10,
    n_neighbors=10,
    plot_detailed=False
):

    # switch into frequency domain
    if activities_list is None:
        Sxx_ppg, hr_at_rel_ts_ppg, f_ppg = create_features_and_labels(ppgs, hrs, nperseg=nperseg, noverlap=noverlap, fmin=fmin, fmax=fmax)
    else:
        Sxx_ppg, hr_at_rel_ts_ppg, f_ppg = create_features_and_labels_activity_dependent(ppgs, hrs, activities_list, chosen_activity, nperseg=nperseg, noverlap=noverlap, fmin=fmin, fmax=fmax)

    # compute score
    score_ppg, mutual_info_ppg = score_pipeline(Sxx_ppg, hr_at_rel_ts_ppg, evaluation_method=evaluation_method, n_neighbors=n_neighbors, num_hr_bins=num_hr_bins)
    
    # iterate through ADM threshold parameters
    score_rec_list = []
    rates_list = []
    for step_factor in step_factor_list:

        # reconstruct signals from ADM spike train
        num_spikes = 0
        rec_signals = []
        for ppg in ppgs:
            rec_signal, num_spikes_curr = to_spikes_and_back(ppg, fs_ppg, step_factor)
            rec_signals.append(rec_signal)
            num_spikes += num_spikes_curr
        
        # switch into frequency domain
        if activities_list is None:
            Sxx_rec, hr_at_rel_ts_rec, f_rec = create_features_and_labels(rec_signals, hrs, nperseg=nperseg, noverlap=noverlap, fmin=fmin, fmax=fmax)
        else:
            Sxx_rec, hr_at_rel_ts_rec, f_rec = create_features_and_labels_activity_dependent(rec_signals, hrs, activities_list, chosen_activity, nperseg=nperseg, noverlap=noverlap, fmin=fmin, fmax=fmax)

        # compute score
        score_rec, mutual_info_rec = score_pipeline(Sxx_rec, hr_at_rel_ts_rec, evaluation_method=evaluation_method, n_neighbors=n_neighbors, num_hr_bins=num_hr_bins)

        # compute average spiking rate
        rate = round(num_spikes / sum(len(ppg) for ppg in ppgs) * 64, 2)
        rates_list.append(rate)
        score_rec_list.append(score_rec)

        if plot_detailed:
            plt.plot(f_ppg, mutual_info_ppg, linewidth=3, label='PPG')
            plt.plot(f_rec, mutual_info_rec, linewidth=1.5, label='Reconstructed signal')
            plt.title('Mutual information (reconstructed signal with average rate of {} Hz)'.format(rate))
            plt.ylabel('Mutual information')
            plt.xlabel('Frequency')
            plt.legend()
            plt.show()
            
    return score_ppg, score_rec_list, rates_list

def scoring_loop_low_pass(
    ppg,
    hr,
    cutoff_freq_list=[],
    fmin=0.5,
    fmax=4,
    fs_ppg=64,
    nperseg=512,
    noverlap=384,
    evaluation_method='mutual_info',
    num_power_bins=6,
    num_hr_bins=10,
    n_neighbors=10,
    plot_detailed=False
):

    score_ppg, f_ppg, mutual_info_ppg = score_pipeline(ppg, hr, nperseg=nperseg, noverlap=noverlap, fmin=fmin, fmax=fmax, evaluation_method=evaluation_method, n_neighbors=n_neighbors, num_hr_bins=num_hr_bins)
    
    score_rec_list = []
    rates_list = []

    for cutoff_freq in cutoff_freq_list:
        rec_signal = first_order_low_pass(ppg, cutoff_freq)
        
        score_rec, f_rec, mutual_info_rec = score_pipeline(rec_signal, hr, nperseg=nperseg, noverlap=noverlap, fmin=fmin, fmax=fmax, evaluation_method=evaluation_method, n_neighbors=n_neighbors, num_hr_bins=num_hr_bins)

        score_rec_list.append(score_rec)

        if plot_detailed:
            plt.plot(f_ppg, mutual_info_ppg, linewidth=3, label='PPG')
            plt.plot(f_rec, mutual_info_rec, linewidth=1.5, label='Reconstructed signal')
            plt.title('Mutual information (reconstructed signal with average rate of {} Hz)'.format(rate))
            plt.ylabel('Mutual information')
            plt.xlabel('Frequency')
            plt.legend()
            plt.show()
            
    return score_ppg, score_rec_list


def plot_scores(score_ppg, score_rec_list, rates_list, score_name, number, title):
    plt.semilogx(rates_list, score_rec_list, label='rec signal')
    plt.axhline(score_ppg, label='original signal', color='orange')
    plt.legend()
    plt.ylabel(score_name)
    plt.xlabel('Spike rate [Hz]')
    plt.title(title + ' - Number {}'.format(number))