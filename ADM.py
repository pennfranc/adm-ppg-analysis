import numpy as np

def ADM(signal, up_threshold, down_threshold, sampling_rate, refractory_period):
    """ Asynchronous Delta Modulation Function based on Master Thesis by Nik Dennler

    Parameters
    ----------
    signal
        The analogue signal from which spikes are generated.
    up_threshold
        The amount by which the current signal value must be above the current bias value to generate a spike.
    down_threshold
        The amount by which the current signal value must be below the current bias value to generate a spike.
    sampling_rate
        The sampling rate of `signal` in Hz.
    refractory_period
        The time period in seconds after a signal which does not permit the generation of any spikes.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Two arrays containing the spike trains of up and down spikes respectively.
    """
    sampling_period = 1 / sampling_rate
    T = len(signal) * sampling_period
    times = np.arange(0, T, sampling_period)
    bias_value = signal[0]
    freeze = 0
    up_spikes = []
    down_spikes = []

    for idx, time, in enumerate(times):
        if freeze > 0:
            freeze -= sampling_period
        elif signal[idx] > bias_value + up_threshold:
            up_spikes.append(time)
            bias_value = signal[idx]
            freeze = refractory_period
        elif signal[idx] < bias_value - down_threshold:
            down_spikes.append(time)
            bias_value = signal[idx]
            freeze = refractory_period

    return np.array(up_spikes), np.array(down_spikes)