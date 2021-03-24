import numpy as numpy

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
    np.ndarray
        Boolean value indicating whether the seasonality is significant given the parameters passed.
    """
    sampling_period = 1 / sampling_rate
    T = len(signal) * sampling_period
    times = 