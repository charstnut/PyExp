### This module contains spectrum-related operations, such as plotting PSDs and
# transforming/filtering signals

import numpy as np


def fourier(time_series: np.ndarray, total_time: float) -> np.ndarray:
    # Returns the continuous fourier transform equivalent of the given time series.
    # This function shows how to transform from DFTs to normal FTs.
    # Assumes 1d array

    n_samples = len(time_series)
    dt = total_time / n_samples
    if np.any(np.is_complex(time_series)):
        return np.fft.fft(time_series) * dt
    else:
        return np.fft.rfft(time_series) * dt


def inv_fourier(freq_series: np.ndarray) -> np.ndarray:
    # This should take in the continous sampling in frequency space and output the original time series
    pass


def psd(time_series, total_time):
    # This should be the same as scipy's periodogram routine with "density"
    # Verified against scipy.csd, psd, etc.

    fft = fourier(time_series, total_time)
    return np.abs(fft)**2 / total_time