### This module contains spectrum-related operations, such as plotting PSDs and
# transforming/filtering signals

import numpy as np


def fourier(time_series: np.ndarray, dt: float, **kwargs) -> np.ndarray:
    # Returns the continuous fourier transform equivalent of the given time series.
    # This function shows how to transform from DFTs to normal FTs.
    # Assumes 1D transform along some axis

    if np.any(np.iscomplex(time_series)):
        return np.fft.fft(time_series, **kwargs) * dt
    else:
        return np.fft.rfft(time_series, **kwargs) * dt


def inv_fourier(freq_series: np.ndarray) -> np.ndarray:
    # This should take in the continous sampling in frequency space and output the original time series
    pass


def psd(time_series: np.ndarray, dt: float, total_time: float,
        **kwargs) -> np.ndarray:
    # This should be the same as scipy's periodogram routine with "density"
    # Verified against scipy.csd, psd, etc.

    fft = fourier(time_series, dt, **kwargs)
    return np.abs(fft)**2 / total_time
