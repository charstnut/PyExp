### This module contains spectrum-related operations, such as plotting PSDs and
# transforming/filtering signals

import numpy as np


def fourier(time_series: np.ndarray, total_time: float,
            complex: bool) -> np.ndarray:
    # Returns the continuous fourier transform of the given time series.
    # This function shows how to transform from DFTs to normal FTs.
    if complex:
        return np.fft.fft(time_series)
    else:
        return np.fft.rfft(time_series)