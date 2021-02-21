import pyexp.spec as spec
import numpy as np
from numpy.testing._private.utils import assert_allclose
from scipy import integrate
import pytest

import pyexp.utils as utils
utils.set_figure_style()
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

### TODO: WARNING, debugging tests seems to fail with parametrization, we need to solve this in vscode...


def box(t, t_max=1):
    # simple box window (not used)
    return np.piecewise(t, [t <= 0, (0 < t) & (t <= t_max), t > t_max],
                        [0, 1, 0])


def window2(t):
    # Gaussian window
    return np.exp(-t**2)


def numerical_fourier(f, f_data: np.ndarray, limit=(-5.0, 5.0)) -> np.ndarray:
    # Compute the numerical fourier using scipy integration
    # Default integration limit is from -5.0 to 5.0
    result = np.zeros(len(f_data), dtype=complex)
    err = np.zeros_like(f_data, dtype=complex)
    for i, ff in enumerate(f_data):
        real, real_err = integrate.quad(
            lambda t: (f(t) * np.exp(-2j * np.pi * ff * t)).real,
            limit[0],
            limit[1],
            limit=100)
        imag, imag_err = integrate.quad(
            lambda t: (f(t) * np.exp(-2j * np.pi * ff * t)).imag,
            limit[0],
            limit[1],
            limit=100)
        result[i], err[i] = real + 1j * imag, real_err * 1j * imag_err
    assert_allclose(err, 0, atol=1e-4)
    return result


# @pytest.mark.parametrize("max_time", np.linspace(1, 2.3, 1))
# def test_fourier(max_time):
# Prepare the mathematical FT of an analytical function f
# DO NOT USE sympy for integration, it is absurdly slow...

if __name__ == "__main__":
    freq = 0.95
    a = 5.0
    phase = 0
    # f1 = lambda t: a * np.sin(2 * np.pi * freq * t + phase)
    f2 = lambda t: window2(t) * a * np.cos(2 * np.pi * freq * t + phase)
    n_samples = 256
    time_series_max = 2.5
    x_data = np.linspace(0, time_series_max, n_samples)
    y_data = f2(x_data)

    # plt.plot(x_data, y_data)
    # plt.show()

    f_data = np.fft.fftfreq(n_samples, time_series_max / n_samples)
    f_data_r = np.fft.rfftfreq(n_samples, time_series_max / n_samples)

    f1_ft_truth = numerical_fourier(f2, f_data, (0.0, time_series_max))
    # f2_ft_truth = numerical_fourier(f2, f_data, (-20, 20))

    f1_ft_fft = np.fft.fft(y_data) * time_series_max / n_samples
    f1_ft_rfft = np.fft.rfft(y_data) * time_series_max / n_samples

    # scratch plotting

    fig: Figure
    fig, ax = plt.subplots()
    fig.suptitle("")
    title = "True FTs"
    ax.set_title(title)
    ax.plot(f_data, f1_ft_truth.real, label="sin FT real")
    ax.plot(f_data, f1_ft_truth.imag, label="sin FT imag")
    # ax.plot(f_data, f1_ft_fft.real, label="sin FFT real")
    # ax.plot(f_data, f1_ft_fft.imag, label="sin FFT imag")
    ax.plot(f_data_r, f1_ft_rfft.real, label="sin rFFT real")
    ax.plot(f_data_r, f1_ft_rfft.imag, label="sin rFFT imag")

    plt.legend()
    plt.show()
