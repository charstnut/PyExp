# %%
# Necessary import
import pyexp
from . import utils
import os

# Directory setup
IMG_DIR = os.path.join(pyexp.VCS_DIR, 'images')
utils.initialize_dir(IMG_DIR)
utils.set_figure_style()

# Regular import
import numpy as np
from scipy import signal
from numpy import fft
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# Simulate the downmix process: note there is no way to simulate noise before the mixing since
# the autocorrelation function is not approximately a delta function, and thus v(t) samples would be correlated
# We can only simulate the downmixing of signals


def check_aliasing(fs: float, decimation: int, LO: float):
    # Checks whether there will be aliasing for the current acq_rate (in MHz)
    # Raises exception if check fails, with safety bandwidth separation of 2 * acq_rate
    fs_sim = fs * decimation
    n = (2 * LO) // fs_sim
    if abs(fs_sim * n - 2 * LO) < 2 * fs or abs(fs_sim *
                                                (n + 1) - 2 * LO) < 2 * fs:
        raise ValueError(
            "Acquisition rate or simulation rate can cause aliasing.")


f_LO = 1e3
fs = 64e3  # actual sampling rate after the low-pass filter
dec = 1
fs_sim = fs * dec  # simulation sampling rate for discarding the high frequency components
check_aliasing(fs, dec, f_LO)

dt = 1 / fs_sim
t = np.arange(0, 1, dt)  # 1 second
s = np.cos((f_LO + 239.3) * 2 * np.pi * t)  # Signal
# s2 = np.exp(-t / 0.25) * np.sin(f_LO * 2 * np.pi * t)


def mix_real(input_time: np.ndarray, input_signal: np.ndarray, LO_freq: float):
    # Outputs the mixed signal (VI, VQ), with real input signal (1d array)
    # v(t) = Re((VI + iVQ) exp(i Omega t))
    # Note the factor of 2 is required for this method (c.f. the hilbert method)
    V_I = 2 * input_signal * np.cos(2 * np.pi * LO_freq * input_time)
    V_Q = -2 * input_signal * np.sin(2 * np.pi * LO_freq * input_time)
    return V_I + 1j * V_Q


def mix_real_hilbert(input_time: np.ndarray, input_signal: np.ndarray,
                     LO_freq: float):
    # Outputs the mixed signal (VI, VQ), with real input signal (1d array)
    # v(t) = Re((VI + iVQ) exp(i Omega t))
    # Uses the hilbert transform
    input_hilbert = signal.hilbert(input_signal)
    output = input_hilbert * np.exp(-1j * 2 * np.pi * LO_freq * input_time)
    return output


# Use the scipy decimate function for downsample and filtering

spec = lambda x, dt: fft.fftshift(fft.fft(x) * dt)
spec_freq = lambda x, dt: fft.fftshift(fft.fftfreq(len(x), dt))

mixed_s = mix_real_hilbert(t, s, f_LO)
envelope = signal.decimate(mixed_s, q=dec)
fig: Figure
fig, ax = plt.subplots(1, 2, figsize=(20, 10))
fig.suptitle("")
env_fft = spec(envelope, 1 / fs)
ax[0].plot(spec_freq(envelope, 1 / fs), np.real(env_fft), label="Re")
ax[0].plot(spec_freq(envelope, 1 / fs), np.imag(env_fft), label="Im")
f, pxx_env = signal.periodogram(envelope,
                                fs=fs,
                                detrend=False,
                                return_onesided=False)
ax[1].plot(f, pxx_env, label='Pxx')

mixed_s = mix_real(t, s, f_LO)
envelope = signal.decimate(mixed_s, q=dec)
env_fft = spec(envelope, 1 / fs)
ax[0].plot(spec_freq(envelope, 1 / fs), np.real(env_fft), label="Re")
ax[0].plot(spec_freq(envelope, 1 / fs), np.imag(env_fft), label="Im")
f, pxx_env = signal.periodogram(envelope,
                                fs=fs,
                                detrend=False,
                                return_onesided=False)
ax[1].plot(f, pxx_env, label='Pxx')
plt.show()

# mixed_s = mix_real_hilbert(t, s2, f_LO)
# envelope = signal.decimate(mixed_s, q=dec)
# fig: Figure
# fig, ax = plt.subplots(1, 2, figsize=(20, 10))
# fig.suptitle("")
# env_fft = spec(envelope, 1 / fs)
# ax[0].plot(spec_freq(envelope, 1 / fs), np.real(env_fft), label="Re")
# ax[0].plot(spec_freq(envelope, 1 / fs), np.imag(env_fft), label="Im")
# f, pxx_env = signal.periodogram(envelope, fs=fs, return_onesided=False)
# ax[1].plot(f, pxx_env, label='Pxx')
# plt.show()
