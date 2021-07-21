# %%
# Necessary import
import pyexp
from . import utils
import os

# Directory setup
IMG_DIR = os.path.join(pyexp.VCS_DIR, 'images')
OUTPUT_DIR = os.path.join(pyexp.VCS_DIR, 'outputs')
utils.initialize_dir(IMG_DIR)
utils.initialize_dir(OUTPUT_DIR)
utils.set_figure_style()

# Regular import
import numpy as np
import pandas as pd
from scipy import signal
from numpy import fft
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# Generate a waveform of relative amplitude between -1 and 1, with number of
# points constrained to the AWG specifications

# Assuming the waveform is a bunch of cosine waves with frequencies [f1, f2, ...]
# phases [phi1, phi2, ...]
# and amplitudes [amp1, amp2, ...]
freqs = np.linspace(90e5, 200e5, 5)
phases = np.linspace(0, np.pi, 5)
amplitudes = np.linspace(1e-3, 4.2e-3, 5)

# Specifications of the AWG
POINT_NUM: int = 8000  # By default the number of points in the waveform is 8000
MIN_FREQ = 100e-6  # 100 uHz
if POINT_NUM > 8 and POINT_NUM <= 8192:
    MAX_FREQ = 5e6  # 5MHz
elif POINT_NUM > 8192 and POINT_NUM <= 12287:
    MAX_FREQ = 2.5e6
elif POINT_NUM > 12287 and POINT_NUM <= 16000:
    MAX_FREQ = 200e3
else:
    raise ValueError("Number of points is not allowed for the current AWG.")

f_max = freqs.max()
fs_min = 2 * f_max  # sampling rate is at least twice is maximum frequency component
fs = 10 * f_max  # recommended sampling rate
if fs < fs_min:
    raise ValueError(
        "Insufficient sampling rate for current signal bandwidth.")
Dt = POINT_NUM / fs  # dt * N = 1/fs * N is the segment time length
f_sig = 1 / Dt  # f_sig is the repetition frequency
if f_sig > MAX_FREQ:
    raise ValueError(
        "Repetition frequency exceeds limit, consider lowering signal bandwidth."
    )

t = np.linspace(0, Dt - 1 / fs, POINT_NUM)
sig = np.zeros_like(t)
for (f, p, a) in zip(freqs, phases, amplitudes):
    sig += a * np.sin(2 * np.pi * f * t + p)
sig /= sig.max()

window = signal.windows.hann(POINT_NUM)
sig *= window

# fig: Figure
# fig, ax = plt.subplots(figsize=(10, 10))
# fig.suptitle("")
# ax.plot(np.fft.rfftfreq(len(t), 1 / fs),
#         np.abs(np.fft.rfft(sig)),
#         label="Sig.")
# ax.legend()
# plt.show()

file = os.path.join(OUTPUT_DIR, "out.csv")
df = pd.DataFrame()
df["Data"] = sig
df["Name"] = "VOLATILE"
df["Freq (Hz)"] = round(f_sig)
df.to_csv(file, index=False)
