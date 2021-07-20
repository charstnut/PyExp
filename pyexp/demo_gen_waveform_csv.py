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
from scipy import signal
from numpy import fft
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# Generate a waveform of relative amplitude between -1 and 1, with number of
# points constrained to the AWG specifications

# Assuming the waveform is a bunch of cosine waves with frequencies [f1, f2, ...]
# phases [phi1, phi2, ...]
# and amplitudes [amp1, amp2, ...]
freqs = np.linspace(16e3, 31e3, 5)
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

t = np.arange(POINT_NUM) * 1e-6
signal = np.zeros_like(t)
for (f, p, a) in zip(freqs, phases, amplitudes):
    signal += a * np.sin(2 * np.pi * f * t + p)
signal /= signal.max()

fig: Figure
fig, ax = plt.subplots(figsize=(10, 10))
fig.suptitle("")
ax.plot(t, signal, label="Sig.")
plt.show()
