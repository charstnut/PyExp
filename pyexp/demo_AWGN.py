import pyexp
from . import utils
from . import spec
import os

IMG_DIR = os.path.join(pyexp.VCS_DIR, 'images')
utils.initialize_dir(IMG_DIR)
utils.set_figure_style()

import numpy as np
import matplotlib.pyplot as plt

### Generate a bunch of gaussian noise
sigma = 3
y_data = np.random.normal(0, sigma, (100, 5000))
dt = 0.001  # 1 / fs, the sampling rate, max freq = fs / 2
total_time = y_data.shape[-1] * dt
y_data_psd = spec.psd(y_data, dt=dt, total_time=total_time, axis=-1)
y_data_avg = sigma**2 * dt  # the average of the psd should be sigma^2 / Fs
freq = np.fft.rfftfreq(y_data.shape[-1], d=dt)
mean_psd = np.mean(y_data_psd, axis=0)
plt.plot(freq, mean_psd)
plt.plot(freq, np.ones_like(freq) * y_data_avg, color="C1")
plt.title("RMS2: {}, total Power from FFT: {}".format(
    sum(y_data[0]**2) / y_data.shape[-1],
    sum(mean_psd) / (total_time) * 2))
plt.show()
