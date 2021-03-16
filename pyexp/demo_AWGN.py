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

y_data = np.random.normal(0, 2, (100, 10000))
y_data_psd = spec.psd(y_data, dt=1, total_time=y_data.shape[-1], axis=-1)
freq = np.fft.rfftfreq(y_data.shape[-1])
plt.plot(freq, np.mean(y_data_psd, axis=0))
plt.show()
