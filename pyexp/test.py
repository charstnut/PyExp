from pyexp import VCS_DIR
import pyexp.utils as utils
import os
import sys

IMG_DIR = os.path.join(VCS_DIR, 'images')
if not os.path.isdir(IMG_DIR):
    print("Creating images directory")
    try:
        os.mkdir(IMG_DIR)
    except:
        print("Unexpected error:", sys.exc_info()[0])
        raise

utils.set_figure_style()

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(1, 10)
y = np.copy(x)

f = plt.figure()
plt.plot(x, y)
plt.show()

title = "Sample Figure"
utils.save_fig(f, title, IMG_DIR)
