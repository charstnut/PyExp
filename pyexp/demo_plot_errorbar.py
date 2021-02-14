from pyexp import VCS_DIR
import pyexp.utils as utils
import os
from typing import Tuple, Union, List

IMG_DIR = os.path.join(VCS_DIR, 'images')
utils.initialize_dir(IMG_DIR)
utils.set_figure_style()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

x: np.ndarray = np.arange(1, 10)
y: np.ndarray = np.copy(x)


def plot_errorbar(x, y) -> Tuple[Figure, Union[Axes, List[Axes]]]:
    # Plotting errorbar

    fig, ax = plt.subplots()
    fig.suptitle("")
    title = "Name of The Axes"
    ax.set_title(title)
    error_kw = dict(marker='.', ls="None", capsize=5, capthick=2)
    ax.errorbar(x,
                y,
                yerr=np.random.uniform(size=x.size),
                label='Data',
                **error_kw)
    ax.set_xlabel(r"X Data $[A.U.]$")
    ax.set_ylabel(r"Y Data $[A.U.]$")
    ax.legend(loc="best")
    return (fig, ax)


fig, ax = plot_errorbar(x, y)
plt.show()
utils.save_fig(fig, IMG_DIR)
