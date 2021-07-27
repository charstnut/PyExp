### Utility functions
# For type checking
from __future__ import annotations
from pathlib import Path
import sys
from typing import Tuple, Union, List

from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.axes import Axes

import numpy as np


def set_figure_style() -> None:
    """Sets the default figure style.
    """

    plt.style.use(['seaborn-talk'])
    FONT = {'family': 'STIXGeneral', 'size': 13}
    mpl.rc('font', **FONT)  # pass in the font dict as kwargs
    mpl.rc(('xtick', 'ytick'), labelsize=16, direction='in')
    mpl.rcParams['figure.autolayout'] = True
    mpl.rcParams['axes.grid'] = True
    mpl.rcParams['axes.titlesize'] = 16
    mpl.rcParams['axes.titleweight'] = 'bold'
    mpl.rcParams['axes.labelsize'] = 14
    mpl.rcParams['axes.labelweight'] = 'bold'
    mpl.rcParams['lines.linewidth'] = 2
    mpl.rcParams['legend.fontsize'] = 13
    mpl.rcParams["figure.figsize"] = [8.0, 8.0]

    print("Figure style set.")


def save_fig(fig: Figure, image_dir: str, title: str = None) -> None:
    # Saves the figure to data/images with title and closes the figure.

    # Default title to axes title if there's only one axes
    if title is None:
        ax_list = fig.get_axes()
        if len(ax_list) == 1:
            title = ax_list[0].get_title()
        else:
            raise ValueError("title cannot be None")
    out_file = os.path.join(image_dir,
                            title.replace(' ', '_').replace('.', '-') + ".pdf")
    fig.savefig(out_file, bbox_inches='tight', dpi=300)
    plt.close(fig)


def initialize_dir(path_name: str) -> None:
    # Initialize a directory

    p = Path(path_name)
    if not p.is_dir():
        print("Creating target [.../{}] directory... ".format(p.parts[-1]))
        try:
            p.mkdir(parents=True)
        except:
            print("Unexpected error:", sys.exc_info()[0])
            raise
    else:
        print("Target directory already exists.")


def sample_plot() -> Tuple[Figure, Union[Axes, List[Axes]]]:
    # This is a sample script for plotting, not for direct use.
    # Copy and modify the script and define plotting function separately.

    # input x, y
    x: np.ndarray = np.arange(1, 10)
    y: np.ndarray = np.copy(x)

    fig, ax = plt.subplots()
    fig.suptitle("")
    title = "Name of The Axes"
    ax.set_title(title)
    error_kw = dict(marker='.', ls="None", capsize=10, capthick=3)
    ax.errorbar(x,
                y,
                yerr=np.random.uniform(size=x.size),
                label='Data',
                **error_kw)
    ax.set_xlabel(r"Temp. $[^{\circ}C]$")
    ax.set_ylabel(r"Signal Strength $[mV]$")
    ax.legend(loc="best")
    return (fig, ax)
