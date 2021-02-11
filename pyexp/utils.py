### Utility functions
# For type checking
from __future__ import annotations
import os

from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib.figure import Figure


def set_figure_style() -> None:
    """Sets the default figure style.
    """

    plt.style.use(['seaborn-talk'])
    FONT = {'family': 'STIXGeneral', 'size': 12}
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

    print("Figure style set.")


def save_fig(fig: Figure, title: str, image_dir: str) -> None:
    # Saves the figure to data/images with title and closes the figure.

    out_file = os.path.join(image_dir,
                            title.replace(' ', '_').replace('.', '') + ".pdf")
    fig.savefig(out_file, bbox_inches='tight', dpi=300)
    plt.close(fig)
