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
from matplotlib.axes import Axes
from lmfit.minimizer import MinimizerResult
from lmfit import Parameters, Minimizer, fit_report
from numpy.testing._private.utils import assert_almost_equal

### Plots the fit results


def plot_errorbar(ax: Axes, x: np.ndarray, y: np.ndarray,
                  y_err: np.ndarray) -> Axes:
    plt_kw = dict(marker='.', ls="None", capsize=5, capthick=2)
    ax.errorbar(x, y, yerr=y_err, label='Data', **plt_kw)
    return ax


def plot_fit(ax: Axes,
             x: np.ndarray,
             params: Parameters,
             func_kw: dict = None,
             **plt_kw) -> Axes:
    sim = model_F(x, params, **func_kw)
    ax.plot(x, sim, **plt_kw)
    return ax


def plot_residual(ax: Axes, x: np.ndarray, result: MinimizerResult) -> Axes:
    plt_kw = dict(ls="None", marker="o")
    ax.plot(x, result.residual[0:-1:2], **plt_kw)
    ax.plot(x[:-1], result.residual[1:-1:2], **plt_kw)
    return ax


#%% Define complex model:


def model_F(t: np.ndarray, params: Parameters, n: int) -> np.ndarray:
    # n is the number of sinusoidal components in the signal:
    # F = sum(A_{-i} -> A_{i}) if n is odd, where i = (n - 1) / 2
    # and F = sum(A_{-i + 1} -> A_{i}) if n is even, where i = n / 2
    # The output is a complex array
    f = np.zeros_like(t) + 0j
    i_min = (-n) // 2 + 1
    for i in range(i_min, n // 2 + 1):
        A_re = params["A_re" + str(i + abs(i_min))]
        A_im = params["A_im" + str(i + abs(i_min))]
        f_c = params["f_c"]  # this is original omega_c - omega_LO
        f_a = params["f_a"]
        fn = 0.5 * (A_re + 1j * A_im) * np.exp(1j * 2 * np.pi *
                                               (i * f_a + f_c) * t)
        f += fn
    return f


np.random.seed(42)
params_truth = Parameters()
A_names = []
n = 3
# add parameters in batch
i_min = (-n) // 2 + 1
for i in range(i_min, n // 2 + 1):
    A_str = "A_re" + str(i + abs(i_min))
    A_str2 = "A_im" + str(i + abs(i_min))
    A_names += [A_str, A_str2]
    params_truth.add(A_str, value=np.random.rand() * 2 - 1)
    params_truth.add(A_str2, value=np.random.rand() * 2 - 1)
for k, v in params_truth.valuesdict().items():
    print(str(k) + ": " + str(v))

T = 1e-4
dt = 1e-8
t = np.arange(0, T, dt)
params_truth.add("f_c", 5e6 + 1202)  # 5e6 Hz
params_truth.add("f_a", 6e6 - 2930)
params_truth.pretty_print()
F = model_F(t, params_truth, n)

# %%
sigma = 0.1
np.random.seed(43)
F_test = F + np.random.normal(
    0, sigma, size=t.shape) + 1j * np.random.normal(0, sigma, size=t.shape)

F_test_f, F_test_pxx = signal.periodogram(F_test,
                                          fs=1 / dt,
                                          return_onesided=False)
F_test_f = fft.fftshift(F_test_f)
F_test_pxx = fft.fftshift(F_test_pxx)

height_min = 100 * (2 *
                    sigma**2) * dt  # 100 times the expected mean noise value
F_peaks, F_peaks_prop = signal.find_peaks(F_test_pxx,
                                          height=(height_min, None),
                                          threshold=(height_min, None))

fig: Figure
fig, ax = plt.subplots(1, 2, figsize=(20, 10))
fig.suptitle("")
ax[0].set_title("Test")
ax[1].set_title("PSD")
plt_kw = dict(ls="-")
ax[0].plot(t, F_test.real, label="Re", **plt_kw)
ax[0].plot(t, F_test.imag, label="Im", **plt_kw)
ax[1].plot(F_test_f, F_test_pxx, label="Pxx", **plt_kw)
ax[1].scatter(F_test_f[F_peaks], F_test_pxx[F_peaks], marker="x", c="C1")
ax[1].hlines(height_min,
             F_test_f.min(),
             F_test_f.max(),
             ls="--",
             colors="gray")
plt.show()

np.random.seed(22)
params_test = Parameters()
n = len(F_peaks)
i_min = (-n) // 2 + 1
for i in range(i_min, n // 2 + 1):
    A_str = "A_re" + str(i + abs(i_min))
    A_str2 = "A_im" + str(i + abs(i_min))
    params_test.add(A_str, value=np.random.rand() * 2 - 1)
    params_test.add(A_str2, value=np.random.rand() * 2 - 1)
peak_freq = F_test_f[F_peaks]
f_c_init = F_test_f[F_peaks[(n - 1) // 2]]
f_a_init = np.average(np.diff(peak_freq))
params_test.add("f_c",
                f_c_init,
                min=f_c_init - 10 / T,
                max=f_c_init + 10 / T,
                vary=True)
params_test.add("f_a",
                f_a_init,
                min=f_a_init - 10 / T,
                max=f_a_init + 10 / T,
                vary=True)
if np.isnan(f_a_init):
    f_a_init = 0
    params_test["omega_a"].set(value=f_a_init, vary=False)
params_test.pretty_print()


def residual(pars: Parameters,
             x: np.ndarray,
             y: np.ndarray,
             y_err: float = None,
             func_kw: dict = None) -> np.ndarray:
    # Calculates the residual, note for complex function with uncorrelated
    # z = x + iy and with the same sigma the error (sigma) is a real float

    if y_err is None:
        y_err = 1.0
    resid = (y - model_F(x, pars, **func_kw)) / y_err
    return resid.view(float)


# scale_covar multiplies covar by redchi (necessary)
# covar is calculated as inv(Hess) * 2 and transformed to external (for bound problems)
mini = Minimizer(residual,
                 params_test,
                 fcn_args=(t, F_test, sigma, dict(n=n)),
                 scale_covar=True)
# if params not provided then use the Minimizer params
mini_result = mini.minimize(method="lbfgsb")
if not mini_result.success:
    raise RuntimeError("Minimization fails.")
print("-" * 50)
print(fit_report(mini_result))

# %%

# Plots residual
# Adds 2 inches for residual
fig2: Figure
fig2, ax = plt.subplots(figsize=(8, 8))
fig2.suptitle("")
plt_kw = dict(ls="-")
ax.plot(t, F_test.real, label="Re", **plt_kw)
ax.plot(t, F_test.imag, label="Im", **plt_kw)
fig2.set_figheight(fig2.get_figheight() + 2)
# Creates a new gridspec
gs = fig2.add_gridspec(5, 1)
# Sets current axis to the gridspec and resize the ratio
ax.set_position(gs[0:4].get_position(fig2))
ax.set_subplotspec(gs[0:4])
ax_resid = fig2.add_subplot(gs[4], sharex=ax)
ax_resid.set_subplotspec(gs[4])
ax_resid = plot_residual(ax_resid, t, mini_result)

fig2.tight_layout()

# Plots fitting result
fig2.set_figwidth(fig2.get_figwidth() + 4)
gs = fig2.add_gridspec(5, 2, width_ratios=[2, 1])
ax.set_position(gs[0:4, 0].get_position(fig2))
ax.set_subplotspec(gs[0:4, 0])
ax_resid.set_position(gs[4:, 0].get_position(fig2))
ax_resid.set_subplotspec(gs[4:, 0])
ax_fitres = fig2.add_subplot(gs[:, 1])
ax_fitres.set_subplotspec(gs[:, 1])
ax_fitres.text(0.01,
               0.5,
               fit_report(mini_result, show_correl=False),
               bbox={
                   'facecolor': 'white',
                   'alpha': 0.5,
               },
               transform=ax_fitres.transAxes)
ax_fitres.grid(False)
ax_fitres.set_axis_off()
ax_fitres.set_frame_on(False)

fig2.tight_layout()

ax.set_xlabel(r"X Data $[A.U.]$")
ax.set_ylabel(r"Y Data $[A.U.]$")
ax.legend(loc="best")
plt.show()
# utils.save_fig(fig, IMG_DIR, title)
