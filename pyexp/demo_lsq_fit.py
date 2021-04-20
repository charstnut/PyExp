from lmfit.minimizer import MinimizerResult
import pyexp
from . import utils
import os
from copy import deepcopy
# from typing import Tuple, Union, List

IMG_DIR = os.path.join(pyexp.VCS_DIR, 'images')
utils.initialize_dir(IMG_DIR)
utils.set_figure_style()

import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from lmfit import Parameter, Parameters, Minimizer, fit_report

### Prepare model and data: assuming validity of gaussian errors
# Data mimicking measurement and uncertainties [6.1(d)]


def model(x: np.ndarray, pars: Parameters) -> np.ndarray:
    m: float = pars["slope"].value
    b: float = pars["intercept"].value
    return m * x + b


def generate_data(x) -> np.ndarray:
    # Convenience function

    params_truth = Parameters()
    params_truth.add("slope", value=2.0)
    params_truth.add("intercept", value=0.4)
    x_err = np.random.normal(loc=0, scale=1, size=x.size)
    return model(x, params_truth) + x_err


params = Parameters()
params.add("slope", 0.5)
params.add("intercept", 10)
x_data = np.linspace(15, 100, 50)  # in Hz
y_data = generate_data(x_data)
y_err = np.ones_like(y_data) * 1.0


def residual(pars: Parameters,
             x: np.ndarray,
             y: np.ndarray,
             y_err: np.ndarray = None) -> np.ndarray:
    # Calculates the residual

    if y_err is None:
        y_err = np.ones_like(y)
    resid = (y - model(x, pars)) / y_err
    return resid


# scale_covar multiplies covar by redchi (necessary)
# covar is calculated as inv(Hess) * 2 and transformed to external (for bound problems)
mini = Minimizer(residual,
                 params,
                 fcn_args=(x_data, y_data, y_err),
                 scale_covar=True)
# if params not provided then use the Minimizer params
mini_result = mini.least_squares()
if not mini_result.success:
    raise RuntimeError("Minimization fails.")
print("-" * 50)
print(fit_report(mini_result))

mini_result2 = mini.minimize(method="lbfgsb", params=params)
if not mini_result2.success:
    raise RuntimeError("Minimization 2 fails.")
print("-" * 50)
print(fit_report(mini_result2))

### Estimate the error using analytical formulas:

d = len(x_data) * (x_data**2).sum() - (x_data.sum())**2
intercept = ((x_data**2).sum() * y_data.sum() - x_data.sum() *
             (x_data * y_data).sum()) / d
slope = (len(x_data) *
         (x_data * y_data).sum() - x_data.sum() * y_data.sum()) / d

intercept_err = np.sqrt(mini_result.redchi) * np.sqrt((x_data**2).sum() / d)
slope_err = np.sqrt(mini_result.redchi) * np.sqrt(len(x_data) / d)

print("Analytical slope: {} +/- {}".format(slope, slope_err))
print("Analytical intercept: {} +/- {}".format(intercept, intercept_err))

### Estimating the error according to measurement and uncertainties (book)
# This is the same as error estimation as in covariance matrix if we turn scale_covar off
# If the redchi2 stat is too large or small, we should turn scale_covar to off

# best_params_cp = deepcopy(mini_result.params)

# def delta_chi2(delta_h, var_name, current_params, origin_result, delta=1):
#     # Finds the change in chi2 for a specific parameter, given some minimization result
#     tmp_params = deepcopy(current_params)
#     tmp_params[var_name].set(tmp_params[var_name].value + delta_h)
#     resid = residual(tmp_params, x_data, y_data, y_err)
#     delta_chi2 = (resid * resid).sum() - origin_result.chisqr
#     return delta_chi2 - delta

# for i in range(10):
#     sol = optimize.root_scalar(delta_chi2,
#                                args=("intercept", best_params_cp, mini_result),
#                                x0=mini_result.params["intercept"].stderr,
#                                x1=mini_result.params["intercept"].stderr / 2)

#     print(sol.root)
#     best_params_cp["intercept"].set(best_params_cp["intercept"].value +
#                                     sol.root)
#     best_params_cp.pretty_print()
#     best_params_cp["intercept"].set(vary=False)
#     tmp_mini = Minimizer(residual,
#                          best_params_cp,
#                          fcn_args=(x_data, y_data, y_err))
#     tmp_mini_result = tmp_mini.least_squares()
#     # print(fit_report(tmp_mini_result))
#     best_params_cp = tmp_mini_result.params

# print(best_params_cp["intercept"].value -
#       mini_result.params["intercept"].value)

### Plots the fit results


def plot_errorbar(ax: Axes, x: np.ndarray, y: np.ndarray,
                  y_err: np.ndarray) -> Axes:
    plt_kw = dict(marker='.', ls="None", capsize=5, capthick=2)
    ax.errorbar(x, y, yerr=y_err, label='Data', **plt_kw)
    return ax


def plot_fit(ax: Axes, x: np.ndarray, result: MinimizerResult) -> Axes:
    sim = model(x, result.params)
    plt_kw = dict(ls="--")
    ax.plot(x, sim, label="Lsq. Fit", **plt_kw)
    return ax


def plot_residual(ax: Axes, x: np.ndarray, result: MinimizerResult) -> Axes:
    plt_kw = dict(ls="None", marker="o")
    ax.plot(x, result.residual, **plt_kw)
    return ax


fig: Figure
fig, ax = plt.subplots()
fig.suptitle("")
title = "Name of The Axes"
ax.set_title(title)
ax = plot_errorbar(ax, x_data, y_data, y_err)
ax = plot_fit(ax, x_data, mini_result)

# Plots residual
# Adds 2 inches for residual
fig.set_figheight(fig.get_figheight() + 2)
# Creates a new gridspec
gs = fig.add_gridspec(5, 1)
# Sets current axis to the gridspec and resize the ratio
ax.set_position(gs[0:4].get_position(fig))
ax.set_subplotspec(gs[0:4])
ax_resid = fig.add_subplot(gs[4], sharex=ax)
ax_resid.set_subplotspec(gs[4])
ax_resid = plot_residual(ax_resid, x_data, mini_result)

fig.tight_layout()

# Plots fitting result
fig.set_figwidth(fig.get_figwidth() + 4)
gs = fig.add_gridspec(5, 2, width_ratios=[2, 1])
ax.set_position(gs[0:4, 0].get_position(fig))
ax.set_subplotspec(gs[0:4, 0])
ax_resid.set_position(gs[4:, 0].get_position(fig))
ax_resid.set_subplotspec(gs[4:, 0])
ax_fitres = fig.add_subplot(gs[:, 1])
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

fig.tight_layout()

ax.set_xlabel(r"X Data $[A.U.]$")
ax.set_ylabel(r"Y Data $[A.U.]$")
ax.legend(loc="best")
plt.show()
utils.save_fig(fig, IMG_DIR, title)
