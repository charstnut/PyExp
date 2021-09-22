from lmfit.minimizer import MinimizerResult
from numpy.testing._private.utils import assert_almost_equal
import pyexp
from . import utils
import os
from typing import Tuple, Union, List

IMG_DIR = os.path.join(pyexp.VCS_DIR, 'images')
utils.initialize_dir(IMG_DIR)
utils.set_figure_style()

import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from lmfit import Parameter, Parameters, Minimizer, fit_report

### Prepare model and data: assuming validity of gaussian errors
# Assumes a time series of exponential decay model


def model(x: np.ndarray, pars: Parameters) -> np.ndarray:
    a1 = pars["a1"].value
    a2 = pars["a2"].value
    t1: float = pars["t1"].value
    t2: float = pars["t2"].value
    return a1 * np.exp(x / t1) + a2 * np.exp(x / t2)


# Generates truth data using ode solver
import sympy
sympy.init_printing()
y = sympy.symbols('y_0:%d' % 2)
t = sympy.symbols('t')
a, b, c, d = sympy.symbols('a b c d')
matrix = sympy.Matrix([[a, b], [c, d]])
f = matrix * sympy.Matrix(y)
f_jac = f.jacobian(y)
f_lamb = sympy.lambdify([t, y, (a, b, c, d)], f)
f_jac_lamb = sympy.lambdify([t, y, (a, b, c, d)], f_jac)


def dydt(t, y, a, b, c, d):
    return f_lamb(t, y, (a, b, c, d)).flatten()


def dydt_jac(t, y, a, b, c, d):
    return f_jac_lamb(t, y, (a, b, c, d))


params_truth = Parameters()
params_truth.add("a", 2.0)
params_truth.add("b", -3.0)
params_truth.add("c", -3.0)
params_truth.add("d", -1.0)
p = tuple([params_truth[pars].value for pars in params_truth])


def generate_data(x) -> np.ndarray:
    # Convenience function

    sol = integrate.solve_ivp(dydt, (x[0], x[-1]),
                              y0=np.array([2, 3]),
                              t_eval=x,
                              jac=dydt_jac,
                              args=p)
    y = sol.y[0, :] + np.random.normal(0, 0.01, size=sol.y[0, :].size)
    return y


params = Parameters()
params.add("a1", 0.5)
params.add("a2", 0.5)
params.add("t1", -0.5)
params.add("t2", 0.5)
x_data = np.linspace(0, 0.5, 100)  # in Hz
y_data = generate_data(x_data)
y_err = np.ones_like(y_data) * 0.01


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

### Find the exact solution for a1, a2, t1, t2
P, D = matrix.evalf(subs=dict(params_truth)).diagonalize(normalize=True)
tau_truth = []
for i in range(len(D.row(0))):
    tau_truth.append(1 / D[i, i])

id = np.argsort(tau_truth)
tau_test = [mini_result.params["t1"].value, mini_result.params["t2"].value]
assert_almost_equal(np.array(tau_truth)[id], tau_test, decimal=1)

A = P.T * np.array([[2, 3]]).T
A_truth = [(A[i] * P.col(i))[0] for i in range(len(D.row(0)))]
A_test = [mini_result.params["a1"].value, mini_result.params["a2"].value]
assert_almost_equal(np.array(A_truth)[id], A_test, decimal=1)

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
ax = plot_errorbar(ax, x_data, y_data, None)
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
