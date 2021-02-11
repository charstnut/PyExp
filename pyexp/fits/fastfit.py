import warnings
import numpy as np
import scipy as sp
import uncertainties as unc
import uncertainties.unumpy as unp
from scipy import optimize, stats, integrate

# FIXME: Change this into a class-based method


def fit(model,
        pars_list,
        x_data,
        y_data,
        y_err=None,
        pars_init=None,
        x_fit_samples=1000,
        fit_kws=None):
    # Simple fit utility, prints output and returns popt
    # as uncertainties array along with simulation and normalized residuals

    if y_err is None:
        y_err = np.ones(y_data)
    if fit_kws is None:
        fit_kws = dict(absolute_sigma=True)

    # Perform fitting
    if pars_init is not None:
        par0 = pars_init  # initial guesses
    else:
        par0 = np.ones(len(pars_list))
    popt, pcov = sp.optimize.curve_fit(model,
                                       x_data,
                                       y_data,
                                       p0=par0,
                                       sigma=y_err,
                                       **fit_kws)
    # print('parameter covariance matrix: \n', pcov)
    perr = np.sqrt(np.diag(pcov))
    pars_opt = unp.uarray(popt, perr)

    res_str = ""
    for i, p in enumerate(pars_list):
        res_str += ('{} = {:.1u} \n'.format(p, pars_opt[i]))

    x_sim = np.linspace(x_data.min(), x_data.max(), x_fit_samples)
    y_fit = model(x_data, *popt)
    y_sim = model(x_sim, *popt)

    resid = (y_data - y_fit) / y_err

    return (pars_opt, x_sim, y_sim, resid, res_str)


def chi_sq(x_data, y_data, y_err, model, popt):
    # Calculate chi_sq according to fit results
    difference = y_data - model(x_data, *popt)

    chi_2 = np.sum((difference / y_err)**2)
    dof = len(difference) - len(popt)

    p_val = 1 - sp.stats.chi2.cdf(chi_2, df=dof)

    return (chi_2, dof, p_val)


def chi_sq_hypotest(bins, counts, model, popt):
    # Perform a chi-squared test on binned distributions, and returns
    # by default.
    # Note: model should output counts (norm * total counts) so that it can be directly integrated. Has to be continuous!
    # Note: counts should be greater than 5!
    if np.any(counts < 5):
        warnings.warn("Some counts are smaller than 5, consider rebinning!")
    assert len(bins) == len(counts) - 1

    expected_cnts = []
    f = lambda x: model(x, *popt)

    expected_cnts.append(sp.integrate.quad(f, -np.inf, bins[0]))
    for i in range(bins):
        expected_cnts.append(sp.integrate.quad(f, bins[i], bins[i + 1]))
    expected_cnts.append(sp.integrate.quad(f, bins[-1], np.inf))
    expected_cnts = np.asarray_chkfinite(expected_cnts)

    dof = len(counts) - len(popt)

    return sp.stats.chisquare(counts, expected_cnts, ddof=dof)
