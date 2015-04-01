import numpy as np

from scipy import special, integrate, interpolate
from cora.util.cosmology import Cosmology
import matplotlib.pyplot as plt
import matplotlib

import matter_power
import angular_terms

# Redshift at which we can detect event of luminocity L*.
Z_STAR = 0.8
ALPHA = -0.7

B_E = 0.9
B_F = 1.3


def d_ln_n(chi):
    """Not vecotized"""

    delta_chi = 0.01 * chi
    n = n_chi(np.array([chi, chi + delta_chi]))

    d_ln_n_d_chi = (n[1] - n[0]) / delta_chi / n[0]
    #print n
    #print d_ln_n_d_chi

    #plt.plot(redshifts, n*chi_s**2)
    #plt.figure()
    #plt.plot(redshifts, n)

    # Finite difference derivative and normalize.
    return d_ln_n_d_chi


def n_chi(chi):
    """Vectorized."""
    # Invert the distance redshift relation by interpolating.
    c = Cosmology()
    redshifts_s = np.linspace(0.01, 3, 50)
    chi_s = c.comoving_distance(redshifts_s)
    redshift_interp = interpolate.interp1d(chi_s, redshifts_s, kind='cubic')
    redshifts = redshift_interp(chi)

    # Calculate the source density at both redshifts.
    dL = c.luminosity_distance(redshifts)
    dL_star = c.luminosity_distance(Z_STAR)
    # Minimum detectable L/L*
    x_min = (dL / dL_star)**2
    # Get source density by integrating the luminocity function.
    n = []
    dndL_fun = lambda x : x**ALPHA * np.exp(-x)
    for x_m in x_min:
        this_n, err = integrate.quad(dndL_fun, x_m, np.inf)
        n.append(this_n)
    n = np.array(n)
    return n


def apply_coeffs(mult_ell, chi, delta):

    coeff = d_ln_n(chi) + 2./chi
    i1, i2, i3 = mult_ell.get_integrals(chi, delta)
    t1 = (B_F - B_E)**2 * i1
    t2 = (B_F - B_E) * B_E * coeff * i2
    t3 = B_E**2 * coeff**2 * i3

    return t1, t2, t3

def plot_spectra(mult_ell, chi, delta, **kwargs):
    ells = mult_ell.ells
    t1, t2, t3 = apply_coeffs(mult_ell, chi, delta)
    plt.loglog(ells, t1 + t2 + t3, '-k', label='total', **kwargs)
    plt.loglog(ells, t1, '--b', label='local term', **kwargs)
    plt.loglog(ells, t2, '-.g', label='cross term', **kwargs)
    plt.loglog(ells, t3, ':r', label='integral term', **kwargs)


def get_ells():
    ells = range(10, 101, 10)
    factor = 1.1
    max = 1000
    while ells[-1] < max:
        ells.append(int(factor * ells[-1]))
    return np.array(ells)


def get_mult_ell(ells=None):
    if ells is None:
        ells = get_ells()
    CHI_MAX = 3600.
    mell = angular_terms.MultiEll(ells, CHI_MAX)
    return mell


def my_plots(mult_ell):
    matplotlib.rcParams.update({'font.size': 14,
                                'text.usetex' : False,
                                'figure.autolayout': True})
    CHI = 1000.
    DELTAS = [5., 20., 50., 100.]
    Y_MIN = 4e-8
    Y_MAX = 1e-5
    X_MIN = 9
    X_MAX = 1001
    
    kwargs = {
            "linewidth" : 2.,
             }

    title = "$\\bar\\chi=%d\\,\\rm{MPc}/h$, $\\Delta\\chi=%d\\,\\rm{MPc}/h$"
    
    for delta in DELTAS:
        plt.figure()
        plot_spectra(mult_ell, CHI, delta, **kwargs)
        plt.ylim([Y_MIN, Y_MAX])
        plt.xlim([X_MIN, X_MAX])
        plt.title(title % (CHI, delta))
        plt.xlabel(r"$\ell$",
                   fontsize=18,
                   )
        plt.ylabel(r"$C^{ss}_\ell(\chi,\chi')$",
                   fontsize=18,
                   )
        plt.legend(loc="lower left", labelspacing=.1, frameon=False)

    chi_s = np.linspace(100, 3500, 50)
    coef_s = np.array([  d_ln_n(chi) + 2./chi for chi in chi_s ])

    plt.figure()
    plt.plot(chi_s, coef_s, **kwargs)
    plt.xlabel(r"$\chi$ ($\rm{MPc}/h$)",
               fontsize=18,
               )
    plt.ylabel(r"$(\frac{1}{\bar{n}_f}\frac{d \bar{n}_f}{d \chi}"
               r"+ \frac{2}{\chi})$  ($h/\rm{MPc})$",
               fontsize=18,
               )

    plt.figure()
    plt.plot(chi_s, chi_s**2 * n_chi(chi_s), **kwargs)
    plt.xlabel(r"$\chi$ ($\rm{MPc}/h$)",
               fontsize=18,
               )
    plt.ylabel(r"$\chi^2\bar{n}_f$  ($h/\rm{MPc})$",
               fontsize=18,
               )
    




