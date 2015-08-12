import numpy as np

from scipy import special, integrate, interpolate
from cora.util.cosmology import Cosmology
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib

import matter_power
import angular_terms

# Redshift at which we can detect event of luminocity L*.
Z_STAR = 1.
ALPHA = -1.

B_E = 1.0
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
    
    chi1 = chi - abs(delta) / 2
    chi2 = chi + abs(delta) / 2

    coeff1 = d_ln_n(chi1) + 2./chi1
    coeff2 = d_ln_n(chi2) + 2./chi2
    i1, i2, i3 = mult_ell.get_integrals(chi, delta)
    t1 = (B_F - B_E)**2 * i1
    t2 = (B_E - B_F) * B_E * coeff2 * i2
    t3 = B_E**2 * coeff1 * coeff2 * i3

    return t1, t2, t3

def plot_spectra(mult_ell, chi, delta, local_only=False, **kwargs):
    ells = mult_ell.ells
    t1, t2, t3 = apply_coeffs(mult_ell, chi, delta)
    #trans = lambda t: ells**2 * abs(t)
    trans = lambda t: abs(t)
    if local_only:
        plt.loglog(ells, trans(t1), '--b', label='local term', **kwargs)
    else:
        plt.loglog(ells, trans(t1 + t2 + t3), '-k', label='total', **kwargs)
        plt.loglog(ells, trans(t1), '--b', label='local term', **kwargs)
        plt.loglog(ells, trans(t3), ':r', label='integral term', **kwargs)
        plt.loglog(ells, trans(t2), '-.g', label='cross term', **kwargs)


def get_ells():
    ells = range(10, 101, 10)
    factor = 1.1
    max = 1000
    while ells[-1] < max:
        ells.append(int(factor * ells[-1]))
    return np.array(ells)


def get_mult_ell(ells=None, limber=True):
    if ells is None:
        ells = get_ells()
    CHI_MAX = 3600.
    mell = angular_terms.MultiEll(ells, CHI_MAX, limber)
    return mell


def my_plots(mult_ell=None, chi=1000):
    matplotlib.rcParams.update({'font.size': 16,
                                'text.usetex' : False,
                                'figure.autolayout': True})

    if mult_ell is None:
        mult_ell = get_mult_ell()
    CHI = float(chi)
    DELTAS = [5., 10., 20., 50., 100.]
    #DELTAS = []
    Y_MIN = 5e-8
    Y_MAX = 5e-5
    X_MIN = 9
    X_MAX = 1001
    
    kwargs = {
            "linewidth" : 2.,
             }

    title = "$\\bar\\chi=%d\\,\\rm{Mpc}/h$, $\\Delta\\chi=%d\\,\\rm{Mpc}/h$"
    
    for delta in DELTAS:
        plt.figure(tight_layout=True)
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
        plt.legend(loc="upper right", labelspacing=.1, frameon=False)
        plt.savefig("terms_chi%d_delta%d.pdf" % (CHI, delta),
                tightlayout=True, bbox_inches='tight'
                )

    

    # Publication plots.
    
    # The terms.
    D1 = 10.
    D2 = 50.

    plt.figure(tight_layout=True)
    plot_spectra(mult_ell, chi, D1, **kwargs)
    plt.ylim([Y_MIN, Y_MAX])
    plt.xlim([X_MIN, X_MAX])
    plt.xlabel(r"$\ell$",
               fontsize=18,
               )
    plt.ylabel(r"$C^{ss}_\ell(\chi,\chi')$",
               fontsize=18,
               )
    plt.legend(loc="upper right", labelspacing=.1, frameon=False)
    plot_spectra(mult_ell, chi, D2, local_only=True, **kwargs)
    plt.savefig('terms.eps', tightlayout=True, bbox_inches='tight')

    # Mean source density plot
    N_TOTAL = 10000
    F_SKY = 0.5

    chi_s = np.linspace(100, 3500, 50.)
    coef_s = np.array([  d_ln_n(chi) + 2./chi for chi in chi_s ])

    n_bar = n_chi(chi_s)
    norm = integrate.simps(n_bar * chi_s**2, chi_s) * 4 * np.pi * F_SKY
    norm /= N_TOTAL
    n_bar /= norm

    f = plt.figure()
    grid = gridspec.GridSpec(2, 1, wspace=0.0, hspace=0.0)
    ax1 = plt.subplot(grid[0])
    d = chi_s**2 * n_bar * 4 * np.pi * F_SKY
    #d /= np.amax(d)
    ax1.plot(chi_s, d, 'k', **kwargs)
    #plt.ylim(0., 1.1)
    #plt.yticks(np.arange(0.1, 1.0,  0.2))
    plt.ylabel(r"$4\pi f_{\rm sky} \chi^2\bar{n}_f$ ($h/\rm{Mpc})$",
               fontsize=18,
               )
    
    ax2 = plt.subplot(grid[1], sharex=ax1)
    #f.subplots_adjust(hspace=0.001)
    ax2.axhline(y=0.0, color='k', linestyle='--')
    ax2.plot(chi_s, coef_s, 'k', **kwargs)
    plt.ylim(-0.01, 0.02)
    plt.yticks(np.arange(-0.005, 0.02,  0.005))
    #plt.ylabel(r"$(\frac{1}{\bar{n}_f}\frac{d \bar{n}_f}{d \chi}"
    #           r"+ \frac{2}{\chi})$  ($h/\rm{Mpc})$",
    #           fontsize=18,
    #           )
    plt.ylabel(r"$A(\chi)$  ($h/\rm{Mpc})$",
               fontsize=18,
               )
    plt.xlabel(r"$\chi$ ($\rm{Mpc}/h$)",
               fontsize=18,
               )
    xticklabels = ax1.get_xticklabels()
    plt.setp(xticklabels, visible=False)
    #plt.ticklabel_format(axis='y',style='sci',scilimits=(1,1))
    plt.savefig('n_f.eps', tightlayout=True, bbox_inches='tight')


    # Sensitivity plot.
    ells = mult_ell.ells
    ind_100, = np.where(ells == 100)

    delta_chi_bins = 100.
    chi_bins = np.arange(550, 3500, delta_chi_bins)
    n_bar_bins = interpolate.interp1d(chi_s, n_bar)(chi_bins)

    delta_l = np.diff(ells[0::2])
    ell_bins = ells[1:-1:2]
    ell_bins_l = ells[0:-2:2]
    ell_bins_r = ells[2::2]

    noise = 1. / n_bar_bins / chi_bins**2 / delta_chi_bins


    plt.figure(tight_layout=True)
    ax = plt.subplot(111)
    global B_F
    #for min_sep in [12, 8, 4, 1]:
    for min_sep in [1]:
        cum_spectrum = 0.
        cum_spectrum_1 = 0.
        normalization = 0.
        cum_var = 0.
        #cum_noise = 0.
        for ii, chi1 in enumerate(chi_bins):
            for jj, chi2 in enumerate(chi_bins):
                if abs(ii - jj) < min_sep:
                    continue
                var = noise[ii]*noise[jj] / (ell_bins*(ell_bins + 1)*delta_l*F_SKY)
                t1, t2, t3 = apply_coeffs(mult_ell, (chi1 + chi2)/2, chi1 - chi2)
                spectrum = t1 + t2 + t3
                weight = spectrum[ind_100] / (noise[ii]*noise[jj])
                cum_spectrum += weight * spectrum
                cum_var += weight**2 * var
                normalization += abs(weight)
                b_f_tmp = B_F
                B_F = 0.8
                t1, t2, t3 = apply_coeffs(mult_ell, (chi1 + chi2)/2, chi1 - chi2)
                spectrum = t1 + t2 + t3
                cum_spectrum_1 += weight * spectrum
                B_F = b_f_tmp
        errors = np.sqrt(cum_var) / normalization
        spectrum = cum_spectrum / normalization
        spectrum_1 = cum_spectrum_1 / normalization
        # Effective noise power spectrum.
        noise_spec = errors * np.sqrt(ell_bins*(ell_bins + 1)*delta_l*F_SKY)
        #plt.loglog(ell_bins, noise_spec, 'k--')
        spectrum_bins = interpolate.interp1d(ells, spectrum)(ell_bins)
        plt.bar(left=ell_bins_l,
                height=2 * errors,
                width=ell_bins_r - ell_bins_l,
                bottom=spectrum_bins - errors, 
                color='y',
                lw=0.5,
                )
        #plt.errorbar(ell_bins, spectrum_bins, errors, color='k', marker='', ls='',
        #        **kwargs)
        s0 = plt.loglog(ells, spectrum, 'k-', **kwargs)
        s1 = plt.loglog(ells, spectrum_1, 'b--', **kwargs)
    ax.set_aspect(0.4)
    plt.ylabel(r"$\langle {C}^{ss}_\ell \rangle$", fontsize=18)
    plt.xlabel(r"$\ell$", fontsize=18)
    Y_MIN = 5e-7
    Y_MAX = 2e-4
    plt.ylim([Y_MIN, Y_MAX])
    plt.xlim([X_MIN, X_MAX])
    plt.legend((r'$b_f=1.3$', r'$b_f=0.8$'), loc=1, frameon=False)
    plt.savefig('sensitivity.eps', tightlayout=True, bbox_inches='tight')

    
    




