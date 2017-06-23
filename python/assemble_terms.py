import numpy as np

from scipy import special, integrate, interpolate
from cora.util.cosmology import Cosmology
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib

import matter_power
import angular_terms

# Redshift at which we can detect event of luminocity L*.
Z_STAR = 0.8
ALPHA = -1.

B_E = 0.0
B_F = 1.0


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
        plt.loglog(ells, trans(t1), '-k',
                label='$\\Delta\\chi=%d\\,\\rm{Mpc}/h$' % delta, **kwargs)
    else:
        plt.loglog(ells, trans(t1 + t2 + t3), '-k', label='total', **kwargs)
        plt.loglog(ells, trans(t1), '--b', label='local term', **kwargs)
        plt.loglog(ells, trans(t3), ':r', label='integral term', **kwargs)
        plt.loglog(ells, trans(t2), '-.g', label='cross term', **kwargs)


def get_ells():
    ells = range(10, 101, 10)
    factor = 1.1
    max = 5000   # XXX
    while ells[-1] < max:
        ells.append(int(factor * ells[-1]))
    return np.array(ells)


def get_mult_ell(ells=None, limber=True):
    if ells is None:
        ells = get_ells()
    CHI_MAX = 1500.
    mell = angular_terms.MultiEll(ells, CHI_MAX, limber)
    return mell


def my_plots(mult_ell=None, chi=1200):
    matplotlib.rcParams.update({'font.size': 12,
                                'text.usetex' : False,
                                'figure.autolayout': True})

    if mult_ell is None:
        mult_ell = get_mult_ell()
    CHI = float(chi)
    DELTAS = [0., 1., 5., 10., 20., 50., 100.]
    #DELTAS = []
    CHI_BIN = 600.

    kwargs = {
            "linewidth" : 2.,
             }

    title = "$\\bar\\chi=%d\\,\\rm{Mpc}/h$, $\\Delta\\chi=%d\\,\\rm{Mpc}/h$"

    plt.figure(tight_layout=True)
    for delta in DELTAS:
        plot_spectra(mult_ell, CHI, delta, local_only=True, **kwargs)
    plt.xlabel(r"$\ell$",
               fontsize=14,
               )
    plt.ylabel(r"$C^{ss}_\ell(\chi,\chi')$",
               fontsize=14,
               )
    plt.title("$\\bar\\chi=%d\\,\\rm{Mpc}/h$" % CHI)
    plt.legend(loc="best", labelspacing=.1, frameon=False)
    plt.savefig('local_terms.eps', tightlayout=True, bbox_inches='tight')


    # Mean source density plot
    N_TOTAL = 10
    F_SKY = 0.2

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
               fontsize=14,
               )

    ax2 = plt.subplot(grid[1], sharex=ax1)
    #f.subplots_adjust(hspace=0.001)
    ax2.axhline(y=0.0, color='k', linestyle='--')
    ax2.plot(chi_s, coef_s, 'k', **kwargs)
    plt.ylim(-0.01, 0.02)
    plt.yticks(np.arange(-0.005, 0.02,  0.005))
    #plt.ylabel(r"$(\frac{1}{\bar{n}_f}\frac{d \bar{n}_f}{d \chi}"
    #           r"+ \frac{2}{\chi})$  ($h/\rm{Mpc})$",
    #           fontsize=14,
    #           )
    plt.ylabel(r"$A(\chi)$  ($h/\rm{Mpc})$",
               fontsize=14,
               )
    plt.xlabel(r"$\chi$ ($\rm{Mpc}/h$)",
               fontsize=14,
               )
    xticklabels = ax1.get_xticklabels()
    plt.setp(xticklabels, visible=False)
    #plt.ticklabel_format(axis='y',style='sci',scilimits=(1,1))
    plt.savefig('n_f.eps', tightlayout=True, bbox_inches='tight')


    # Calculate the average spectrum over chi bin .
    plt.figure(tight_layout=True)
    ells = mult_ell.ells
    n_delta_samples = 30
    deltas = np.linspace(-CHI_BIN/2, CHI_BIN/2, n_delta_samples, endpoint=True)
    mean_spectrum = 0.
    for ii in range(n_delta_samples):
        if ii == 0:
            # Nested to finely sample half bin near delta=0.
            t1 = 0.
            for jj in range(n_delta_samples):
                delta = jj * CHI_BIN / 2 / n_delta_samples**2 / 2
                #print ii, jj, delta
                t1_tmp, t2, t3 = apply_coeffs(mult_ell, CHI, delta)
                t1 += t1_tmp
                #plt.loglog(ells, t1_tmp, '-r', **kwargs)
            t1 /= n_delta_samples * 2    # 2: this is half a bin.
        else:
            delta = ii * CHI_BIN / 2 / n_delta_samples
            #print ii, delta
            t1, t2, t3 = apply_coeffs(mult_ell, CHI, delta)
        #plt.loglog(ells, t1, '-b', **kwargs)
        mean_spectrum += t1
    # Note this loop ommits the final half bin, which we assume is ~0.
    mean_spectrum /= n_delta_samples

    plt.loglog(ells, mean_spectrum, '-k', **kwargs)
    plt.xlabel(r"$\ell$",
               fontsize=14,
               )
    plt.ylabel(r"$C^{ss}_\ell$",
               fontsize=14,
               )
    plt.title("$\\bar\\chi=%d\\,\\rm{Mpc}/h$, bin=$%d\\,\\rm{Mpc}/h$"
              % (CHI, CHI_BIN))
    plt.savefig('mean_spectrum.eps', tightlayout=True, bbox_inches='tight')


    # Sensitivity plot.
    plt.figure(tight_layout=True)

    ell_bins = ells[1:-1:2]
    ell_bins_l = ells[0:-2:2]
    ell_bins_r = ells[2::2]
    delta_l = ell_bins_r - ell_bins_l

    N_GAL = 1e7
    noise_frb = 4 * np.pi * F_SKY / N_TOTAL
    noise_gal = 4 * np.pi * F_SKY / N_GAL
    spectrum_binned_ell = mean_spectrum[1:-1:2]
    C_tot_frb = spectrum_binned_ell + noise_frb
    C_tot_gal = spectrum_binned_ell + noise_gal
    var =  C_tot_gal * C_tot_frb + spectrum_binned_ell**2
    var /= (ell_bins * (ell_bins + 1) * delta_l * F_SKY)
    errors = np.sqrt(var)

    bottom = spectrum_binned_ell - errors
    height = 2 * errors
    negative = bottom <= 0
    height[negative] += bottom[negative]
    bottom[negative] = 1e-7

    plt.bar(
            left=ell_bins,   # Should be ell_bins_l, but seems to be a bug in plt
            height=height,
            width=ell_bins_r - ell_bins_l,
            bottom=bottom,
            color='y',
            lw=0.5,
            )
    s0 = plt.loglog(ells, mean_spectrum, 'k-', **kwargs)
    plt.xlabel(r"$\ell$",
               fontsize=14,
               )
    plt.ylabel(r"$C^{ss}_\ell$",
               fontsize=14,
               )
    plt.title("$\\bar\\chi=%d\\,\\rm{Mpc}/h$, $\\Delta\\chi=%d\\,\\rm{Mpc}/h$, $N_{\\rm frb}=%d$"
              % (CHI, CHI_BIN, N_TOTAL))
    plt.savefig('sensitivity.eps', tightlayout=True, bbox_inches='tight')

    signal_to_noise_sq = spectrum_binned_ell**2 / errors**2
    cum_sn = np.sqrt(np.cumsum(signal_to_noise_sq))
    plt.figure()
    plt.title("$\\bar\\chi=%d\\,\\rm{Mpc}/h$, $\\Delta\\chi=%d\\,\\rm{Mpc}/h$, $N_{\\rm frb}=%d$"
              % (CHI, CHI_BIN, N_TOTAL))
    plt.loglog(ell_bins, cum_sn, 'ko')
    plt.ylabel(r"Cumulative $S/N$",
               fontsize=14,
               )
    plt.xlabel(r"$\ell_{\rm max}$",
               fontsize=14,
               )
    plt.savefig('cum_sn.eps', tightlayout=True, bbox_inches='tight')





