import numpy as np
import matplotlib.pyplot as plt
from scipy import special, integrate

import matter_power


d_H = 3000.

# Roughly volume limited between these limits.
Z_MIN = 0.01
Z_MAX = 0.05

K_CUTOFF = 10.


z_mean = (Z_MIN + Z_MAX) / 2

thetas = np.radians(np.arange(0.1, 20, 0.05))

chi_min = Z_MIN * d_H
chi_max = Z_MAX * d_H

chi = np.linspace(chi_min, chi_max, 4, endpoint=True)
nchi = len(chi)

pk = matter_power.p_k_interp(d_H * z_mean, fill_value=0.)
if False:
    k = np.logspace(-3, 1, 100)
    #plt.figure()
    #plt.plot(k, pk(k))
    plt.figure()
    plt.loglog(k, k * pk(k) * np.exp(-(k / K_CUTOFF)**2))

    plt.figure()
    plt.loglog(k, k**3 * pk(k) / (2 * np.pi**2) * np.exp(-(k / K_CUTOFF)**2))

    #plt.figure()
    #plt.loglog(matter_power.K, matter_power.K**2 * matter_power.DATA[0])


ells = np.arange(5, 2000, 5)

tmp_k = ells[:] / chi[:, None]
P2 = pk(tmp_k) / chi[:, None]**2 * np.exp(-(tmp_k / K_CUTOFF)**2)

plt.figure()
for ii in range(nchi):
    plt.plot(ells, ells * P2[ii])
plt.plot(ells, 2 * special.jv(0, ells * thetas[0]))


integrand = ells * P2[:, None, :] * special.jv(0, ells * thetas[None, :, None])
integrand = integrand / (2 * np.pi)

wtheta_chi = integrate.simps(integrand, ells, even='first')

plt.figure()
for ii in range(nchi):
    plt.semilogx(np.degrees(thetas), wtheta_chi[ii])


window = chi**2
window /= integrate.simps(window, chi)

wtheta = integrate.simps(wtheta_chi * window[:, None], chi, axis=0)

plt.figure()
plt.semilogx(np.degrees(thetas), wtheta)



# Redshift at which we can detect event of luminocity L*.
Z_STAR = 0.5
ALPHA = -1.


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






plt.show()

