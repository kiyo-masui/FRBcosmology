
import numpy as np
from numpy.random import rand
from scipy.special import eval_legendre
import matplotlib.pyplot as plt

n_frb = 1000

l_bin_max = np.arange(1000, 100, 100)
n_l_bin = len(l_bin_max)

# Generate random catalogue positions.
ra = rand(n_frb) * 2 * np.pi
dec = np.arcsin(2*rand(n_frb) -1)

n_dot_n = (np.sin(ra[:,None]) * np.sin(ra[None,:])
           + np.cos(ra[:,None]) * np.cos(ra[None,:])
           * np.cos(dec[:,None] - dec[None,:]))

C_tilde_alpha = np.zeros((n_l_bin, n_frb, n_frb), dtype=np.float64)

l_bin_min = 0
for ii in range(n_l_bin):
    for ll in range(l_bin_min, l_bin_max[ii]):
        legendre = eval_legendre(ll, n_dot_n)
        C_tilde_alpha[ii,:,:] += ((2 * ll + 1) / 2 / np.pi) * legendre
    l_bin_min = l_bin_max[ii]
C_tilde_alpha.flat[::n_frb + 1] = 0

auto_var = 200**2   # Pulled random number from Matt McQuinn's paper. pc/cm^3.





#plt.plot(n_dot_n, 'b.')
#plt.show()

#plt.plot(ra, dec, '.')
#plt.show()
