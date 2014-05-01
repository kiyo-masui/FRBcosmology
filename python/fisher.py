import numpy as np
from numpy import linalg
from numpy.random import rand
from scipy.special import eval_legendre
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from numpy import random
#random.seed(10)  # XXX Repeatable.

n_frb = 500
# DTYPE = np.float32
DTYPE = np.float64

#l_bin_max = np.arange(100, 500, 100)
#l_bin_max = np.arange(20, 130, 20)
l_min = 0
step = 1
stop = 5
l_bin_max = np.arange(l_min + step, stop, step)
l_max = l_bin_max[-1]
c_alpha = np.ones(len(l_bin_max)) * 0   # XXX sample variance turned off.
n_l_bin = len(l_bin_max)

# Generate random catalogue positions.
ra = rand(n_frb) * 2 * np.pi
dec = np.arcsin(2*rand(n_frb) - 1)

n_dot_n = (np.sin(ra[:,None]) * np.sin(ra[None,:])
           + np.cos(ra[:,None]) * np.cos(ra[None,:])
           * np.cos(dec[:,None] - dec[None,:]))
# Correct numerical errors on diagonal.
n_dot_n.flat[::n_frb + 1] = 1

C_tilde_alpha = np.zeros((n_l_bin, n_frb, n_frb), dtype=DTYPE)

this_l_bin_min = l_min
for ii in range(n_l_bin):
    for ll in range(this_l_bin_min, l_bin_max[ii]):
        interp_domain = np.linspace(-1., 1., 4 * l_max, endpoint=True)
        legen_interp = interp1d(interp_domain,
                                eval_legendre(ll, interp_domain), kind="linear")
        legendre = legen_interp(n_dot_n)
        C_tilde_alpha[ii,:,:] += ((2 * ll + 1) / 4. / np.pi) * legendre
    C_tilde_alpha[ii].flat[::n_frb + 1] = 0
    this_l_bin_min = l_bin_max[ii]

print "C-tilde-alpha computed."

#auto_var = 200**2   # Pulled random number from Matt McQuinn's paper. pc/cm^3.
auto_var = 100**2   # Round number for testing.

Tr_ab = np.zeros((n_l_bin, n_l_bin), dtype=float)
Tr_gab = np.zeros((n_l_bin, n_l_bin, n_l_bin), dtype=float)

for ii in range(n_l_bin):
    for jj in range(n_l_bin):
        AB = np.dot(C_tilde_alpha[ii,:,:], C_tilde_alpha[jj,:,:])
        AB_diag = AB.flat[::n_frb + 1].astype(float)
        tr = np.sum(AB_diag)
        Tr_ab[ii,jj] = tr
        for kk in range(n_l_bin):
            s = np.sum(AB[:,:] * C_tilde_alpha[kk,:,:], dtype=float)
            Tr_gab[kk,ii,jj] += s
            Tr_gab[kk,jj,ii] += s


#print Tr_ab
F = (Tr_ab - np.sum(c_alpha[:,None,None] * Tr_gab, 0) / auto_var)
F /= 2 * auto_var**2

print F

C = linalg.inv(F)
#print C

C_diag = C.flat[::n_l_bin + 1]
r = C / np.sqrt(C_diag[:,None] * C_diag[None,:])
print r

print l_bin_max
print (np.sqrt(C_diag))


#plt.plot(n_dot_n, 'b.')
#plt.show()

#plt.plot(ra, dec, '.')
#plt.show()
