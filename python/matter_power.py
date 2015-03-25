from os import path

import numpy as np
from scipy import interpolate

from cora.util.cosmology import Cosmology


DATA_DIR = path.join(path.dirname(__file__), 'data')
FILE_PAT = "Pk_z%3.1f.dat"
REDSHIFTS = np.arange(0, 2.001, 0.2)
K_MAX = 2.

# k values are the same for all redshifts.
K = np.loadtxt(path.join(DATA_DIR, FILE_PAT % REDSHIFTS[0]))[:,0]
DATA = np.concatenate([np.loadtxt(path.join(DATA_DIR, FILE_PAT % z))[None,:,1]
                       for z in REDSHIFTS])


def interpolator():
    c = Cosmology()
    chi = c.comoving_distance(REDSHIFTS)
    interpolator = interpolate.interp2d(K, chi, DATA, kind='cubic')
    return interpolator


INTERPOLATOR = interpolator()

def p_k_chi_interp():
    return INTERPOLATOR


def p_k_interp(chi):
    i = INTERPOLATOR
    data = i(K, chi)
    return interpolate.interp1d(K, data, kind='cubic')
