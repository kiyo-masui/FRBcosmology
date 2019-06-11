from os import path

import numpy as np
from scipy import interpolate

from cora.util.cosmology import Cosmology


DATA_DIR = path.join(path.dirname(__file__), 'data')
#FILE_PAT = "Pk_z%3.1f.dat"
FILE_PAT = "camb_00891002_matterpower_z%3.1f.dat"
REDSHIFTS = np.arange(0, 2.001, 0.2)
K_MAX = 20.

# k values are the same for all redshifts.
K = np.loadtxt(path.join(DATA_DIR, FILE_PAT % REDSHIFTS[0]))[:,0]
DATA = np.concatenate([np.loadtxt(path.join(DATA_DIR, FILE_PAT % z))[None,:,1]
                       for z in REDSHIFTS])


def interpolator(fill_value=None):
    c = Cosmology()
    chi = c.comoving_distance(REDSHIFTS)
    if fill_value is not None:
        be = False
    else:
        be = True
    interpolator = interpolate.interp2d(K, chi, DATA, kind='cubic', fill_value=fill_value,
            bounds_error=be)
    return interpolator


INTERPOLATOR = interpolator()

def p_k_chi_interp():
    return INTERPOLATOR


def p_k_interp(chi, fill_value=None):
    if fill_value is not None:
        be = False
    else:
        be = True
    i = INTERPOLATOR
    data = i(K, chi)
    return interpolate.interp1d(K, data, kind='cubic', fill_value=fill_value, bounds_error=be)
