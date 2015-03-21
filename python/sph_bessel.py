"""Spherical bessel functions and an integrator."""

import numpy as np
import math

from scipy import special, integrate


def jn(n, x):
    """Not safe for x = 0."""
    return np.sqrt(np.pi / (2 * x)) * special.jv(n + 1./2, x)


def approx_jn(n, x):
    nu = n + 1./2
    arg = np.pi / 4
    arg -= np.arccos(nu / x) * nu
    arg += np.sqrt(x * x - nu * nu)
    out = np.sin(arg)
    out /= x * (1 - nu * nu / x / x)**(1./4)
    return out


def jnjn(n, mean, delta, x):
    a = mean - delta/2
    b = mean + delta/2
    return jn(n, a * x) * jn(n, b * x)


def approx_jnjn(n, mean, delta, x):
    a = mean - delta/2
    b = mean + delta/2
    nu = n + 1./2
    t = np.sqrt(mean**2 * x**2 - nu**2)
    arg = (mean * delta) * (x**2 / t)
    arg += (delta**2 / 2) * (x**2 / t)
    arg -= (mean**2 * delta**2 / 2) * (x**4 / t**3)
    arg -= np.arccos(nu / b / x) * nu
    arg += np.arccos(nu / a / x) * nu
    out = (1./2) * np.cos(arg)
    out /= a * x * (1 - nu**2 / (a * x)**2)**(1./4)
    out /= b * x * (1 - nu**2 / (b * x)**2)**(1./4)
    return out


def sample_jnjn(n, mean, delta, x_max):
    """Doesn't work for n=0.

    Notes
    -----

    Integration is in 4 Sections:
    - 0: x*mean = 0 to n - 2*sqrt(n), 0 points
    - 1: x*mean = n-2*sqrt(n) to n, 9 points
    - 2: x*mean = n to 2n, n points
    - 3: x*mean = 2n to max

    """

    # Figure out total number of samples:
    bound_lower = max((n - 1 * math.sqrt(n)) / mean, 0)
    bound_middle = n / mean
    bound_upper = (n + 5 * math.sqrt(n)) / mean
    if bound_upper > x_max:
        bound_upper = x_max
        nparts_upper = 0
        delta_upper = 0
    else:
        nparts_upper = max(pow_2_gt((x_max - bound_upper) * delta), 8)
        delta_upper = (x_max - bound_upper) / nparts_upper
    
    nparts_middle = pow_2_gt((bound_upper - bound_middle) * mean)
    nparts_lower = 8
    delta_lower = (bound_middle - bound_lower) / nparts_lower
    delta_middle = (bound_upper - bound_middle) / nparts_middle
    
    points_lower = bound_lower + np.arange(nparts_lower) * delta_lower
    points_middle = bound_middle + np.arange(nparts_middle) * delta_middle
    points_upper = bound_upper + np.arange(nparts_upper + 1) * delta_upper

    all_points = np.concatenate((points_lower, points_middle, points_upper))

    return (all_points, (nparts_lower, nparts_middle, nparts_upper), 
            (delta_lower, delta_middle, delta_upper))


def pow_2_gt(n):
    """Power of 2 greater than given value."""

    k = int(math.ceil(math.log(n) / math.log(2)))
    return 2**k







