"""Spherical bessel functions and an integrator."""

import numpy as np
import math

from scipy import special, integrate
import matplotlib.pyplot as plt


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
    # The following is the tailor expantion of around delta. Not accurate
    # enough.
    #t = np.sqrt(mean**2 * x**2 - nu**2)
    #arg = (mean * delta) * (x**2 / t)
    #arg += (delta**2 / 2) * (x**2 / t)
    #arg -= (mean**2 * delta**2 / 2) * (x**4 / t**3)
    arg = np.sqrt((b*x)**2 - nu**2)
    arg -= np.sqrt((a*x)**2 - nu**2)
    arg -= np.arccos(nu / b / x) * nu
    arg += np.arccos(nu / a / x) * nu
    out = (1./2) * np.cos(arg)
    out /= a * x * (1 - nu**2 / (a * x)**2)**(1./4)
    out /= b * x * (1 - nu**2 / (b * x)**2)**(1./4)
    return out


def sample_jnjn(n, mean, delta, x_max):
    """

    Notes
    -----

    Integration is in 4 Sections:
    - 0: x*mean = 0 to n - 2*sqrt(n), 0 points
    - 1: x*mean = n-2*sqrt(n) to n, 9 points
    - 2: x*mean = n to 2n, n points
    - 3: x*mean = 2n to max

    """

    adelta = abs(delta)
    if adelta <= mean * 1e-6:
        adelta = mean * 1e-6
    umean = mean + adelta / 2
    lmean = mean - adelta / 2
    # Figure out total number of samples:
    bound_lower = max((n - 2 * math.sqrt(n)) / umean, 0)
    #bound_middle = n / mean
    bound_upper = max(n + 15 * math.sqrt(n), n + 40) / lmean
    if bound_upper >= x_max:
        bound_upper = x_max
        nparts_upper = 0
        delta_upper = 0
        if bound_lower > x_max:
            bound_lower = x_max - 1. / umean
    else:
        nparts_upper = max(pow_2_gt(4. * (x_max - bound_upper) * adelta), 128)
        delta_upper = (x_max - bound_upper) / nparts_upper
    
    #print bound_upper, bound_lower, x_max
    #print 5. * (bound_upper - bound_lower) * mean
    nparts_lower = max(pow_2_gt(5. * (bound_upper - bound_lower) * mean), 16)
    #nparts_lower = 8
    #delta_lower = (bound_middle - bound_lower) / nparts_lower
    delta_lower = (bound_upper - bound_lower) / nparts_lower
    
    points_lower = bound_lower + np.arange(nparts_lower) * delta_lower
    #points_middle = bound_middle + np.arange(nparts_middle) * delta_middle
    points_upper = bound_upper + np.arange(nparts_upper + 1) * delta_upper

    all_points = np.concatenate((points_lower,  points_upper))

    return (all_points, (nparts_lower,  nparts_upper), 
            (delta_lower, delta_upper))

def integrate_f_jnjn(f, n, mean, delta, x_max):
    """Doesn't work for n=0, because we set f=0 for the first point.
    """

    if n == 0:
        raise NotImplementedError()

    x_max = float(x_max)
    mean = float(mean)
    delta = float(delta)

    # Reduce x_max if envelope is significant.
    if delta == 0:
        envelop_width = 1000 * x_max
    else:
        envelop_width = 5 * (2 * np.pi / delta)
    x_max = min(x_max, 5 * envelop_width)

    x, ntuple, delta_tuple = sample_jnjn(n, mean, delta, x_max)

    # Envelope of a Gaussian with width of several oscillations. This
    # controls the oscillations out to high k.
    envelope = np.exp(-0.5*(x - n / mean)**2 / envelop_width**2)

    f_eval = np.empty_like(x)
    f_eval[0] = 0
    f_eval[1:] = f(x[1:])

    lower = np.s_[:ntuple[0] + 1]
    jnjn_lower = np.empty_like(x[lower])
    jnjn_lower[0] = 0
    jnjn_lower[1:] =  jnjn(n, mean, delta, x[lower][1:])
    integral = integrate.romb(f_eval[lower] * jnjn_lower, dx=delta_tuple[0])
    
    if ntuple[1]:
        upper = np.s_[ntuple[0]:]
        jnjn_upper = approx_jnjn(n, mean, delta, x[upper])
        integral += integrate.romb(f_eval[upper] * jnjn_upper * envelope[upper],
                                   dx=delta_tuple[1])

    # XXX
    #print "n points:", ntuple
    #plt.plot(x[lower], jnjn_lower * f_eval[lower])
    #plt.plot(x[upper], jnjn_upper * f_eval[upper])
    #plt.plot(x[lower], jnjn_lower * f_eval[lower] * envelope[lower])
    #plt.plot(x[upper], jnjn_upper * f_eval[upper] * envelope[upper])
    #plt.show()

    return integral

def integrate_f_jnjn_brute(f, n, mean, delta, x_max):

    x_max = float(x_max)
    mean = float(mean)
    delta = float(delta)

    x_min = n / mean / 2
    npoints = 4 * mean * (x_max - x_min)
    npoints = pow_2_gt(npoints)

    delta_x = (x_max - x_min) / npoints
    x = np.arange(npoints + 1, dtype=float) * delta_x + x_min
    integrand = np.empty_like(x)
    integrand[0] = 0
    integrand[0:] = f(x[0:])
    integrand *= jnjn(n, mean, delta, x)

    # XXX
    #print "brute n points:", npoints
    #plt.plot(x, integrand)

    return integrate.romb(integrand, dx=delta_x)


def pow_2_gt(n):
    """Power of 2 greater than given value."""

    k = int(math.ceil(math.log(n) / math.log(2)))
    return 2**k







