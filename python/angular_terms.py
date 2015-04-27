"""Computes the variouse integrals, each of which is a term in the angular
PS."""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate, interpolate, fftpack, signal

import sph_bessel
import matter_power



class SingleEll(object):

    def __init__(self, ell, chi_max, limber=False):
        ell = int(ell)
        chi_max = float(chi_max)
        self._ell = ell
        self._chi_max = chi_max

        nchi = 20
        #delta_chi = chi_max / nchi
        #chi_list = np.arange(1, nchi + 1, dtype=float) / nchi * chi_max
        
        CHI_MIN = 500.
        chi_list = np.linspace(CHI_MIN, chi_max, nchi, endpoint=True)
        delta_chi = chi_list[1] - chi_list[0]

        data = []

        n_total = 0
        for chi in chi_list:
            this_chi_data = {}
            p_k_interp = matter_power.p_k_interp(chi)
            # Put in Gaussian cut-off to control oscillations.

            if limber:
                # Local term.
                delta_k = 2 * np.pi / (2 * chi)
                k_max = np.sqrt(max(matter_power.K_MAX**2 - (ell/chi)**2, 0))
                k = np.arange(0, k_max, delta_k)
                k = np.concatenate((k, -k[:0:-1]))
                nk = len(k)   # Guaranteed to be odd.
                if nk > 10:
                    p_k_local = p_k_interp(np.sqrt((ell/chi)**2 + k**2))
                    #window = fftpack.ifftshift(signal.kaiser(nk, beta=14, sym=True))
                    window = fftpack.ifftshift(signal.hann(nk, sym=True))
                    p_k_local *= window
                    fft_norm = len(k) * delta_k / 2 / np.pi / chi**2
                    I1 = fftpack.ifft(p_k_local).real
                    I1 = fftpack.fftshift(I1) * fft_norm
                    deltas = np.linspace(-chi, chi, len(I1), endpoint=True)
                else:
                    I1 = np.zeros(10, dtype=float)
                    deltas = np.linspace(-chi, chi, len(I1), endpoint=True)

                # Limber term.
                if ell/chi > matter_power.K_MAX:
                    I2 = 0
                else:
                    I2 = p_k_interp(ell/chi) / chi**2
            else:
                p_k = lambda k: k**2 * p_k_interp(k)

                scale = ell / chi
                nscales = min(5, 2 * ell - 1)
                # These deltas for the integral.
                deltas, delta_u = exp_sample(scale, nscales)
                ndelta_int = len(deltas)
                # Extra deltas for interpolation.
                delta_max = min(1.99 * chi, 1.99 * (chi_max - chi))
                factor = 1.5
                deltas = list(deltas)
                while factor * deltas[-1] < delta_max:
                    deltas.append(factor * deltas[-1])
                deltas.append(delta_max)
                deltas = np.array(deltas)

                I1 = np.empty_like(deltas)
                for ii in range(len(deltas)):
                    I1[ii] = sph_bessel.integrate_f_jnjn(p_k, ell, chi, deltas[ii],
                            matter_power.K_MAX) * (2 / np.pi)
                I2 = integrate.romb(I1[:ndelta_int]
                                    * np.exp(scale * deltas[:ndelta_int]),
                                    dx=delta_u) * 2

            this_chi_data['chi'] = chi
            this_chi_data['deltas'] = deltas
            this_chi_data['I1'] = I1
            this_chi_data['I2'] = I2
            data.append(this_chi_data)
            n_total += len(deltas)

        I2 = [td["I2"] for td in data]
        I2 = np.array(I2)
        I3 = integrate.cumtrapz(I2, dx=delta_chi, initial=0.)
        for ii in range(nchi):
            data[ii]["I3"] = I3[ii]

        chi_delta = np.empty((n_total, 2), dtype=float)
        I1 = np.empty(n_total, dtype=float)
        ii = 0
        for d in data:
            ndelta = len(d["deltas"])
            chi_delta[ii:ii + ndelta,0] = d["chi"]
            chi_delta[ii:ii + ndelta,1] = d["deltas"]
            I1[ii:ii + ndelta] = d["I1"]
            ii += ndelta

        self.i1 = interpolate.LinearNDInterpolator(chi_delta, I1, fill_value=0.)
        self.i2 = interpolate.interp1d(chi_list, I2, kind="cubic")
        self.i3 = interpolate.interp1d(chi_list, I3, kind="cubic")
        self.data = data



class MultiEll(object):

    def __init__(self, ells, chi_max, limber=False):

        #ells = np.arange(20,500,10, dtype=int)

        integrals = []
        for ell in ells:
            integrals.append(SingleEll(ell, chi_max, limber))

        self.integrals = integrals
        self.ells = ells


    def get_integrals(self, chi, delta_chi):
        chi_m = chi - abs(delta_chi) / 2

        i1 = []
        i2 = []
        i3 = []

        for integ in self.integrals:
            i1.append(integ.i1(chi, delta_chi))
            i2.append(integ.i2(chi_m))
            i3.append(integ.i3(chi_m))

        return np.array(i1), np.array(i2), np.array(i3)


def local_term(ells, chi, delta, flat=False):
    
    out = []
    chi = float(chi)
    delta = float(delta)

    for ell in ells:
        ell = int(ell)
        p_k_interp = matter_power.p_k_interp(chi)

        if flat:
            if (ell/chi) > matter_power.K_MAX:
                out.append(0.)
                continue
            # Maximum k_par.
            k_max = np.sqrt(max(matter_power.K_MAX**2 - (ell/chi)**2, 0))
            # Reduce x_max if envelope is significant.
            if delta == 0:
                envelop_width = 1000 * k_max
            else:
                envelop_width = 5 * (2 * np.pi / delta)
            k_max = min(k_max, 5 * envelop_width)

            nk = sph_bessel.pow_2_gt(k_max * delta * 5) + 1
            k = np.linspace(0, k_max, nk, endpoint=True)

            delta_k = k_max / (nk - 1)
            
            # Envelope of a Gaussian with width of several oscillations. This
            # controls the oscillations out to high k.
            envelope = np.exp(-0.5*k**2 / envelop_width**2)

            p_k_local = p_k_interp(np.sqrt((ell/chi)**2 + k**2))
            p_k_local *= envelope
            # Fourier factor.
            p_k_local *= np.cos(k * delta)

            I1 = integrate.romb(p_k_local, dx=delta_k)
            I1 /= np.pi * chi**2
        else:
            p_k = lambda k: k**2 * p_k_interp(k)
            I1 = sph_bessel.integrate_f_jnjn(p_k, ell, chi, delta,
                    matter_power.K_MAX) * (2 / np.pi)
        out.append(I1)
    return out





def exp_sample(scale, nscale):
    u_max = 1. / scale * (1 - np.exp(-nscale))
    npoint = sph_bessel.pow_2_gt(4 * nscale) + 1
    delta_u = u_max / (npoint - 1)
    u = np.linspace(0, u_max, npoint, endpoint=True)

    x = -1. / scale * np.log(1 - scale * u)
    return x, delta_u

