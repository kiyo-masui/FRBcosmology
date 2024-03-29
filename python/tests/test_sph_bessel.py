import unittest

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate, special

import sph_bessel


camb_dat = np.loadtxt("data/Pk_z0.0.dat")
matter_power = interpolate.interp1d(camb_dat[:,0], camb_dat[:,1])
matter_power_k2 = lambda k: matter_power(k) * k**2
k_camb = camb_dat[1:-1,0].copy()


class TestSph_jn(unittest.TestCase):

    def test_j0(self):
        x = np.arange(1, 1000) / 10.
        right_ans = np.sin(x) / x
        sph_j0 = sph_bessel.jn(0, x)
        self.assertTrue(np.allclose(sph_j0, right_ans))

    def test_jn(self):
        n = 1234
        x = 2300.
        right_ans = special.sph_jn(n, x)[0][-1]
        j = sph_bessel.jn(n, x)
        self.assertTrue(np.allclose(j, right_ans))

    def test_approx(self):
        x = np.arange(10, 50, 0.1)
        n = 5
        j = sph_bessel.jn(n, x)
        approx_j = sph_bessel.approx_jn(n, x)
        #plt.plot(x, j)
        #plt.plot(x, approx_j)
        #plt.plot(x, approx_j-j)
        #plt.show()
        self.assertTrue(np.allclose(j, approx_j, atol=0.003))

    def test_approx_prod(self):
        n = 50
        mean = 1000.
        delta = 10.
        x = np.arange(0.1, 2, 0.001)
        approx = sph_bessel.approx_jnjn(n, mean, delta, x)
        right_ans = sph_bessel.jnjn(n, mean, delta, x)
        ra_s = np.cumsum(right_ans[::-1])
        a_s = np.cumsum(approx[::-1])
        #plt.plot(a * x[::-1], ra_s)
        #plt.plot(a * x[::-1], a_s)
        #plt.show()
        self.assertTrue(np.allclose(a_s, ra_s, atol=0.0001))

class TestIntegration(unittest.TestCase):

    def test_sample(self):
        n = 4
        mean = 1000.
        delta = 100.
        k_max = 2
        x_dense = np.arange(0.001, 2, 0.0001)
        x, n_tuple, delta_tuple = sph_bessel.sample_jnjn(n, mean, delta, k_max)
        #print np.diff(x)
        #print n_tuple, delta_tuple
        #approx = sph_bessel.approx_jnjn(n, mean, delta, x)
        #right_ans = sph_bessel.jnjn(n, mean, delta, x)
        #right_ans_d = sph_bessel.jnjn(n, mean, delta, x_dense)
        #approx_d = sph_bessel.approx_jnjn(n, mean, delta, x_dense)
        #ra_s = np.cumsum(right_ans[::-1])
        #a_s = np.cumsum(approx[::-1])
        #plt.figure()
        #plt.plot(mean * x_dense, right_ans_d)
        #plt.plot(mean * x_dense, approx_d)
        #plt.plot(mean * x, right_ans, '.')
        #plt.figure()
        #plt.plot(mean * x[::-1], ra_s)
        #plt.plot(mean * x[::-1], a_s)
        #plt.show()

    def test_integrate(self):
        n_l = [2**i for i in range(12)]
        mean = 1000.
        delta_l = [0., 1., 10., 20, 50, 100., 300., 600, 1200, 1980]
        k_max = 2
        for n in n_l:
            first = True
            for delta in delta_l:
                I = sph_bessel.integrate_f_jnjn(matter_power_k2, n, mean, delta,
                        k_max)
                I2 = sph_bessel.integrate_f_jnjn_brute(matter_power_k2, n, mean,
                        delta, k_max)
                if first:
                    atol = I2 * 0.05
                    first = False
                #print n, delta, I, I2
                self.assertTrue(np.allclose(I, I2, atol=atol))





if __name__ == "__main__":
    unittest.main()
