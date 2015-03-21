import unittest

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate, special

import matter_power



class TestInterpolator(unittest.TestCase):

    def test_z0(self):
        i = matter_power.interpolator()
        k = matter_power.K.copy()
        ans = matter_power.DATA[0.,:]
        self.assertTrue(np.allclose(i(k, 0.0001), ans))




if __name__ == "__main__":
    unittest.main()
