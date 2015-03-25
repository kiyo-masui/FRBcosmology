import unittest

import numpy as np
import matplotlib.pyplot as plt

import angular_terms

class TestSingleEll(unittest.TestCase):

    def test_init(self):
        pass
        s = angular_terms.SingleEll(80, 3000)
        delta = np.arange(0.1, 1800, 5)
        plt.loglog(delta, abs(s.i1(2000, delta)))
        plt.show()


class TestMultiEll(unittest.TestCase):

    def test_init(self):
        pass
        #ells = np.arange(10, 800, 10)
        #at = angular_terms.MultiEll(ells, 3000)
        #plt.show()



#s = angular_terms.SingleEll(200, 1000)
if __name__ == "__main__":
    unittest.main()
