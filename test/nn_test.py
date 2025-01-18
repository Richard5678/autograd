import unittest
import numpy as np
from autograd.nn import Linear


class TestNN(unittest.TestCase):
    def test_nn(self):
        in_dim, out_dim = 3, 5
        B = 1
        linear = Linear(in_dim, out_dim)
        X = np.ones((B, 3))
        y = linear.forward(X)

        dy = np.zeros_like(y)
        dx, dw = linear.backward(dy)

        self.assertEqual(dx.shape, X.shape)
        self.assertEqual(dw.shape, linear.W.shape)


if __name__ == "__main__":
    unittest.main()
