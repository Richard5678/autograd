import unittest
import numpy as np
from autograd.nn import Tensor, Linear

class TestNN(unittest.TestCase):
    def test_nn(self):
        in_dim, out_dim = 3, 5
        B = 1
        linear = Linear(in_dim, out_dim)
        X = Tensor(np.ones((B, in_dim)))
        y = linear.forward(X)

        dy = Tensor(np.ones_like(y.value))
        dx, dw = linear.backward(dy)

        self.assertEqual(dx.shape, X.shape)
        self.assertEqual(dw.shape, linear.W.shape)


if __name__ == "__main__":
    unittest.main()
