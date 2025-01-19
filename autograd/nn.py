import numpy as np
from autograd.tensor import Tensor


# Ops:
#   - Forward
#   - Backward
class Linear:
    def __init__(self, input_dim, output_dim):
        # self.W = Tensor(np.random.random((output_dim, input_dim)))
        self.W = Tensor(np.ones((output_dim, input_dim)))
        # self.b = Tensor(np.random.random(()))

    def forward(self, X: Tensor) -> Tensor:
        """Linear transformation on X

        Args:
            X (Tensor): shape = (B, input_dim)

        Returns:
            Tensor: shape = (B, output_dim)
        """
        self.X = X
        Y = Tensor(X.value.dot(self.W.value.T), prev=[X, self.W])
        Y._backward = self.backward
        Y._op = "linear"

        return Y

    def backward(self, dL_dY: Tensor) -> Tensor:
        """Backward pass of linear layer

        Args:
            dL_dY (Tensor): dL/dY shape = Y.shape

        Returns:
            Tensor: dL/dX shape = X.shape, dL/dW  shape = W.shape
        """

        # print(type(dL_dY))
        dL_dX = Tensor(dL_dY.value.dot(self.W.value))
        dL_dW = Tensor(dL_dY.value.T.dot(self.X.value))

        return dL_dX, dL_dW
