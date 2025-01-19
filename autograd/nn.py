import numpy as np


class Tensor:
    def __init__(self, value, prev=[]):
        self.value = value
        self._prev = prev
        self.grad = None
        self._backward = None
        self._op = None

    @property
    def shape(self):
        return self.value.shape

    def backward(self):
        # topological sort
        topo_order = []
        visited = set()

        def build_topo(tensor):
            visited.add(tensor)
            for prev in tensor._prev:
                if prev not in visited:
                    build_topo(prev)

            topo_order.append(tensor)

        build_topo(self)

        if self.grad == None:
            self.grad = Tensor(np.ones_like(self.value))

        # backprop
        for tensor in reversed(topo_order):
            if tensor._backward == None:
                continue

            grads = tensor._backward(tensor.grad)
            for prev, grad in zip(tensor._prev, grads):
                if prev.grad == None:
                    prev.grad = grad
                else:
                    prev.grad += grad


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
