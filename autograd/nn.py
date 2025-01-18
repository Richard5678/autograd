import numpy as np


class Tensor:
    def __init__(self, value, prev=[]):
        self.value = value
        self.grad = None
        self._prev = set(prev)

    def backward(self):
        """Backward pass
        - Find computational graph via topological sort
        - Backpropagation
        """

        # topological sort
        topo_order = []
        visited = set()

        def build_topo(node):
            visited.add(node)
            for prev in node._prev:
                if prev not in visited:
                    build_topo(prev)

            topo_order.append(node)

        build_topo(self)

        if self.grad == None:
            self.grad = Tensor(np.ones_like(self.value))

        # back propagatioin
        for tensor in reversed(topo_order):
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
    def __init__(self, in_dim, out_dim):
        self.W = Tensor(np.random.random((in_dim, out_dim)))

    def forward(self, X: Tensor) -> Tensor:
        self.X = X
        Y = Tensor(X.value.dot(self.W.value), prev=[X, self.W])
        Y._backward = self.backward

        return Y

    def backward(self, dy: Tensor) -> Tensor:
        dx = Tensor(dy.value.dot(self.W.value.T))
        dw = Tensor(self.X.value.T.dot(dy.value))

        return dx, dw
