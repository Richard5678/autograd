import numpy as np


class Tensor:
    def __init__(self, value, prev=[], op=None):
        self.value = value
        self._prev = prev
        self.grad = None
        self._backward = None
        self._op = op

    @property
    def shape(self):
        return self.value.shape

    def __sub__(self, other):
        return Tensor(self.value - other.value, prev=[self, other], op="sub")

    def __add__(self, other):
        return Tensor(self.value + other.value, prev=[self, other], op="add")

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return Tensor(self.value * other.value, prev=[self, other], op="mul")

    def __rmul__(self, other):
        return self * other

    def __pow__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return Tensor(self.value**other.value, prev=[self, other], op="pow")

    def __rpow__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return Tensor(other.value**self.value, prev=[self, other], op="pow")

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
