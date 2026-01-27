from energizer.neural_network import Module
from energizer.tensor import Tensor
import numpy as np

class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        y = 1 / (1 + np.exp(-x.data))
        return Tensor(y, requires_grad=x.requires_grad)

    def parameters(self):
        return []