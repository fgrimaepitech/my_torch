from energizer.neural_network import Module
from energizer.tensor import Tensor
import numpy as np
import mlx.core as mx

class Sigmoid(Module):
    def __init__(self, device: str = 'cpu'):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        if x.device == 'gpu':
            y = 1 / (1 + mx.exp(-x.data))
        else:
            y = 1 / (1 + np.exp(-x.data))
        return Tensor(y, requires_grad=x.requires_grad, device=x.device)
