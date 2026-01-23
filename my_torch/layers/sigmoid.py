from my_torch.neural_network import Module
from my_torch.tensor import Tensor
import numpy as np
from my_torch.device import get_array_module

class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        xp = get_array_module(x.device)
        y = 1 / (1 + xp.exp(-x.data))
        return Tensor(y, requires_grad=x.requires_grad, device=x.device)

    def parameters(self):
        return []