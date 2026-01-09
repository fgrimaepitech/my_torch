from ctypes import Union
from my_torch.neural_network import Module
import numpy as np

from my_torch.tensor import Tensor
from my_torch.functionnal import max

class ReLU(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return max(x)

    def extra_repr(self):
        return ""
