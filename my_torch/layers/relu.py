from my_torch.neural_network import Module
import numpy as np

from my_torch.tensor import Tensor

class ReLU(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return Tensor(np.maximum(0, x.data), requires_grad=x.requires_grad)

    def extra_repr(self):
        return ""
