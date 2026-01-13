from my_torch.neural_network import Module
from my_torch.tensor import Tensor
import numpy as np

class Flatten(Module):
    def __init__(self, start_dim: int = 1, end_dim: int = -1):
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x: Tensor) -> Tensor:
        if self.start_dim < 0:
            self.start_dim = len(x.data.shape) + self.start_dim
        if self.end_dim < 0:
            self.end_dim = len(x.data.shape) + self.end_dim

        return x.data.reshape(x.data.shape[0], -1)

    def parameters(self):
        return []