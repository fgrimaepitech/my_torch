from my_torch.neural_network import Module
from my_torch.tensor import Tensor
import numpy as np

class Trim(Module):
    def __init__(self, start: int, end: int):
        super().__init__()
        self.start = start
        self.end = end

    def forward(self, x: Tensor) -> Tensor:
        return x.data[self.start:self.end]

    def parameters(self):
        return []