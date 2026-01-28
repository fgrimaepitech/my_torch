from energizer.neural_network import Module
from energizer.tensor import Tensor
import numpy as np

class MSELoss(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return ((x - y) ** 2).mean()
