from my_torch.neural_network import Module
from my_torch.tensor import Tensor
import numpy as np

class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return 1 / (1 + np.exp(-x))

    def parameters(self):
        return []