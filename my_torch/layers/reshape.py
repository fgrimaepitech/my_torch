from my_torch.neural_network import Module
from my_torch.tensor import Tensor
import numpy as np

class Reshape(Module):
    def __init__(self, shape: tuple):
        super().__init__()
        self.shape = shape

    def forward(self, x: Tensor) -> Tensor:
        self.input_shape = x.data.shape
        batch_size = x.data.shape[0]

        target_shape = []
        for dim in self.shape:
            if dim == -1:
                total_elements = x.data.size // batch_size
                known_elements = 1
                for d in self.shape:
                    if d != -1:
                        known_elements *= d
                inferred = total_elements // known_elements
                target_shape.append(inferred)
            else:
                target_shape.append(dim)

        if target_shape[0] != batch_size and target_shape[0] != -1:
            target_shape = (batch_size,) + tuple(target_shape)

        reshaped_data = x.data.reshape(target_shape)
        return Tensor(reshaped_data, requires_grad=x.requires_grad)

    def parameters(self):
        return []