from my_torch.neural_network import Module
from my_torch.tensor import Tensor
import numpy as np

class Reshape(Module):
    def __init__(self, shape: tuple):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        new_shape = list(self.shape)
        if -1 in new_shape:
            total_elements = np.prod(x.data.shape)
            known_elements = np.prod([abs(d) for d in new_shape if d != -1])
            new_shape[new_shape.index(-1)] = total_elements // known_elements
        
        reshaped = x.data.reshape(tuple(new_shape))
        return Tensor(reshaped, requires_grad=x.requires_grad)


    def parameters(self):
        return []