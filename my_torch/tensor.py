import numpy as np
from typing import Union

class Tensor:
    """
    A tensor is a multi-dimensional array.
    """
    def __init__(self, data: Union[np.ndarray, list, tuple], requires_grad: bool = False):
        assert isinstance(data, (np.ndarray, list, tuple)), "data must be a numpy array, list, or tuple"
        self.data = np.array(data)
        self.requires_grad = requires_grad

    def __str__(self):
        return f"tensor({str(self.data)}, requires_grad={self.requires_grad})"

def tensor(data, requires_grad=False):
    return Tensor(data, requires_grad)