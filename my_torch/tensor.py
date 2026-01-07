import numpy as np
from typing import Union, Optional

import my_torch.derivatives as dv
from .function import Function

class Tensor:
    """
    A tensor is a multi-dimensional array.
    """
    def __init__(self, data: Union[np.ndarray, list, tuple], requires_grad: bool = False, grad_fn: Optional["Function"] = None):
        assert isinstance(data, (np.ndarray, list, tuple)), "data must be a numpy array, list, or tuple"
        self.data = np.array(data)
        self.requires_grad = requires_grad
        self.grad_fn = grad_fn
        self.grad = None

    def __str__(self):
        return f"tensor({str(self.data)}, requires_grad={self.requires_grad})"

    def requires_grad_(self, requires_grad: bool = True):
        self.requires_grad = requires_grad
        return self

    def __mul__(self, other: Union[int, float, 'Tensor']) -> 'Tensor':
        return Tensor(self.data * (other if isinstance(other, (int, float)) else other.data), requires_grad=self.requires_grad, grad_fn=Function(dv.mul_backward, [self, other]))
    
    def __rmul__(self, other: Union[int, float]) -> 'Tensor':
        return Tensor(other * self.data, requires_grad=self.requires_grad, grad_fn=Function(dv.mul_backward, [self, other]))

    def __add__(self, other: Union[int, float, 'Tensor']) -> 'Tensor':
        return Tensor(self.data + (other if isinstance(other, (int, float)) else other.data), requires_grad=self.requires_grad, grad_fn=Function(dv.add_backward, [self, other]))

    def backward(self, grad_outputs: Optional[Tensor] = None):
        if grad_outputs is None:
            grad_outputs = Tensor(np.ones_like(self.data))
        if self.requires_grad and self.grad_fn is not None:
            self.grad_fn.backward([grad_outputs])
    

def tensor(data, requires_grad=False):
    return Tensor(data, requires_grad)