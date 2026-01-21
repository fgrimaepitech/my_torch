import numpy as np
from typing import Union, Optional

import my_torch.derivatives as dv
from .function import Function

class Tensor:
    """
    A tensor is a multi-dimensional array.
    """
    def __init__(self, data, requires_grad: bool = False, grad_fn: Optional["Function"] = None):
        self.data = np.array(data)
        self.shape = self.data.shape
        self.requires_grad = requires_grad
        self.grad_fn = grad_fn
        self.grad = None

    def __str__(self):
        data_str = np.array2string(
            self.data,
            separator=', ',
            precision=8,
            suppress_small=False,
            floatmode='maxprec_equal'
        )
        return f"tensor({data_str}, requires_grad={self.requires_grad})"

    def requires_grad_(self, requires_grad: bool = True):
        self.requires_grad = requires_grad
        return self

    def __mul__(self, other: Union[int, float, 'Tensor']) -> 'Tensor':
        return Tensor(self.data * (other if isinstance(other, (int, float)) else other.data), requires_grad=self.requires_grad, grad_fn=Function(dv.mul_backward, [self, other]))
    
    def __rmul__(self, other: Union[int, float]) -> 'Tensor':
        return Tensor(other * self.data, requires_grad=self.requires_grad, grad_fn=Function(dv.mul_backward, [self, other]))

    def __add__(self, other: Union[int, float, 'Tensor']) -> 'Tensor':
        return Tensor(self.data + (other if isinstance(other, (int, float)) else other.data), requires_grad=self.requires_grad, grad_fn=Function(dv.add_backward, [self, other]))

    def __sub__(self, other: Union[int, float, 'Tensor']) -> 'Tensor':
        return Tensor(self.data - (other if isinstance(other, (int, float)) else other.data), requires_grad=self.requires_grad, grad_fn=Function(dv.sub_backward, [self, other]))

    def __neg__(self) -> 'Tensor':
        return Tensor(isinstance(self.data, (int, float)) -self.data if isinstance(self.data, (int, float)) else -self.data, requires_grad=self.requires_grad, grad_fn=Function(dv.neg_backward, [self]))

    def __truediv__(self, other: Union[int, float, 'Tensor']) -> 'Tensor':
        return Tensor(self.data / (other if isinstance(other, (int, float)) else other.data), requires_grad=self.requires_grad, grad_fn=Function(dv.truediv_backward, [self, other]))

    def __rtruediv__(self, other: Union[int, float]) -> 'Tensor':
        return Tensor(other / self.data, requires_grad=self.requires_grad, grad_fn=Function(dv.truediv_backward, [other, self]))

    def __matmul__(self, other: 'Tensor') -> 'Tensor':        
        return Tensor(
            self.data @ other.data, 
            requires_grad=self.requires_grad or other.requires_grad,
            grad_fn=Function(dv.matmul_backward, [self, other])
        )

    def __rmatmul__(self, other: 'Tensor') -> 'Tensor':
        return self.__matmul__(other)


    @property
    def T(self) -> 'Tensor':
        return Tensor(self.data.T, requires_grad=self.requires_grad)

    def size(self):
        return self.data.shape

    def sum(self, dim: Optional[int] = None) -> 'Tensor':
        result = np.sum(self.data, axis=dim)
        if not isinstance(result, np.ndarray):
            result = np.array(result)
        return Tensor(result, requires_grad=self.requires_grad, grad_fn=Function(dv.sum_backward, [self, dim]))

    def mean(self, dim: Optional[int] = None) -> 'Tensor':
        result = np.mean(self.data, axis=dim)
        if not isinstance(result, np.ndarray):
            result = np.array(result)
        return Tensor(result, requires_grad=self.requires_grad, grad_fn=Function(dv.mean_backward, [self, dim]))

    @staticmethod
    def randn(*args, **kwargs):
        return Tensor(np.random.randn(*args, **kwargs), requires_grad=False)

    def backward(self, grad_outputs: Optional[Tensor] = None):
        if grad_outputs is None:
            grad_outputs = Tensor(np.ones_like(self.data))
        if self.requires_grad and self.grad_fn is not None:
            self.grad_fn.backward([grad_outputs])
    

def tensor(data, requires_grad=False):
    return Tensor(data, requires_grad)