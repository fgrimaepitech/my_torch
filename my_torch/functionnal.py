from ctypes import Union
from my_torch.tensor import Tensor
import numpy as np

from my_torch.function import Function
import my_torch.derivatives as dv

def max(tensor : Tensor, floor: Union[int, float] = 0) -> Tensor:
    return Tensor(np.maximum(floor, tensor.data), requires_grad=tensor.requires_grad, grad_fn=Function(dv.max_backward, [tensor, floor]))