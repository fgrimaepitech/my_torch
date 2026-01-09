from my_torch.tensor import Tensor
import numpy as np

from my_torch.function import Function
import my_torch.derivatives as dv

def max(tensor : Tensor) -> Tensor:
    return Tensor(np.maximum(0, tensor.data), requires_grad=tensor.requires_grad, grad_fn=Function(dv.max_backward, [tensor]))