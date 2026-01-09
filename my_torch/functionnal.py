from ctypes import Union
from my_torch.tensor import Tensor
import numpy as np
import numpy.lib.stride_tricks as np_stride_tricks

from my_torch.function import Function
import my_torch.derivatives as dv

def max(tensor : Tensor, floor: Union[int, float] = 0) -> Tensor:
    return Tensor(np.maximum(floor, tensor.data), requires_grad=tensor.requires_grad, grad_fn=Function(dv.max_backward, [tensor, floor]))

def as_strided(tensor: Tensor, shape: tuple, strides: tuple, storage_offset: int = 0) -> 'Tensor':
    np_array = tensor.data

    itemsize = np_array.itemsize
    byte_strides = tuple(s * itemsize for s in strides)

    strided_data = np_stride_tricks.as_strided(
        np_array,
        shape=shape,
        strides=byte_strides,
    )

    return Tensor(strided_data, requires_grad=tensor.requires_grad, grad_fn=Function(dv.as_strided_backward, [tensor, shape, strides, storage_offset]))
