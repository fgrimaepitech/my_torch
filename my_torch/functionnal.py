from ctypes import Union
from my_torch.tensor import Tensor
import numpy as np
import numpy.lib.stride_tricks as np_stride_tricks

from my_torch.function import Function
import my_torch.derivatives as dv
from .device import get_array_module

def max(tensor : Tensor, floor: Union[int, float] = 0) -> Tensor:
    xp = get_array_module(tensor.device)
    return Tensor(xp.maximum(floor, tensor.data), requires_grad=tensor.requires_grad, device=tensor.device, grad_fn=Function(dv.max_backward, [tensor, floor]))

def as_strided(tensor: Tensor, shape: tuple, strides: tuple, storage_offset: int = 0) -> 'Tensor':
    xp = get_array_module(tensor.device)
    np_array = tensor.data

    itemsize = np_array.itemsize
    byte_strides = tuple(s * itemsize for s in strides)
    
    if storage_offset > 0:
        flat_array = np_array.flatten()
        if storage_offset >= len(flat_array):
            raise ValueError(f"storage_offset {storage_offset} is out of bounds for array of size {len(flat_array)}")
        offset_array = flat_array[storage_offset:]
        np_array = offset_array

    # as_strided is numpy-specific, so we may need to convert for GPU
    if tensor.device == 'cuda':
        # For GPU, convert to numpy, do operation, convert back
        import cupy as cp
        np_array_cpu = cp.asnumpy(np_array) if isinstance(np_array, cp.ndarray) else np_array
        strided_data_cpu = np_stride_tricks.as_strided(
            np_array_cpu,
            shape=shape,
            strides=byte_strides,
        )
        strided_data = cp.asarray(strided_data_cpu)
    else:
        strided_data = np_stride_tricks.as_strided(
            np_array,
            shape=shape,
            strides=byte_strides,
        )

    return Tensor(strided_data, requires_grad=tensor.requires_grad, device=tensor.device, grad_fn=Function(dv.as_strided_backward, [tensor, shape, strides, storage_offset]))

def trace(tensor: Tensor) -> Tensor:
    xp = get_array_module(tensor.device)
    result = xp.trace(tensor.data)
    if not hasattr(result, 'shape') or result.shape == ():
        result = xp.array(result)
    return Tensor(result, requires_grad=tensor.requires_grad, device=tensor.device, grad_fn=Function(dv.trace_backward, [tensor]))
