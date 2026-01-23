from __future__ import annotations
from typing import Any
import my_torch.tensor as ts
import numpy as np
from .device import get_array_module

def _get_grad_data(grad_output):
    """Extract gradient data, handling both Tensor and raw arrays."""
    if isinstance(grad_output, ts.Tensor):
        return grad_output.data
    return grad_output

def _create_tensor_grad(tensor, grad_data):
    """Create a gradient tensor with the same device as the original tensor."""
    return ts.Tensor(grad_data, device=tensor.device)

def mul_backward(tensors: Any, grad_outputs: Any) -> Any:
    grad_data = _get_grad_data(grad_outputs[0])
    if isinstance(tensors[0], ts.Tensor) and tensors[0].requires_grad:
        other_data = tensors[1] if isinstance(tensors[1], (int, float)) else (tensors[1].data if isinstance(tensors[1], ts.Tensor) else tensors[1])
        tensors[0].grad = _create_tensor_grad(tensors[0], grad_data * other_data)
        tensors[0].backward(tensors[0].grad)
    if isinstance(tensors[1], ts.Tensor) and tensors[1].requires_grad:
        other_data = tensors[0] if isinstance(tensors[0], (int, float)) else (tensors[0].data if isinstance(tensors[0], ts.Tensor) else tensors[0])
        tensors[1].grad = _create_tensor_grad(tensors[1], grad_data * other_data)
        tensors[1].backward(tensors[1].grad)
    return grad_data

def add_backward(tensors: Any, grad_outputs: Any) -> Any:
    grad_data = _get_grad_data(grad_outputs[0])
    if isinstance(tensors[0], ts.Tensor) and tensors[0].requires_grad:
        tensors[0].grad = _create_tensor_grad(tensors[0], grad_data)
        tensors[0].backward(tensors[0].grad)
    if isinstance(tensors[1], ts.Tensor) and tensors[1].requires_grad:
        tensors[1].grad = _create_tensor_grad(tensors[1], grad_data)
        tensors[1].backward(tensors[1].grad)
    return grad_data

def sub_backward(tensors: Any, grad_outputs: Any) -> Any:
    grad_data = _get_grad_data(grad_outputs[0])
    if isinstance(tensors[0], ts.Tensor) and tensors[0].requires_grad:
        tensors[0].grad = _create_tensor_grad(tensors[0], grad_data)
        tensors[0].backward(tensors[0].grad)
    if isinstance(tensors[1], ts.Tensor) and tensors[1].requires_grad:
        xp = get_array_module(tensors[1].device)
        tensors[1].grad = _create_tensor_grad(tensors[1], -grad_data)
        tensors[1].backward(tensors[1].grad)
    return grad_data

def neg_backward(tensors: Any, grad_outputs: Any) -> Any:
    grad_data = _get_grad_data(grad_outputs[0])
    if isinstance(tensors[0], ts.Tensor) and tensors[0].requires_grad:
        xp = get_array_module(tensors[0].device)
        tensors[0].grad = _create_tensor_grad(tensors[0], -grad_data)
        tensors[0].backward(tensors[0].grad)
    return grad_data

def truediv_backward(tensors: Any, grad_outputs: Any) -> Any:
    grad_data = _get_grad_data(grad_outputs[0])
    is_reverse_div = isinstance(tensors[0], (int, float)) and isinstance(tensors[1], ts.Tensor)

    if is_reverse_div:
        if isinstance(tensors[1], ts.Tensor) and tensors[1].requires_grad:
            xp = get_array_module(tensors[1].device)
            tensors[1].grad = _create_tensor_grad(tensors[1], -grad_data * tensors[0] / (tensors[1].data ** 2))
            tensors[1].backward(tensors[1].grad)
        return grad_data

    if isinstance(tensors[0], ts.Tensor) and tensors[0].requires_grad:
        other_data = tensors[1] if isinstance(tensors[1], (int, float)) else (tensors[1].data if isinstance(tensors[1], ts.Tensor) else tensors[1])
        tensors[0].grad = _create_tensor_grad(tensors[0], grad_data / other_data)
        tensors[0].backward(tensors[0].grad)
    if isinstance(tensors[1], ts.Tensor) and tensors[1].requires_grad:
        xp = get_array_module(tensors[1].device)
        tensors[1].grad = _create_tensor_grad(tensors[1], -grad_data * tensors[0].data / (tensors[1].data ** 2))
        tensors[1].backward(tensors[1].grad)
    return grad_data

def matmul_backward(tensors: Any, grad_outputs: Any) -> Any:
    grad_data = _get_grad_data(grad_outputs[0])
    if isinstance(tensors[0], ts.Tensor) and tensors[0].requires_grad:
        tensors[0].grad = _create_tensor_grad(tensors[0], grad_data @ tensors[1].data.T)
        tensors[0].backward(tensors[0].grad)
    if isinstance(tensors[1], ts.Tensor) and tensors[1].requires_grad:
        tensors[1].grad = _create_tensor_grad(tensors[1], tensors[0].data.T @ grad_data)
        tensors[1].backward(tensors[1].grad)
    return grad_data

def max_backward(tensors: Any, grad_outputs: Any) -> Any:
    grad_data = _get_grad_data(grad_outputs[0])
    if isinstance(tensors[0], ts.Tensor) and tensors[0].requires_grad:
        xp = get_array_module(tensors[0].device)
        mask = tensors[0].data > tensors[1]
        tensors[0].grad = _create_tensor_grad(tensors[0], grad_data * mask)
        tensors[0].backward(tensors[0].grad)
    return grad_data

def sum_backward(tensors: Any, grad_outputs: Any) -> Any:
    grad_data = _get_grad_data(grad_outputs[0])
    
    if isinstance(tensors[0], ts.Tensor) and tensors[0].requires_grad:
        xp = get_array_module(tensors[0].device)
        if not hasattr(grad_data, 'shape') or grad_data.shape == ():
            grad_data = xp.array(grad_data)
        tensors[0].grad = _create_tensor_grad(tensors[0], xp.broadcast_to(grad_data, tensors[0].data.shape).copy())
        tensors[0].backward(tensors[0].grad)
    return grad_data

def mean_backward(tensors: Any, grad_outputs: Any) -> Any:
    grad_data = _get_grad_data(grad_outputs[0])
    if isinstance(tensors[0], ts.Tensor) and tensors[0].requires_grad:
        xp = get_array_module(tensors[0].device)
        grad_per_element = grad_data / tensors[0].data.size
        tensors[0].grad = _create_tensor_grad(tensors[0], xp.broadcast_to(grad_per_element, tensors[0].data.shape).copy())
        tensors[0].backward(tensors[0].grad)
    return grad_data

def as_strided_backward(tensors: Any, grad_outputs: Any) -> Any:
    grad_data = _get_grad_data(grad_outputs[0])
    if isinstance(tensors[0], ts.Tensor) and tensors[0].requires_grad:
        xp = get_array_module(tensors[0].device)
        # Note: as_strided is numpy-specific, may need special handling for cupy
        if tensors[0].device == 'cuda':
            # For GPU, we'll need to handle this differently or convert temporarily
            import numpy as np
            grad_np = np.asarray(grad_data) if hasattr(grad_data, 'get') else grad_data
            strides_np = np.asarray(tensors[1].data.strides) if hasattr(tensors[1].data, 'strides') else tensors[1].data.strides
            offset_np = np.asarray(tensors[2].data) if hasattr(tensors[2].data, 'get') else tensors[2].data
            result_np = np.lib.stride_tricks.as_strided(grad_np, shape=tensors[0].data.shape, strides=strides_np, storage_offset=offset_np)
            tensors[0].grad = _create_tensor_grad(tensors[0], xp.asarray(result_np))
        else:
            tensors[0].grad = _create_tensor_grad(tensors[0], np.lib.stride_tricks.as_strided(grad_data, shape=tensors[0].data.shape, strides=tensors[1].data.strides, storage_offset=tensors[2].data))
        tensors[0].backward(tensors[0].grad)
    return grad_data

def trace_backward(tensors: Any, grad_outputs: Any) -> Any:
    grad_data = _get_grad_data(grad_outputs[0])
    if isinstance(tensors[0], ts.Tensor) and tensors[0].requires_grad:
        xp = get_array_module(tensors[0].device)
        trace_result = xp.trace(grad_data)
        tensors[0].grad = _create_tensor_grad(tensors[0], trace_result)
        tensors[0].backward(tensors[0].grad)
    xp = get_array_module(tensors[0].device) if isinstance(tensors[0], ts.Tensor) else np
    return xp.trace(grad_data)

def conv1d_backward(tensors: Any, grad_outputs: Any, stride: int = 1, padding: int = 0) -> Any:
    x, weight, bias = tensors
    grad_output = _get_grad_data(grad_outputs[0])
    
    # Determine device from input tensor
    device = x.device if isinstance(x, ts.Tensor) else 'cpu'
    xp = get_array_module(device)
    
    weight_data = weight.data if isinstance(weight, ts.Tensor) else weight
    
    if isinstance(x, ts.Tensor) and x.requires_grad:
        grad_x = conv1d_grad_input(grad_output, weight_data, stride, padding, device)
        if grad_x is not None:
            x.grad = _create_tensor_grad(x, grad_x)
            x.backward(x.grad)
    
    if isinstance(weight, ts.Tensor) and weight.requires_grad:
        grad_weight = conv1d_grad_weight(grad_output, x, stride, padding, device)
        if grad_weight is not None:
            weight.grad = _create_tensor_grad(weight, grad_weight)
            weight.backward(weight.grad)
    
    if bias is not None and isinstance(bias, ts.Tensor) and bias.requires_grad:
        grad_bias = xp.sum(grad_output, axis=(0, 2))
        bias.grad = _create_tensor_grad(bias, grad_bias)
        bias.backward(bias.grad)
    
    return grad_output

def conv1d_grad_input(grad_output, weight: 'ts.Tensor', 
                      stride: int = 1, padding: int = 0, device: str = 'cpu') -> Any:
    xp = get_array_module(device)
    batch_size, out_channels, out_length = grad_output.shape
    _, in_channels, kernel_size = weight.data.shape
    
    length = (out_length - 1) * stride + kernel_size - 2 * padding
    
    grad_input_padded = xp.zeros((batch_size, in_channels, length + 2 * padding))
    
    for b in range(batch_size):
        for oc in range(out_channels):
            for ol in range(out_length):
                il_start = ol * stride
                
                for k in range(kernel_size):
                    grad_input_padded[b, :, il_start + k] += \
                        weight.data[oc, :, k] * grad_output[b, oc, ol]
    
    if padding > 0:
        return grad_input_padded[:, :, padding:-padding]
    return grad_input_padded


def conv1d_grad_weight(grad_output, x: 'ts.Tensor',
                       stride: int = 1, padding: int = 0, device: str = 'cpu') -> Any:
    xp = get_array_module(device)
    batch_size, out_channels, out_length = grad_output.shape
    _, in_channels, _ = x.data.shape
    
    kernel_size = x.data.shape[2] - (out_length - 1) * stride + 2 * padding
    
    grad_weight = xp.zeros((out_channels, in_channels, kernel_size))
    
    if padding > 0:
        x_padded = xp.pad(x.data, 
                         ((0, 0), (0, 0), (padding, padding)), 
                         mode='constant')
    else:
        x_padded = x.data
    
    for b in range(batch_size):
        for oc in range(out_channels):
            for ol in range(out_length):
                il_start = ol * stride
                il_end = il_start + kernel_size
                
                x_window = x_padded[b, :, il_start:il_end]
                
                grad_weight[oc] += x_window * grad_output[b, oc, ol]
    
    return grad_weight

def reshape_backward(tensors: Any, grad_outputs: Any) -> Any:
    grad_data = _get_grad_data(grad_outputs[0])
    if isinstance(tensors[0], ts.Tensor) and tensors[0].requires_grad:
        tensors[0].grad = _create_tensor_grad(tensors[0], grad_data.reshape(tensors[0].data.shape))
        tensors[0].backward(tensors[0].grad)
    return grad_data.reshape(tensors[0].data.shape) if isinstance(tensors[0], ts.Tensor) else grad_data