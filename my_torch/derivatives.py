from __future__ import annotations
from typing import Any
import my_torch.tensor as ts
import numpy as np

def mul_backward(tensors: Any, grad_outputs: Any) -> Any:
    if isinstance(tensors[0], ts.Tensor) and tensors[0].requires_grad:
        tensors[0].grad = grad_outputs[0] * (tensors[1] if isinstance(tensors[1], (int, float)) else tensors[1].data)
        tensors[0].backward(tensors[0].grad)
    if isinstance(tensors[1], ts.Tensor) and tensors[1].requires_grad:
        tensors[1].grad = grad_outputs[0] * (tensors[0] if isinstance(tensors[0], (int, float)) else tensors[0].data)
        tensors[1].backward(tensors[1].grad)
    else:
        return grad_outputs[0] * tensors[1]
    return grad_outputs[0] * tensors[0]

def add_backward(tensors: Any, grad_outputs: Any) -> Any:
    if isinstance(tensors[0], ts.Tensor) and tensors[0].requires_grad:
        tensors[0].grad = grad_outputs[0]
        tensors[0].backward(tensors[0].grad)
    if isinstance(tensors[1], ts.Tensor) and tensors[1].requires_grad:
        tensors[1].grad = grad_outputs[0]
        tensors[1].backward(tensors[1].grad)
    else:
        return grad_outputs[0]
    return grad_outputs[0]

def sub_backward(tensors: Any, grad_outputs: Any) -> Any:
    if isinstance(tensors[0], ts.Tensor) and tensors[0].requires_grad:
        tensors[0].grad = grad_outputs[0]
        tensors[0].backward(tensors[0].grad)
    if isinstance(tensors[1], ts.Tensor) and tensors[1].requires_grad:
        tensors[1].grad = -grad_outputs[0]
        tensors[1].backward(tensors[1].grad)
    else:
        return -grad_outputs[0]
    return grad_outputs[0]

def neg_backward(tensors: Any, grad_outputs: Any) -> Any:
    if isinstance(tensors[0], ts.Tensor) and tensors[0].requires_grad:
        tensors[0].grad = -grad_outputs[0] if isinstance(grad_outputs[0], (int, float)) else -grad_outputs[0].data
        tensors[0].backward(tensors[0].grad)
    else:
        return -grad_outputs[0] if isinstance(grad_outputs[0], (int, float)) else -grad_outputs[0].data
    return grad_outputs[0]

def truediv_backward(tensors: Any, grad_outputs: Any) -> Any:
    is_reverse_div = isinstance(tensors[0], (int, float)) and isinstance(tensors[1], ts.Tensor)

    if is_reverse_div:
        if isinstance(tensors[1], ts.Tensor) and tensors[1].requires_grad:
            tensors[1].grad = -grad_outputs[0] * tensors[0] / (tensors[1].data ** 2)
            tensors[1].backward(tensors[1].grad)
        return grad_outputs[0]

    if isinstance(tensors[0], ts.Tensor) and tensors[0].requires_grad:
        tensors[0].grad = grad_outputs[0] / (tensors[1] if isinstance(tensors[1], (int, float)) else tensors[1].data)
        tensors[0].backward(tensors[0].grad)
    if isinstance(tensors[1], ts.Tensor) and tensors[1].requires_grad:
        tensors[1].grad = -grad_outputs[0] * tensors[0].data / (tensors[1].data ** 2)
        tensors[1].backward(tensors[1].grad)
    return grad_outputs[0]

def matmul_backward(tensors: Any, grad_outputs: Any) -> Any:
    grad_data = grad_outputs[0].data if isinstance(grad_outputs[0], ts.Tensor) else grad_outputs[0]
    if isinstance(tensors[0], ts.Tensor) and tensors[0].requires_grad:
        tensors[0].grad = grad_data @ tensors[1].data.T
        tensors[0].backward(tensors[0].grad)
    if isinstance(tensors[1], ts.Tensor) and tensors[1].requires_grad:
        tensors[1].grad = tensors[0].data.T @ grad_data
        tensors[1].backward(tensors[1].grad)
    return grad_data @ tensors[1].data.T

def max_backward(tensors: Any, grad_outputs: Any) -> Any:
    if isinstance(tensors[0], ts.Tensor) and tensors[0].requires_grad:
        tensors[0].grad = grad_outputs[0] * (tensors[0].data > tensors[1])
        tensors[0].backward(tensors[0].grad)
    return grad_outputs[0] * (tensors[0].data > tensors[1])

def sum_backward(tensors: Any, grad_outputs: Any) -> Any:
    grad_data = grad_outputs[0].data if isinstance(grad_outputs[0], ts.Tensor) else grad_outputs[0]
    
    if isinstance(tensors[0], ts.Tensor) and tensors[0].requires_grad:
        if not isinstance(grad_data, np.ndarray):
            grad_data = np.array(grad_data)
        tensors[0].grad = np.broadcast_to(grad_data, tensors[0].data.shape).copy()
        tensors[0].backward(tensors[0].grad)
    return grad_data