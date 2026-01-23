import numpy as np
from typing import Union, Optional

import my_torch.derivatives as dv
from .function import Function
from .device import get_array_module, get_device, asnumpy

class Tensor:
    """
    A tensor is a multi-dimensional array.
    Supports both CPU (numpy) and GPU (CuPy) operations.
    """
    def __init__(self, data, requires_grad: bool = False, grad_fn: Optional["Function"] = None, device: Optional[str] = None):
        # If data is a Tensor, extract its data and device
        if hasattr(data, 'device') and hasattr(data, 'data'):
            # If it's a Tensor, extract its data and use its device if device not specified
            if device is None:
                device = data.device
            data = data.data
        
        # Detect device from data if it's already a cupy array
        if device is None:
            try:
                import cupy as cp
                if isinstance(data, cp.ndarray):
                    device = 'cuda'
                elif isinstance(data, np.ndarray):
                    device = 'cpu'
                else:
                    # Default to 'cuda' if available, else 'cpu'
                    device = get_device(None)
            except ImportError:
                device = 'cpu'
        
        # Determine device (default to 'cuda' if available, else 'cpu')
        self.device = get_device(device)
        
        # Get the appropriate array module (numpy or cupy)
        xp = get_array_module(self.device)
        
        # Convert to the appropriate array type
        if self.device == 'cuda':
            # For GPU, convert numpy arrays to cupy arrays
            if isinstance(data, np.ndarray):
                self.data = xp.asarray(data)
            else:
                # Try to convert to cupy array
                try:
                    self.data = xp.asarray(data)
                except:
                    self.data = xp.array(data)
        else:
            # For CPU, ensure it's a numpy array
            if isinstance(data, np.ndarray):
                self.data = np.asarray(data)
            else:
                # Check if it's a cupy array and convert to numpy
                try:
                    import cupy as cp
                    if isinstance(data, cp.ndarray):
                        self.data = np.asarray(cp.asnumpy(data))
                    else:
                        self.data = np.array(data)
                except ImportError:
                    self.data = np.array(data)
        
        self.shape = self.data.shape
        self.requires_grad = requires_grad
        self.grad_fn = grad_fn
        self.grad = None
    
    def to(self, device: str) -> 'Tensor':
        """
        Move tensor to the specified device.
        
        Args:
            device: 'cpu' or 'cuda'
        
        Returns:
            New tensor on the specified device
        """
        device = get_device(device)
        if self.device == device:
            return self
        
        # Get source and target array modules
        xp_src = get_array_module(self.device)
        xp_dst = get_array_module(device)
        
        # Convert data
        if device == 'cuda':
            # CPU to GPU
            new_data = xp_dst.asarray(self.data)
        else:
            # GPU to CPU
            new_data = asnumpy(self.data)
        
        # Create new tensor with same properties
        new_tensor = Tensor(
            new_data,
            requires_grad=self.requires_grad,
            grad_fn=self.grad_fn,
            device=device
        )
        
        # Copy gradient if it exists
        if self.grad is not None:
            if device == 'cuda':
                new_tensor.grad = Tensor(xp_dst.asarray(self.grad.data), device=device)
            else:
                new_tensor.grad = Tensor(asnumpy(self.grad.data), device=device)
        
        return new_tensor
    
    def cuda(self) -> 'Tensor':
        """Move tensor to GPU."""
        return self.to('cuda')
    
    def cpu(self) -> 'Tensor':
        """Move tensor to CPU."""
        return self.to('cpu')

    def __str__(self):
        # Convert to numpy for string representation if on GPU
        data_for_str = asnumpy(self.data) if self.device == 'cuda' else self.data
        data_str = np.array2string(
            data_for_str,
            separator=', ',
            precision=8,
            suppress_small=False,
            floatmode='maxprec_equal'
        )
        return f"tensor({data_str}, requires_grad={self.requires_grad}, device='{self.device}')"

    def requires_grad_(self, requires_grad: bool = True):
        self.requires_grad = requires_grad
        return self

    def _ensure_same_device(self, other: Union[int, float, 'Tensor']) -> 'Tensor':
        """Ensure other tensor is on the same device as self."""
        if isinstance(other, Tensor) and other.device != self.device:
            return other.to(self.device)
        return other
    
    def __mul__(self, other: Union[int, float, 'Tensor']) -> 'Tensor':
        other = self._ensure_same_device(other)
        other_data = other if isinstance(other, (int, float)) else other.data
        return Tensor(
            self.data * other_data,
            requires_grad=self.requires_grad or (isinstance(other, Tensor) and other.requires_grad),
            grad_fn=Function(dv.mul_backward, [self, other]),
            device=self.device
        )
    
    def __rmul__(self, other: Union[int, float]) -> 'Tensor':
        return Tensor(
            other * self.data,
            requires_grad=self.requires_grad,
            grad_fn=Function(dv.mul_backward, [self, other]),
            device=self.device
        )

    def __add__(self, other: Union[int, float, 'Tensor']) -> 'Tensor':
        other = self._ensure_same_device(other)
        other_data = other if isinstance(other, (int, float)) else other.data
        return Tensor(
            self.data + other_data,
            requires_grad=self.requires_grad or (isinstance(other, Tensor) and other.requires_grad),
            grad_fn=Function(dv.add_backward, [self, other]),
            device=self.device
        )

    def __sub__(self, other: Union[int, float, 'Tensor']) -> 'Tensor':
        other = self._ensure_same_device(other)
        other_data = other if isinstance(other, (int, float)) else other.data
        return Tensor(
            self.data - other_data,
            requires_grad=self.requires_grad or (isinstance(other, Tensor) and other.requires_grad),
            grad_fn=Function(dv.sub_backward, [self, other]),
            device=self.device
        )

    def __neg__(self) -> 'Tensor':
        return Tensor(
            -self.data,
            requires_grad=self.requires_grad,
            grad_fn=Function(dv.neg_backward, [self]),
            device=self.device
        )

    def __truediv__(self, other: Union[int, float, 'Tensor']) -> 'Tensor':
        other = self._ensure_same_device(other)
        other_data = other if isinstance(other, (int, float)) else other.data
        return Tensor(
            self.data / other_data,
            requires_grad=self.requires_grad or (isinstance(other, Tensor) and other.requires_grad),
            grad_fn=Function(dv.truediv_backward, [self, other]),
            device=self.device
        )

    def __rtruediv__(self, other: Union[int, float]) -> 'Tensor':
        return Tensor(
            other / self.data,
            requires_grad=self.requires_grad,
            grad_fn=Function(dv.truediv_backward, [other, self]),
            device=self.device
        )

    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        other = self._ensure_same_device(other)
        return Tensor(
            self.data @ other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            grad_fn=Function(dv.matmul_backward, [self, other]),
            device=self.device
        )

    def __rmatmul__(self, other: 'Tensor') -> 'Tensor':
        return self.__matmul__(other)


    @property
    def T(self) -> 'Tensor':
        return Tensor(
            self.data.T,
            requires_grad=self.requires_grad,
            device=self.device
        )

    def size(self):
        return self.data.shape

    def sum(self, dim: Optional[int] = None) -> 'Tensor':
        xp = get_array_module(self.device)
        result = xp.sum(self.data, axis=dim)
        if not hasattr(result, 'shape') or result.shape == ():
            # Scalar result, convert to array
            result = xp.array(result)
        return Tensor(
            result,
            requires_grad=self.requires_grad,
            grad_fn=Function(dv.sum_backward, [self, dim]),
            device=self.device
        )

    def mean(self, dim: Optional[int] = None) -> 'Tensor':
        xp = get_array_module(self.device)
        result = xp.mean(self.data, axis=dim)
        if not hasattr(result, 'shape') or result.shape == ():
            # Scalar result, convert to array
            result = xp.array(result)
        return Tensor(
            result,
            requires_grad=self.requires_grad,
            grad_fn=Function(dv.mean_backward, [self, dim]),
            device=self.device
        )
    
    def item(self):
        """
        Returns the value of this tensor as a standard Python number.
        This only works for tensors with one element.
        """
        if self.device == 'cuda':
            # Convert to numpy first, then get item
            return float(asnumpy(self.data).item())
        else:
            return float(self.data.item())

    @staticmethod
    def randn(*args, device: Optional[str] = None, **kwargs):
        device = get_device(device)
        xp = get_array_module(device)
        return Tensor(xp.random.randn(*args, **kwargs), requires_grad=False, device=device)

    def backward(self, grad_outputs: Optional[Tensor] = None):
        if grad_outputs is None:
            xp = get_array_module(self.device)
            grad_outputs = Tensor(xp.ones_like(self.data), device=self.device)
        else:
            # Ensure grad_outputs is on the same device
            grad_outputs = self._ensure_same_device(grad_outputs)
        if self.requires_grad and self.grad_fn is not None:
            self.grad_fn.backward([grad_outputs])
    

def tensor(data, requires_grad=False, device=None):
    """
    Create a tensor from data.
    
    Args:
        data: Array-like data to convert to tensor
        requires_grad: Whether the tensor requires gradient computation
        device: 'cpu' or 'cuda' (defaults to 'cuda' if available, else 'cpu')
    
    Returns:
        Tensor instance
    """
    return Tensor(data, requires_grad=requires_grad, device=device)