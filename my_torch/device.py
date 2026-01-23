"""
Device management utilities for CPU and GPU operations.
"""
import numpy as np

# Try to import CuPy, fallback to None if not available
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None


def get_array_module(device='cpu'):
    """
    Returns the appropriate array module (numpy or cupy) based on device.
    
    Args:
        device: 'cpu' or 'cuda'
    
    Returns:
        numpy or cupy module
    """
    if device == 'cuda':
        if not CUPY_AVAILABLE:
            raise RuntimeError(
                "CuPy is not available. Please install it with: "
                "pip install cupy-cuda12x (or cupy-cuda11x for CUDA 11.x)"
            )
        return cp
    return np


def asnumpy(array):
    """
    Convert array to numpy array, handling both numpy and cupy arrays.
    
    Args:
        array: numpy or cupy array
    
    Returns:
        numpy array
    """
    if CUPY_AVAILABLE and isinstance(array, cp.ndarray):
        return cp.asnumpy(array)
    return np.asarray(array)


def is_cuda_available():
    """
    Check if CUDA is available.
    
    Returns:
        bool: True if CuPy is installed and CUDA is available
    """
    if not CUPY_AVAILABLE:
        return False
    try:
        # Try to create a small array to verify CUDA is working
        test_array = cp.array([1.0])
        del test_array
        return True
    except Exception:
        return False


def get_device(device=None):
    """
    Get the device string, defaulting to 'cuda' if available, else 'cpu'.
    
    Args:
        device: Optional device string ('cpu' or 'cuda')
    
    Returns:
        str: 'cpu' or 'cuda'
    """
    if device is None:
        return 'cuda' if is_cuda_available() else 'cpu'
    if device == 'cuda' and not is_cuda_available():
        print("Warning: CUDA requested but not available. Falling back to CPU.")
        return 'cpu'
    return device
