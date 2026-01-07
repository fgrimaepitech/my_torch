"""
Tests comparing my_torch operators with PyTorch operators.
Tests mul, rmul, and add operations.
"""
import unittest
import numpy as np
import torch
import sys
import os

# Add parent directory to path to import my_torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import my_torch


class TestOperators(unittest.TestCase):
    """Test operators: mul, rmul, and add"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.rtol = 1e-5  # Relative tolerance for floating point comparison
        self.atol = 1e-8  # Absolute tolerance for floating point comparison
    
    def assert_tensors_close(self, my_tensor, torch_tensor, msg=""):
        """Helper to compare my_torch tensor with PyTorch tensor"""
        # Convert PyTorch tensor to numpy if needed
        if isinstance(torch_tensor, torch.Tensor):
            torch_data = torch_tensor.detach().numpy()
        else:
            torch_data = np.array(torch_tensor)
        
        # Compare data
        np.testing.assert_allclose(
            my_tensor.data, 
            torch_data, 
            rtol=self.rtol, 
            atol=self.atol,
            err_msg=f"Tensor data mismatch. {msg}"
        )
    
    # ========== MUL TESTS ==========
    
    def test_mul_tensor_tensor_1d(self):
        """Test tensor * tensor for 1D tensors"""
        # my_torch
        a = my_torch.tensor([1.0, 2.0, 3.0])
        b = my_torch.tensor([4.0, 5.0, 6.0])
        result = a * b
        
        # PyTorch
        a_torch = torch.tensor([1.0, 2.0, 3.0])
        b_torch = torch.tensor([4.0, 5.0, 6.0])
        result_torch = a_torch * b_torch
        
        self.assert_tensors_close(result, result_torch, "1D tensor * tensor")
    
    def test_mul_tensor_tensor_2d(self):
        """Test tensor * tensor for 2D tensors"""
        # my_torch
        a = my_torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        b = my_torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        result = a * b
        
        # PyTorch
        a_torch = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        b_torch = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        result_torch = a_torch * b_torch
        
        self.assert_tensors_close(result, result_torch, "2D tensor * tensor")
    
    def test_mul_tensor_scalar(self):
        """Test tensor * scalar"""
        # my_torch
        a = my_torch.tensor([1.0, 2.0, 3.0])
        result = a * 2.5
        
        # PyTorch
        a_torch = torch.tensor([1.0, 2.0, 3.0])
        result_torch = a_torch * 2.5
        
        self.assert_tensors_close(result, result_torch, "tensor * scalar")
    
    def test_mul_tensor_int(self):
        """Test tensor * integer"""
        # my_torch
        a = my_torch.tensor([1.0, 2.0, 3.0])
        result = a * 3
        
        # PyTorch
        a_torch = torch.tensor([1.0, 2.0, 3.0])
        result_torch = a_torch * 3
        
        self.assert_tensors_close(result, result_torch, "tensor * int")
    
    def test_mul_tensor_zero(self):
        """Test tensor * 0"""
        # my_torch
        a = my_torch.tensor([1.0, 2.0, 3.0])
        result = a * 0
        
        # PyTorch
        a_torch = torch.tensor([1.0, 2.0, 3.0])
        result_torch = a_torch * 0
        
        self.assert_tensors_close(result, result_torch, "tensor * 0")
    
    def test_mul_tensor_negative(self):
        """Test tensor * negative scalar"""
        # my_torch
        a = my_torch.tensor([1.0, 2.0, 3.0])
        result = a * -2.0
        
        # PyTorch
        a_torch = torch.tensor([1.0, 2.0, 3.0])
        result_torch = a_torch * -2.0
        
        self.assert_tensors_close(result, result_torch, "tensor * negative")
    
    # ========== RMUL TESTS ==========
    
    def test_rmul_scalar_tensor(self):
        """Test scalar * tensor (right multiplication)"""
        # my_torch
        a = my_torch.tensor([1.0, 2.0, 3.0])
        result = 2.5 * a
        
        # PyTorch
        a_torch = torch.tensor([1.0, 2.0, 3.0])
        result_torch = 2.5 * a_torch
        
        self.assert_tensors_close(result, result_torch, "scalar * tensor")
    
    def test_rmul_int_tensor(self):
        """Test integer * tensor"""
        # my_torch
        a = my_torch.tensor([1.0, 2.0, 3.0])
        result = 3 * a
        
        # PyTorch
        a_torch = torch.tensor([1.0, 2.0, 3.0])
        result_torch = 3 * a_torch
        
        self.assert_tensors_close(result, result_torch, "int * tensor")
    
    def test_rmul_zero_tensor(self):
        """Test 0 * tensor"""
        # my_torch
        a = my_torch.tensor([1.0, 2.0, 3.0])
        result = 0 * a
        
        # PyTorch
        a_torch = torch.tensor([1.0, 2.0, 3.0])
        result_torch = 0 * a_torch
        
        self.assert_tensors_close(result, result_torch, "0 * tensor")
    
    def test_rmul_negative_tensor(self):
        """Test negative scalar * tensor"""
        # my_torch
        a = my_torch.tensor([1.0, 2.0, 3.0])
        result = -2.0 * a
        
        # PyTorch
        a_torch = torch.tensor([1.0, 2.0, 3.0])
        result_torch = -2.0 * a_torch
        
        self.assert_tensors_close(result, result_torch, "negative * tensor")
    
    def test_rmul_tensor_2d(self):
        """Test scalar * 2D tensor"""
        # my_torch
        a = my_torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = 2.0 * a
        
        # PyTorch
        a_torch = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        result_torch = 2.0 * a_torch
        
        self.assert_tensors_close(result, result_torch, "scalar * 2D tensor")
    
    # ========== ADD TESTS ==========
    
    def test_add_tensor_tensor_1d(self):
        """Test tensor + tensor for 1D tensors"""
        # my_torch
        a = my_torch.tensor([1.0, 2.0, 3.0])
        b = my_torch.tensor([4.0, 5.0, 6.0])
        result = a + b
        
        # PyTorch
        a_torch = torch.tensor([1.0, 2.0, 3.0])
        b_torch = torch.tensor([4.0, 5.0, 6.0])
        result_torch = a_torch + b_torch
        
        self.assert_tensors_close(result, result_torch, "1D tensor + tensor")
    
    def test_add_tensor_tensor_2d(self):
        """Test tensor + tensor for 2D tensors"""
        # my_torch
        a = my_torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        b = my_torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        result = a + b
        
        # PyTorch
        a_torch = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        b_torch = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        result_torch = a_torch + b_torch
        
        self.assert_tensors_close(result, result_torch, "2D tensor + tensor")
    
    def test_add_tensor_scalar(self):
        """Test tensor + scalar"""
        # my_torch
        a = my_torch.tensor([1.0, 2.0, 3.0])
        result = a + 2.5
        
        # PyTorch
        a_torch = torch.tensor([1.0, 2.0, 3.0])
        result_torch = a_torch + 2.5
        
        self.assert_tensors_close(result, result_torch, "tensor + scalar")
    
    def test_add_tensor_int(self):
        """Test tensor + integer"""
        # my_torch
        a = my_torch.tensor([1.0, 2.0, 3.0])
        result = a + 3
        
        # PyTorch
        a_torch = torch.tensor([1.0, 2.0, 3.0])
        result_torch = a_torch + 3
        
        self.assert_tensors_close(result, result_torch, "tensor + int")
    
    def test_add_tensor_zero(self):
        """Test tensor + 0"""
        # my_torch
        a = my_torch.tensor([1.0, 2.0, 3.0])
        result = a + 0
        
        # PyTorch
        a_torch = torch.tensor([1.0, 2.0, 3.0])
        result_torch = a_torch + 0
        
        self.assert_tensors_close(result, result_torch, "tensor + 0")
    
    def test_add_tensor_negative(self):
        """Test tensor + negative scalar"""
        # my_torch
        a = my_torch.tensor([1.0, 2.0, 3.0])
        result = a + (-2.0)
        
        # PyTorch
        a_torch = torch.tensor([1.0, 2.0, 3.0])
        result_torch = a_torch + (-2.0)
        
        self.assert_tensors_close(result, result_torch, "tensor + negative")
    
    def test_add_float_values(self):
        """Test addition with float values"""
        # my_torch
        a = my_torch.tensor([0.1, 0.2, 0.3])
        b = my_torch.tensor([0.4, 0.5, 0.6])
        result = a + b
        
        # PyTorch
        a_torch = torch.tensor([0.1, 0.2, 0.3])
        b_torch = torch.tensor([0.4, 0.5, 0.6])
        result_torch = a_torch + b_torch
        
        self.assert_tensors_close(result, result_torch, "float addition")
    
    # ========== COMBINED OPERATIONS ==========
    
    def test_combined_operations(self):
        """Test combined operations: (a * b) + c"""
        # my_torch
        a = my_torch.tensor([1.0, 2.0, 3.0])
        b = my_torch.tensor([2.0, 3.0, 4.0])
        c = my_torch.tensor([1.0, 1.0, 1.0])
        result = (a * b) + c
        
        # PyTorch
        a_torch = torch.tensor([1.0, 2.0, 3.0])
        b_torch = torch.tensor([2.0, 3.0, 4.0])
        c_torch = torch.tensor([1.0, 1.0, 1.0])
        result_torch = (a_torch * b_torch) + c_torch
        
        self.assert_tensors_close(result, result_torch, "combined operations")


if __name__ == '__main__':
    unittest.main()

