from my_torch.neural_network import Module
from my_torch.tensor import Tensor
import numpy as np
from my_torch.device import get_array_module

class BatchNorm1d(Module):
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        self.gamma = Tensor(np.zeros(num_features), requires_grad=True)
        self.beta = Tensor(np.zeros(num_features), requires_grad=True)

        self.running_mean = None  # Will be initialized on first forward pass with device
        self.running_var = None

        self.training = True

    def forward(self, x: Tensor) -> Tensor:
        xp = get_array_module(x.device)
        
        # Initialize running stats on first forward pass
        if self.running_mean is None:
            self.running_mean = xp.zeros(self.num_features, dtype=x.data.dtype)
            self.running_var = xp.ones(self.num_features, dtype=x.data.dtype)
        elif hasattr(self.running_mean, 'device') and self.running_mean.device != x.device:
            # Move running stats to same device as input
            self.running_mean = xp.asarray(self.running_mean)
            self.running_var = xp.asarray(self.running_var)
        
        if self.training:
            if len(x.data.shape) == 2:
                mean = x.data.mean(axis=0)
                var = x.data.var(axis=0)
            else:
                mean = x.data.mean(axis=(0, 2))
                var = x.data.var(axis=(0, 2))

            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
            
            x_normalized = (x.data - mean) / xp.sqrt(var + self.eps)
        
        else:
            x_normalized = (x.data - self.running_mean) / xp.sqrt(self.running_var + self.eps)

        if len(x.data.shape) == 2:
            result = x_normalized * self.gamma.data + self.beta.data
        else:
            gamma_reshaped = self.gamma.data.reshape(1, -1, 1)
            beta_reshaped = self.beta.data.reshape(1, -1, 1)
            result = x_normalized * gamma_reshaped + beta_reshaped

        return Tensor(result, requires_grad=x.requires_grad, device=x.device)

    def eval(self):
        self.training = False

    def train(self):
        self.training = True

    def parameters(self):
        return [self.gamma, self.beta]

class BatchNorm2d(Module):
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1, device: str = 'cuda'):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        self.gamma = Tensor(np.ones(num_features), requires_grad=True)
        self.beta = Tensor(np.zeros(num_features), requires_grad=True)

        self.running_mean = None  # Will be initialized on first forward pass with device
        self.running_var = None

        self.training = True

    def forward(self, x: Tensor) -> Tensor:
        xp = get_array_module(x.device)
        if len(x.data.shape) != 4:
            raise ValueError("BatchNorm2d only supports 4D input tensors")

        batch_size, channels, height, width = x.data.shape
        
        # Initialize running stats on first forward pass
        if self.running_mean is None:
            self.running_mean = xp.zeros(self.num_features, dtype=x.data.dtype)
            self.running_var = xp.ones(self.num_features, dtype=x.data.dtype)
        elif hasattr(self.running_mean, 'device') and self.running_mean.device != x.device:
            # Move running stats to same device as input
            self.running_mean = xp.asarray(self.running_mean)
            self.running_var = xp.asarray(self.running_var)

        if self.training:
            mean = x.data.mean(axis=(0, 2, 3))
            var = x.data.var(axis=(0, 2, 3))

            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var

            # reshape for broadcasting
            mean_reshaped = mean.reshape(1, channels, 1, 1)
            var_reshaped = var.reshape(1, channels, 1, 1)
            
        else:
            mean_reshaped = self.running_mean.reshape(1, channels, 1, 1)
            var_reshaped = self.running_var.reshape(1, channels, 1, 1)

        x_normalized = (x.data - mean_reshaped) / xp.sqrt(var_reshaped + self.eps)

        gamma_reshaped = self.gamma.data.reshape(1, channels, 1, 1)
        beta_reshaped = self.beta.data.reshape(1, channels, 1, 1)

        result = x_normalized * gamma_reshaped + beta_reshaped
        
        return Tensor(result, requires_grad=x.requires_grad, device=x.device)

    def eval(self):
        self.training = False

    def train(self):
        self.training = True

    def parameters(self):
        return [self.gamma, self.beta]
            