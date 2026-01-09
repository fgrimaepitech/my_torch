from my_torch import tensor
from my_torch.neural_network import Module
import numpy as np

class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.weight = tensor(np.random.randn(out_features, in_features) * 0.01, requires_grad=True)
        if bias:
            self.bias = tensor(np.zeros(out_features), requires_grad=True)
        else:
            self.bias = None

    def forward(self, x):
        return x @ self.weight.T + self.bias

    def parameters(self):
        return [self.weight, self.bias]

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"

    def reset_parameters(self):
        self.weight = tensor(np.random.randn(self.out_features, self.in_features) * 0.01, requires_grad=True)
        self.bias = tensor(np.zeros(self.out_features), requires_grad=True)