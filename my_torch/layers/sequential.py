from my_torch.neural_network import Module
import my_torch

class Sequential(Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = layers

    def forward(self, x: my_torch.Tensor) -> my_torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [layer.parameters() for layer in self.layers]