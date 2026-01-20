from my_torch.neural_network import Module
import my_torch

class Sequential(Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = layers
        # Register layers as submodules so that base Module.parameters() works
        for idx, layer in enumerate(layers):
            self.add_module(str(idx), layer)

    def forward(self, x: my_torch.Tensor) -> my_torch.Tensor:
        for idx, layer in enumerate(self.layers):
            print(f"[Sequential] Before layer {idx} ({layer.__class__.__name__})")
            x = layer(x)
            print(f"[Sequential] After  layer {idx} ({layer.__class__.__name__})")
        return x