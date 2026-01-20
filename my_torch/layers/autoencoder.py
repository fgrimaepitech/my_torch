from my_torch.neural_network import Module
import my_torch
from my_torch.tensor import Tensor

class AutoEncoder(Module):
    def __init__(self):
        super().__init__()
        self.encoder = my_torch.Sequential(
            my_torch.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=1),
            my_torch.LeakyReLU(0.01),
            my_torch.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(2, 2), padding=1),
            my_torch.LeakyReLU(0.01),
            my_torch.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(2, 2), padding=1),
            my_torch.LeakyReLU(0.01),
            my_torch.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1),
            my_torch.Flatten(),
            my_torch.Linear(3136, 2)
        )

        self.decoder = my_torch.Sequential(
            my_torch.Linear(2, 3136),
            my_torch.Reshape((-1, 64, 7, 7)),
            my_torch.ConvTranspose2d(64, 64, stride=(1, 1), kernel_size=(3, 3), padding=1),
            my_torch.LeakyReLU(0.01),
            my_torch.ConvTranspose2d(64, 64, stride=(2, 2), kernel_size=(3, 3), padding=1),
            my_torch.LeakyReLU(0.01),
            my_torch.ConvTranspose2d(64, 32, stride=(2, 2), kernel_size=(3, 3), padding=0),
            my_torch.LeakyReLU(0.01),
            my_torch.ConvTranspose2d(32, 1, stride=(1, 1), kernel_size=(3, 3), padding=1, output_padding=1),
            my_torch.Sigmoid()
        )

        self.add_module('encoder', self.encoder)
        self.add_module('decoder', self.decoder)

    def forward(self, x: Tensor) -> Tensor:
        print("x shape in encoder", x.data.shape)
        x = self.encoder(x)
        print("x shape in decoder", x.data.shape)
        x = self.decoder(x)
        print("x shape in the final result", x.data.shape)
        return x

    def train(self, mode: bool = True):
        self.encoder.train(mode)
        self.decoder.train(mode)
        return self
        