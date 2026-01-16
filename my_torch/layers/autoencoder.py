from my_torch.functionnal import reshape
from my_torch.neural_network import Module
import my_torch

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
            reshape((-1, 64, 7, 7)),
            

        )
        