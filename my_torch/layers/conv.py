from my_torch.function import Function
from my_torch.neural_network import Module
from my_torch.tensor import Tensor
import numpy as np
import my_torch.functionnal as F
import my_torch.derivatives as dv

class ConvNd(Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        std = np.sqrt(2.0 / (in_channels * kernel_size))
        self.weight = Tensor(np.random.randn(out_channels, in_channels, kernel_size) * std, requires_grad=True)
        if bias:
            self.bias = Tensor(np.zeros(out_channels), requires_grad=True)
        else:
            self.bias = None


    def forward(self) -> Tensor:
        pass

    def parameters(self):
        params = [self.weight]
        if self.bias is not None:
            params.append(self.bias)
        return params


class Conv1d(ConvNd):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = True):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, bias)

    def forward(self, input: Tensor) -> Tensor:
        self._input = input
        self._input_padded = self._pad1d(input, self.padding) if self.padding > 0 else input

        batch_size, in_channels, length = self._input_padded.data.shape
        out_length = ((length - self.kernel_size) // self.stride) + 1

        x_2d = self._input_padded.data.reshape(-1, length)

        windows = []
        for i in range(x_2d.shape[0]):
            row_windows = F.as_strided(Tensor(x_2d[i]), shape=(out_length, self.kernel_size), strides=(self.stride, 1)).data
            windows.append(row_windows)

        windows_stacked = np.stack(windows, axis=0)

        windows_reshaped = windows_stacked.reshape(
            batch_size, in_channels, out_length, self.kernel_size).transpose(0, 2, 1, 3).reshape(batch_size, out_length, -1
        )

        weight_reshaped = self.weight.data.reshape(self.out_channels, -1)

        result = np.einsum('bij,oj->bio', windows_reshaped, weight_reshaped)
        result = result.transpose(0, 2, 1)

        if self.bias is not None:
            result += self.bias.data.reshape(1, -1, 1)

        return Tensor(result, requires_grad=input.requires_grad, grad_fn=Function(dv.conv1d_backward, [self._input, self.weight, self.bias], stride=self.stride, padding=self.padding))

    def _pad1d(self, input: Tensor, padding: int) -> Tensor:
        batch_size, channels, length = input.data.shape
        padded_length = length + 2 * padding
        padded_data = np.zeros((batch_size, channels, padded_length))
        padded_data[:, :, padding:padding+length] = input.data
        return Tensor(padded_data, requires_grad=input.requires_grad)

    def _unfold1d(self, input: Tensor, kernel_size: int, stride: int, out_length: int) -> Tensor:
        batch_size, channels, length = input.data.shape
        unfolded_data = np.zeros((batch_size, out_length, channels, kernel_size))
        for b in range(batch_size):
            for c in range(channels):
                channel_data = input.data[b, c, :]

                window = F.as_strided(Tensor(channel_data), shape=(out_length, kernel_size), strides=(stride, 1))

                unfolded_data[b, :, c, :] = window.data

        return Tensor(unfolded_data, requires_grad=input.requires_grad)

    def parameters(self):
        params = [self.weight]
        if self.bias is not None:
            params.append(self.bias)
        return params