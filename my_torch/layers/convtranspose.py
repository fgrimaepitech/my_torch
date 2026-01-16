from my_torch.neural_network import Module
import my_torch
import numpy as np

from my_torch.tensor import Tensor

class ConvTranspose2d(Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = 1, padding: tuple = 0, output_padding: tuple = 0, groups: int = 1, bias: bool = True, dilation: tuple = 1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.output_padding = output_padding if isinstance(output_padding, tuple) else (output_padding, output_padding)
        self.groups = groups
        self.bias = bias
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)

        self.weight = Tensor(
            np.random.randn(in_channels, out_channels, self.kernel_size[0], self.kernel_size[1]),
            requires_grad=True
        )
        if bias:
            self.bias = Tensor(np.zeros(out_channels), requires_grad=True)
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        print(x.data.shape)
        batch_size, in_channels, height, width = x.data.shape
        out_height = (height - 1) * self.stride[0] - 2 * self.padding[0] + self.dilation[0] * (self.kernel_size[0] - 1) + self.output_padding[0] + 1
        out_width = (width - 1) * self.stride[1] - 2 * self.padding[1] + self.dilation[1] * (self.kernel_size[1] - 1) + self.output_padding[1] + 1

        output = np.zeros((batch_size, self.out_channels, out_height, out_width))

        for b in range(batch_size):
            for oc in range(self.out_channels):
                channel_output = np.zeros((out_height, out_width))
                for ic  in range (self.in_channels):
                    weight = self.weight.data[ic, oc]
                    input_channel = x.data[b, ic]

                    if self.stride[0] > 1 or self.stride[1] > 1:
                        dilated_h = height + (height - 1) * (self.dilation[0] - 1)
                        dilated_w = width + (width - 1) * (self.dilation[1] - 1)
                        dilated_input = np.zeros((dilated_h, dilated_w))

                        for i in range(height):
                            for j in range(width):
                                dilated_input[i * self.dilation[0], j * self.dilation[1]] = input_channel[i, j]
                                
                        input_to_use = dilated_input
                    else:
                        input_to_use = input_channel

                    flipped_weight = np.flip(weight)

                    pad_h = self.kernel_size[0] - 1 - self.padding[0]
                    pad_w = self.kernel_size[1] - 1 - self.padding[1]

                    if pad_h > 0 or pad_w > 0:
                        padded_input = np.pad(
                            input_to_use, 
                            ((pad_h, pad_h), (pad_w, pad_w)),
                            mode='constant'
                        )
                    else:
                        padded_input = input_to_use

                    conv_h = padded_input.shape[0] - flipped_weight.shape[0] + 1
                    conv_w = padded_input.shape[1] - flipped_weight.shape[1] + 1

                    for i in range(conv_h):
                        for j in range(conv_w):
                            window = padded_input[i:i+flipped_weight.shape[0], j:j+flipped_weight.shape[1]]
                            channel_output[i, j] += np.sum(window * flipped_weight)
                

                output[b, oc] += channel_output

        if self.bias is not None:
            for oc in range(self.out_channels):
                output[:, oc, :, :] += self.bias.data[oc]

        return Tensor(output, requires_grad=x.requires_grad)

    def parameters(self):
        params = [self.weight]
        if self.bias is not None:
            params.append(self.bias)
        return params
