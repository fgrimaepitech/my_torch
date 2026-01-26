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
        batch_size, in_channels, height, width = x.data.shape

        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding
        oph, opw = self.output_padding
        dh, dw = self.dilation

        if (dh, dw) != (1, 1):
            raise NotImplementedError("ConvTranspose2d dilation != 1 is not supported yet")

        # Expected output size (matches PyTorch formula for dilation=1)
        out_height = (height - 1) * sh - 2 * ph + (kh - 1) + oph + 1
        out_width = (width - 1) * sw - 2 * pw + (kw - 1) + opw + 1

        # 1) Upsample input by stride (insert zeros)
        up_h = (height - 1) * sh + 1
        up_w = (width - 1) * sw + 1
        upsampled = np.zeros((batch_size, in_channels, up_h, up_w), dtype=x.data.dtype)
        upsampled[:, :, ::sh, ::sw] = x.data

        # 2) Pad so that a standard conv with flipped kernel matches transposed conv geometry
        pad_h = kh - 1 - ph
        pad_w = kw - 1 - pw
        padded = np.pad(
            upsampled,
            ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)),
            mode='constant'
        )

        # 3) Convolve (valid) with flipped kernel
        flipped_weight = np.flip(self.weight.data, axis=(2, 3))
        conv_h = padded.shape[2] - kh + 1
        conv_w = padded.shape[3] - kw + 1
        output = np.zeros((batch_size, self.out_channels, conv_h, conv_w), dtype=x.data.dtype)

        for b in range(batch_size):
            for oc in range(self.out_channels):
                channel_sum = np.zeros((conv_h, conv_w), dtype=x.data.dtype)
                for ic in range(in_channels):
                    k = flipped_weight[ic, oc]
                    for i in range(conv_h):
                        for j in range(conv_w):
                            patch = padded[b, ic, i:i+kh, j:j+kw]  # always (kh, kw)
                            channel_sum[i, j] += np.sum(patch * k)
                output[b, oc] = channel_sum

        # 4) Apply output_padding by padding on the bottom/right (PyTorch semantics)
        if oph or opw:
            output = np.pad(output, ((0, 0), (0, 0), (0, oph), (0, opw)), mode='constant')

        # 5) Crop/pad defensively to the computed out_height/out_width (in case of edge mismatches)
        output = output[:, :, :out_height, :out_width]

        if self.bias is not None:
            output += self.bias.data.reshape(1, -1, 1, 1)

        return Tensor(output, requires_grad=x.requires_grad)

    def parameters(self):
        params = [self.weight]
        if self.bias is not None:
            params.append(self.bias)
        return params
