import os
import sys
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import my_torch
import numpy as np
import torch

def execute_train(args):
    # ConvTranspose2d test - NEED 4D INPUT!
    # Wrong: (2, 3, 4) - 3D tensor
    # Correct: (batch_size, in_channels, height, width)
    
    # Create 4D input: (batch_size=2, channels=3, height=4, width=4)
    input = my_torch.tensor(np.random.randn(2, 3, 4, 4))  # ← 4D!
    
    print(f"Input shape: {input.data.shape}")
    print(f"Expected: (2, 3, 4, 4) - (batch, channels, height, width)")
    
    # Create ConvTranspose2d layer
    conv_transpose = my_torch.ConvTranspose2d(
        in_channels=3,      # Should match input channels
        out_channels=2,     # Output channels
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1)
    )
    
    # Forward pass
    output = conv_transpose(input)
    print(f"\nOutput shape: {output.data.shape}")
    
    # Calculate expected output shape
    # Formula: H_out = (H_in - 1) * stride - 2*padding + dilation*(kernel_size-1) + output_padding + 1
    # With stride=1, padding=1, kernel=3, output_padding=0, dilation=1:
    # H_out = (4 - 1)*1 - 2*1 + 1*(3-1) + 0 + 1 = 3 - 2 + 2 + 1 = 4
    print(f"Expected output shape: (2, 2, 4, 4)")
    
    # Compare with PyTorch
    import torch
    import torch.nn as nn
    
    torch_conv_transpose = nn.ConvTranspose2d(
        in_channels=3,
        out_channels=2,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=True
    )
    
    # Copy weights for fair comparison
    # PyTorch weight shape: (in_channels, out_channels, kernel_h, kernel_w)
    # Your weight shape: (in_channels, out_channels, kernel_h, kernel_w) - same!
    with torch.no_grad():
        torch_conv_transpose.weight.copy_(
            torch.tensor(conv_transpose.weight.data)
        )
        torch_conv_transpose.bias.copy_(
            torch.tensor(conv_transpose.bias.data)
        )
    
    torch_input = torch.tensor(input.data, dtype=torch.float32)
    torch_output = torch_conv_transpose(torch_input)
    
    print(f"\nPyTorch output shape: {torch_output.shape}")
    
    # Compare
    diff = np.abs(output.data - torch_output.detach().numpy())
    print(f"\nComparison:")
    print(f"Max difference: {diff.max():.6f}")
    print(f"Mean difference: {diff.mean():.6f}")
    
    return output

def main():
    parser = argparse.ArgumentParser(prog="my_torch_analyzer")
    exec_mode = parser.add_mutually_exclusive_group(required=True)
    exec_mode.add_argument(
        '--train',
        action='store_true',
        help="Launch the neural network in training mode. Each chessboard in CHESSFILE must contain inputs to send to the neural network in FEN notation and the expected output separated by space."
    )
    exec_mode.add_argument(
        '--predict',
        action='store_true',
        help="Launch the neural network in prediction mode. Each chessboard in CHESSFILE must contain inputs to send to the neural network in FEN notation, and optionally an expected output."
    )

    parser.add_argument(
        '--save',
        metavar='SAVEFILE',
        help="Only valid with --train. If specified, the newly trained neural network will be saved in SAVEFILE; otherwise, it will be saved back into LOADFILE."
    )
    parser.add_argument("LOADFILE", help="File containing an artificial neural network")
    parser.add_argument("CHESSFILE", help="File containing chessboards")

    args = parser.parse_args()

    if args.save and not args.train:
        parser.error("--save can only be used together with --train")

    execute_train(args)

if __name__ == "__main__":
    main()