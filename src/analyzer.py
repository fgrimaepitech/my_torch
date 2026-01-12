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
    conv = my_torch.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    x = my_torch.Tensor(np.random.randn(4, 3, 32, 32))
    y = conv(x)
    print(f"Input shape: {x.data.shape}")
    print(f"Output shape: {y.data.shape}")
    print(f"Weight shape: {conv.weight.data.shape}")
    conv_real = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    x_real = torch.randn(4, 3, 32, 32)
    y_real = conv_real(x_real)
    print(f"Input shape: {x_real.shape}")
    print(f"Output shape: {y_real.shape}")
    print(f"Weight shape: {conv_real.weight.data.shape}")

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