import os
import sys
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import my_torch
import numpy as np

def execute_train(args):
    tensor1 = my_torch.tensor([[1.1, 2.2], [0.2, 0.4], [4.4, 1.2]], requires_grad=True)
    tensor2 = my_torch.tensor([[1.4, 2.8], [0.4, 0.9], [4.3, 1.1]])
    tensor = tensor1 + tensor2
    print(tensor)
    tensor.backward()
    print(tensor1.grad)
    print(tensor2.grad)

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