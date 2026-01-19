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
    """
    Super simple Adam test.
    """
    print("Testing Adam Optimizer")
    
    # Create parameter
    w = my_torch.tensor([1.0, 2.0], requires_grad=True)
    w_torch = torch.tensor([1.0, 2.0], requires_grad=True)
    
    # Option A: Pass as list of tensors (if you fix Adam)
    # adam = my_torch.Adam([w], lr=0.1)
    
    # Option B: Pass as parameter group (works with current code)
    adam = my_torch.Adam([{'params': [w], 'lr': 0.1}])
    adam_torch = torch.optim.Adam([w_torch], lr=0.1)
    
    print(f"Initial w: {w.data}")
    print(f"Initial w_torch: {w_torch.data}")
    # Set fake gradient
    w.grad = my_torch.tensor([0.1, 0.2])
    w_torch.grad = torch.tensor([0.1, 0.2])
    print(f"Gradient: {w.grad}")
    print(f"Gradient w_torch: {w_torch.grad}")

    # Take step
    adam.step()
    adam_torch.step()
    
    print(f"After step: {w.data}")
    print(f"After step w_torch: {w_torch.data}")
    print(f"Adam: {adam.state_dict()}")
    print(f"Adam Torch: {adam_torch.state_dict()}")
    return adam


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