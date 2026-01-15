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
    print("=== CORRECT COMPARISON ===")
    
    # 1. Create identical random input
    np.random.seed(42)
    x_data = np.random.randn(2, 4, 8, 8).astype(np.float32)
    
    # 2. Your bottleneck
    my_bottleneck = my_torch.BottleneckBlock(in_channels=4, out_channels=8, stride=1)
    x_my = my_torch.Tensor(x_data)
    y_my = my_bottleneck(x_my)
    
    print(f"MyTorch output shape: {y_my.data.shape}")
    print(f"MyTorch min: {y_my.data.min():.4f}, max: {y_my.data.max():.4f}")
    print(f"MyTorch has negatives: {(y_my.data < 0).any()}")
    
    # 3. PyTorch EXACT match
    import torch.nn as nn
    import torch
    
    class CorrectTorchBottleneck(nn.Module):
        def __init__(self):
            super().__init__()
            # First conv: 4 -> 2 (8//4=2)
            self.conv1 = nn.Conv2d(4, 2, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm2d(2)
            # Second conv: 2 -> 2
            self.conv2 = nn.Conv2d(2, 2, kernel_size=3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(2)
            # Third conv: 2 -> 8
            self.conv3 = nn.Conv2d(2, 8, kernel_size=1, bias=False)
            self.bn3 = nn.BatchNorm2d(8)
            # Skip connection (4 -> 8)
            self.shortcut = nn.Conv2d(4, 8, kernel_size=1, bias=False)
            self.shortcut_bn = nn.BatchNorm2d(8)
            
        def forward(self, x):
            identity = x
            
            out = nn.functional.relu(self.bn1(self.conv1(x)))
            out = nn.functional.relu(self.bn2(self.conv2(out)))
            out = self.bn3(self.conv3(out))
            
            identity = self.shortcut_bn(self.shortcut(identity))
            out += identity
            out = nn.functional.relu(out)
            
            return out
    
    torch_bottleneck = CorrectTorchBottleneck()
    
    # 4. Copy weights from my_torch to PyTorch
    with torch.no_grad():
        # Copy conv1
        torch_bottleneck.conv1.weight.copy_(
            torch.tensor(my_bottleneck.conv1.weight.data)
        )
        # Copy bn1
        torch_bottleneck.bn1.weight.copy_(
            torch.tensor(my_bottleneck.bn1.gamma.data)
        )
        torch_bottleneck.bn1.bias.copy_(
            torch.tensor(my_bottleneck.bn1.beta.data)
        )
        torch_bottleneck.bn1.running_mean.copy_(
            torch.tensor(my_bottleneck.bn1.running_mean)
        )
        torch_bottleneck.bn1.running_var.copy_(
            torch.tensor(my_bottleneck.bn1.running_var)
        )
        
        # Copy conv2
        torch_bottleneck.conv2.weight.copy_(
            torch.tensor(my_bottleneck.conv2.weight.data)
        )
        # Copy bn2
        torch_bottleneck.bn2.weight.copy_(
            torch.tensor(my_bottleneck.bn2.gamma.data)
        )
        torch_bottleneck.bn2.bias.copy_(
            torch.tensor(my_bottleneck.bn2.beta.data)
        )
        torch_bottleneck.bn2.running_mean.copy_(
            torch.tensor(my_bottleneck.bn2.running_mean)
        )
        torch_bottleneck.bn2.running_var.copy_(
            torch.tensor(my_bottleneck.bn2.running_var)
        )
        
        # Copy conv3
        torch_bottleneck.conv3.weight.copy_(
            torch.tensor(my_bottleneck.conv3.weight.data)
        )
        # Copy bn3
        torch_bottleneck.bn3.weight.copy_(
            torch.tensor(my_bottleneck.bn3.gamma.data)
        )
        torch_bottleneck.bn3.bias.copy_(
            torch.tensor(my_bottleneck.bn3.beta.data)
        )
        torch_bottleneck.bn3.running_mean.copy_(
            torch.tensor(my_bottleneck.bn3.running_mean)
        )
        torch_bottleneck.bn3.running_var.copy_(
            torch.tensor(my_bottleneck.bn3.running_var)
        )
        
        # Copy shortcut
        if hasattr(my_bottleneck, 'shortcut') and my_bottleneck.shortcut:
            torch_bottleneck.shortcut.weight.copy_(
                torch.tensor(my_bottleneck.shortcut.weight.data)
            )
            torch_bottleneck.shortcut_bn.weight.copy_(
                torch.tensor(my_bottleneck.shortcut_bn.gamma.data)
            )
            torch_bottleneck.shortcut_bn.bias.copy_(
                torch.tensor(my_bottleneck.shortcut_bn.beta.data)
            )
    
    # 5. Run PyTorch
    x_torch = torch.tensor(x_data)
    y_torch = torch_bottleneck(x_torch)
    
    print(f"\nPyTorch output shape: {y_torch.shape}")
    print(f"PyTorch min: {y_torch.min():.4f}, max: {y_torch.max():.4f}")
    print(f"PyTorch has negatives: {(y_torch.detach().numpy() < 0).any()}")
    
    # 6. Compare
    print(f"\n=== COMPARISON ===")
    y_torch_np = y_torch.detach().numpy()
    
    # Check if both are ≥ 0 (ReLU applied)
    if (y_my.data < 0).any():
        print("❌ MyTorch has negative values (ReLU not working?)")
    else:
        print("✓ MyTorch output ≥ 0 (ReLU working)")
    
    if (y_torch_np < 0).any():
        print("❌ PyTorch has negative values")
    else:
        print("✓ PyTorch output ≥ 0")
    
    # Calculate difference
    diff = np.abs(y_my.data - y_torch_np)
    print(f"\nMax difference: {diff.max():.6f}")
    print(f"Mean difference: {diff.mean():.6f}")
    
    return y_my, y_torch

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