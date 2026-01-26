import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import my_torch
import torch

# Create a tensor
x = my_torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)

# Get item with gradient tracking
y = x[0, 1]  # Returns tensor(2.0)
print(y)  # tensor(2.0, requires_grad=True)

# Get scalar with .item()
if y.data.size == 1:
    scalar = y.item()  # Returns 2.0 as Python float
    print(scalar)  # 2.0

# Set item (raises error if requires_grad=True)
x_copy = x.copy()  # Make a copy first
x_copy[0, 1] = 5.0  # This works if requires_grad=False

# Safer way with gradient tracking
x_modified = x[0, 1] = 5.0
# x_modified is a new tensor with the modification

print(x)
print(x_modified)