import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import my_torch

def execute_train(args):
    """Test Conv2d layer on a simple generated image."""
    
    # Create a simple test image
    print("Creating test image...")
    img_size = 64
    img = create_simple_test_image(img_size)
    
    # Convert to tensor format: (batch_size, channels, height, width)
    # From (H, W, C) to (1, C, H, W)
    x = my_torch.Tensor(img.transpose(2, 0, 1)[np.newaxis, ...])
    
    print(f"Image shape: {img.shape} (height, width, channels)")
    print(f"Tensor shape: {x.data.shape} (batch, channels, height, width)")
    
    # Create Conv2d layer
    print("\nCreating Conv2d layer...")
    conv = my_torch.Conv2d(
        in_channels=3, 
        out_channels=16, 
        kernel_size=(3, 3), 
        stride=(2, 2), 
        padding=(1, 1)
    )
    
    # Apply convolution
    print("Applying convolution...")
    y = conv(x)
    
    # Print shapes
    print("\n" + "="*50)
    print("SHAPE INFORMATION:")
    print("="*50)
    print(f"Input shape: {x.data.shape}")
    print(f"Output shape: {y.data.shape}")
    print(f"Weight shape: {conv.weight.data.shape}")
    if conv.bias is not None:
        print(f"Bias shape: {conv.bias.data.shape}")
    
    # Calculate expected output shape
    batch_size, in_channels, height, width = x.data.shape
    kh, kw = conv.kernel_size
    sh, sw = conv.stride
    ph, pw = conv.padding
    
    expected_h = (height + 2*ph - kh) // sh + 1
    expected_w = (width + 2*pw - kw) // sw + 1
    
    print(f"\nExpected output shape: (batch, 16, {expected_h}, {expected_w})")
    print(f"Actual output shape: {y.data.shape}")
    
    # Compare with PyTorch
    print("\n" + "="*50)
    print("COMPARISON WITH PYTORCH:")
    print("="*50)
    
    conv_real = torch.nn.Conv2d(
        in_channels=3, 
        out_channels=16, 
        kernel_size=3, 
        stride=2, 
        padding=1,
        bias=True
    )
    
    # Use same weights for fair comparison
    with torch.no_grad():
        conv_real.weight.copy_(torch.tensor(conv.weight.data))
        if conv.bias is not None:
            conv_real.bias.copy_(torch.tensor(conv.bias.data))
    
    x_real = torch.tensor(x.data)
    y_real = conv_real(x_real)
    
    print(f"PyTorch input shape: {x_real.shape}")
    print(f"PyTorch output shape: {y_real.shape}")
    print(f"PyTorch weight shape: {conv_real.weight.shape}")
    
    # Visualize results
    print("\n" + "="*50)
    print("VISUALIZING RESULTS:")
    print("="*50)
    
    visualize_conv_results(img, x.data, y.data)
    
    return conv, x, y

def create_simple_test_image(size=64):
    """Create a simple test image with patterns."""
    # Create RGB image
    img = np.zeros((size, size, 3), dtype=np.float32)
    
    # Fill with gradient
    for i in range(size):
        for j in range(size):
            img[i, j, 0] = i / size  # Red vertical gradient
            img[i, j, 1] = j / size  # Green horizontal gradient
            img[i, j, 2] = (i + j) / (2 * size)  # Blue diagonal gradient
    
    # Add a white square in the center
    center = size // 2
    square_size = size // 4
    img[center-square_size:center+square_size, center-square_size:center+square_size, :] = 1.0
    
    # Add a colored circle
    for i in range(size):
        for j in range(size):
            # Circle in top-left
            if (i - size//4)**2 + (j - size//4)**2 < (size//8)**2:
                img[i, j, 0] = 1.0  # Red
                img[i, j, 1] = 0.0
                img[i, j, 2] = 0.0
            
            # Circle in bottom-right  
            if (i - 3*size//4)**2 + (j - 3*size//4)**2 < (size//8)**2:
                img[i, j, 0] = 0.0
                img[i, j, 1] = 1.0  # Green
                img[i, j, 2] = 0.0
    
    # Add some text pattern (simplified)
    # Horizontal and vertical lines
    img[size//8, :, :] = 1.0  # Horizontal line
    img[:, size//8, :] = 1.0  # Vertical line
    img[7*size//8, :, :] = 1.0  # Horizontal line
    img[:, 7*size//8, :] = 1.0  # Vertical line
    
    return img

def visualize_conv_results(original_img, input_tensor, output_tensor):
    """Visualize input and output of convolution."""
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Original image
    ax1 = plt.subplot(2, 4, 1)
    ax1.imshow(original_img)
    ax1.set_title("Original Image (RGB)")
    ax1.axis('off')
    
    # 2. Input tensor channels
    ax2 = plt.subplot(2, 4, 2)
    ax2.imshow(input_tensor[0, 0], cmap='Reds', vmin=0, vmax=1)
    ax2.set_title("Input - Red Channel")
    ax2.axis('off')
    
    ax3 = plt.subplot(2, 4, 3)
    ax3.imshow(input_tensor[0, 1], cmap='Greens', vmin=0, vmax=1)
    ax3.set_title("Input - Green Channel")
    ax3.axis('off')
    
    ax4 = plt.subplot(2, 4, 4)
    ax4.imshow(input_tensor[0, 2], cmap='Blues', vmin=0, vmax=1)
    ax4.set_title("Input - Blue Channel")
    ax4.axis('off')
    
    # 3. Output tensor - first 4 channels
    output_channels = min(4, output_tensor.shape[1])  # Show first 4 channels
    
    for i in range(output_channels):
        ax = plt.subplot(2, 4, 5 + i)
        channel_data = output_tensor[0, i]
        
        # Normalize for display
        if channel_data.max() > channel_data.min():
            channel_display = (channel_data - channel_data.min()) / (channel_data.max() - channel_data.min())
        else:
            channel_display = channel_data
        
        ax.imshow(channel_display, cmap='viridis')
        ax.set_title(f"Output Channel {i}")
        ax.axis('off')
    
    plt.suptitle("Conv2D Visualization: Input vs Output", fontsize=16, y=0.98)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('conv2d_test.png', dpi=150, bbox_inches='tight')
    print("Visualization saved as 'conv2d_test.png'")
    
    # Print statistics
    print("\nOUTPUT STATISTICS:")
    print("-" * 30)
    for i in range(min(4, output_tensor.shape[1])):
        channel = output_tensor[0, i]
        print(f"Channel {i}: min={channel.min():.4f}, max={channel.max():.4f}, "
              f"mean={channel.mean():.4f}, std={channel.std():.4f}")
    
    # Show the plot
    plt.show()

def create_simple_test_image_alt(size=64):
    """Alternative: Create an even simpler test image."""
    # Create a chessboard pattern
    img = np.zeros((size, size, 3), dtype=np.float32)
    
    # Create 8x8 chessboard
    square_size = size // 8
    for i in range(8):
        for j in range(8):
            if (i + j) % 2 == 0:
                # White squares
                img[i*square_size:(i+1)*square_size, j*square_size:(j+1)*square_size, :] = 1.0
    
    # Add a red diagonal
    for i in range(size):
        if i < size and i < size:
            img[i, i, 0] = 1.0  # Red
            img[i, i, 1] = 0.0
            img[i, i, 2] = 0.0
    
    return img

# Alternative execute_train if you want to test with a downloaded image
def execute_train_with_real_image(args):
    """Test Conv2d with a real image from file or URL."""
    try:
        # Try to load a local image
        image_path = "test_image.jpg"  # Change this to your image path
        
        if os.path.exists(image_path):
            img = Image.open(image_path).convert('RGB')
            img = img.resize((64, 64))
            img_array = np.array(img) / 255.0
            print(f"Loaded image from {image_path}")
        else:
            # Create a test pattern if no image found
            print(f"Image not found at {image_path}, creating test pattern...")
            img_array = create_simple_test_image(64)
            
        # Rest of the function remains the same as execute_train
        # ... [same code as above] ...
        
    except Exception as e:
        print(f"Error loading image: {e}")
        print("Falling back to generated test pattern...")
        return execute_train(args)

if __name__ == "__main__":
    # For testing without the analyzer
    class Args:
        pass
    
    args = Args()
    
    # Import torch for comparison
    try:
        import torch
        execute_train(args)
    except ImportError:
        print("PyTorch not installed, running without comparison...")
        # Create a dummy torch module
        class DummyTorch:
            class nn:
                class Conv2d:
                    def __init__(self, **kwargs):
                        pass
                    def __call__(self, x):
                        return None
        
        torch = DummyTorch()
        execute_train(args)