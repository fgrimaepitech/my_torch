import os
import sys
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import my_torch
# Remove these imports or keep only for DataLoader
import torch
# Remove: import torch.nn as nn
# Remove: import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

# Hyperparameters
RANDOM_SEED = 49
LEARNING_RATE = 0.0005
BATCH_SIZE = 8
NUM_EPOCHS = 2
NUM_CLASSES = 10

def get_dataloaders_mnist(batch_size: int, num_workers: int = 0, train_transforms = None, test_transforms = None):
    if train_transforms is None:
        train_transforms = transforms.ToTensor()
    if test_transforms is None:
        test_transforms = transforms.ToTensor()
    train_dataset = datasets.MNIST(root='data',
                                  train=True,
             transform=train_transforms,download=True)
    valid_dataset = datasets.MNIST(root='data',
                                   train=True,
                                   transform=test_transforms)
    test_dataset = datasets.MNIST(root='data',
                                  train=False,
                                  transform=test_transforms)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              shuffle=True)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             shuffle=False)
    return train_loader, valid_dataset, test_loader

def train_autoencoder(num_epochs, model, optimizer,
                     train_loader, loss_fn=None,
                     logging_interval=100,
                     skip_epoch_stats=False,
                     save_model=None):
    log_dict = {'train_loss_per_batch': [],
                'train_loss_per_epoch': []}
    
    # Use custom MSE loss if not provided
    if loss_fn is None:
        # Define a simple MSE loss function for your tensors
        def mse_loss_custom(pred, target):
            diff = pred - target          # Tensor
            sq = diff * diff              # Tensor
            total = sq.sum()              # Tensor (scalar)
            factor = 1.0 / sq.data.size   # Python float
            return total * factor         # Tensor
        loss_fn = mse_loss_custom
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        model.train()
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        
        for batch_idx, (features, _) in enumerate(train_loader):
            print(f"  Processing batch {batch_idx}/{len(train_loader)}...")
            
            # CONVERT PyTorch tensors to your custom tensors
            features_np = features.numpy()  # Convert to numpy
            features_custom = my_torch.Tensor(features_np, requires_grad=True)
            
            # FORWARD AND BACK PROP
            print("    Forward pass...")
            print("Features custom:", features_custom.shape)
            logits = model(features_custom)
            
            print("    Loss calculation...")
            loss = loss_fn(logits, features_custom)
            
            print("    Backward pass...")
            optimizer.zero_grad()
            loss.backward()
            
            print("    Optimizer step...")
            optimizer.step()
            
            # LOGGING
            current_loss = loss.data.item() if hasattr(loss.data, 'item') else float(loss.data)
            log_dict['train_loss_per_batch'].append(current_loss)
            
            if not batch_idx % logging_interval:
                print('Epoch: %03d/%03d | Batch %04d/%04d | Loss: %.4f'
                      % (epoch+1, num_epochs, batch_idx,
                         len(train_loader), current_loss))
            
            # Break early for debugging
            if batch_idx >= 2:  # Process only 3 batches per epoch for testing
                print("  Breaking early for debugging...")
                break
        
        if not skip_epoch_stats:
            model.eval()
            train_loss = compute_epoch_loss_autoencoder_custom(
                model, train_loader, loss_fn)
            print('\n***Epoch: %03d/%03d | Loss: %.3f' % (
                  epoch+1, num_epochs, train_loss))
            log_dict['train_loss_per_epoch'].append(train_loss)
        
        print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
        
        # Break early for debugging
        if epoch >= 1:  # Run only 2 epochs for testing
            print("\nBreaking early after 2 epochs for debugging...")
            break
    
    print('\nTotal Training Time: %.2f min' % ((time.time() - start_time)/60))
    
    if save_model is not None:
        # Implement your own save function in my_torch
        model.save_state_dict(save_model)
    
    return log_dict

def compute_epoch_loss_autoencoder_custom(model, loader, loss_fn):
    """Custom version for your model that converts tensors."""
    total_loss = 0.0
    batch_count = 0
    
    # Just test with first few batches
    max_batches = 3
    
    for batch_idx, (data, _) in enumerate(loader):
        if batch_idx >= max_batches:
            break
            
        # Convert PyTorch tensor to custom tensor
        data_np = data.numpy()
        data_custom = my_torch.Tensor(data_np, requires_grad=False)
        
        output = model(data_custom)
        loss = loss_fn(output, data_custom)
        
        current_loss = loss.data.item() if hasattr(loss.data, 'item') else float(loss.data)
        total_loss += current_loss
        batch_count += 1
    
    return total_loss / batch_count if batch_count > 0 else 0.0

def execute_train(args):
    print("Loading MNIST dataset...")
    train_loader, valid_loader, test_loader = get_dataloaders_mnist(
        batch_size=BATCH_SIZE, num_workers=2)
    
    print("Creating AutoEncoder model...")
    model = my_torch.AutoEncoder()
    
    print(f"Model has {sum(p.data.size for p in model.parameters())} parameters")
    
    # Create custom optimizer
    print("Creating Adam optimizer...")
    optimizer = my_torch.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Create custom MSE loss function
    def mse_loss_custom(pred, target):
        """Simple MSE loss for your tensors."""
        diff = pred - target          # Tensor
        sq = diff * diff              # Tensor
        total = sq.sum()              # Tensor (scalar)
        factor = 1.0 / sq.data.size   # Python float
        return total * factor

    print("Starting training...")
    train_autoencoder(NUM_EPOCHS, model, optimizer, train_loader, 
                     loss_fn=mse_loss_custom,
                     logging_interval=1,  # Log every batch for debugging
                     skip_epoch_stats=False, 
                     save_model=None)

def main():
    parser = argparse.ArgumentParser(prog="my_torch_analyzer")
    exec_mode = parser.add_mutually_exclusive_group(required=True)
    exec_mode.add_argument(
        '--train',
        action='store_true',
        help="Launch the neural network in training mode."
    )
    exec_mode.add_argument(
        '--predict',
        action='store_true',
        help="Launch the neural network in prediction mode."
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