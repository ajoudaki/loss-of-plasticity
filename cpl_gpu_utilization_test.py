#!/usr/bin/env python
# gpu_utilization_test.py - Test and measure GPU utilization with PyTorch

import os
import time
import argparse
import subprocess
import threading
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np

# ------------------------- GPU Monitoring Utilities -------------------------

class GPUMonitor:
    """Monitors GPU utilization in a separate thread"""
    def __init__(self, device_id=0, interval=0.2):
        self.device_id = device_id
        self.interval = interval
        self.utilization = []
        self.memory_used = []
        self.stop_flag = False
        self.monitor_thread = None
    
    def start(self):
        """Start GPU monitoring in background thread"""
        self.stop_flag = False
        self.utilization = []
        self.memory_used = []
        self.monitor_thread = threading.Thread(target=self._monitor_gpu)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop(self):
        """Stop GPU monitoring"""
        if self.monitor_thread:
            self.stop_flag = True
            self.monitor_thread.join(timeout=1.0)
            return self.get_stats()
        return None
    
    def _monitor_gpu(self):
        """Thread function that periodically checks GPU stats"""
        while not self.stop_flag:
            try:
                # Get GPU utilization using nvidia-smi
                result = subprocess.run(
                    ['nvidia-smi', 
                     f'--id={self.device_id}', 
                     '--query-gpu=utilization.gpu,memory.used',
                     '--format=csv,noheader,nounits'],
                    stdout=subprocess.PIPE, 
                    text=True,
                    check=True)
                
                util, mem = map(float, result.stdout.strip().split(','))
                self.utilization.append(util)
                self.memory_used.append(mem)
            except Exception as e:
                print(f"Error monitoring GPU: {e}")
            
            time.sleep(self.interval)
    
    def get_stats(self):
        """Return statistics about GPU utilization"""
        if not self.utilization:
            return None
        
        return {
            'util_min': min(self.utilization),
            'util_max': max(self.utilization),
            'util_avg': sum(self.utilization) / len(self.utilization),
            'util_recent_avg': sum(self.utilization[-20:]) / min(len(self.utilization), 20),
            'util_values': self.utilization,
            'memory_min': min(self.memory_used),
            'memory_max': max(self.memory_used),
            'memory_avg': sum(self.memory_used) / len(self.memory_used)
        }

# ------------------------- Model Definitions -------------------------

class SimpleMLP(nn.Module):
    """A simple MLP (Multi-Layer Perceptron) for image classification"""
    def __init__(self, num_classes=10, channels=3, input_size=32):
        super(SimpleMLP, self).__init__()
        input_dim = channels * input_size * input_size  # Flatten the image
        
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.layers(x)
        return x

class LargeMLP(nn.Module):
    """A larger MLP (Multi-Layer Perceptron) for image classification"""
    def __init__(self, num_classes=10, channels=3, input_size=32):
        super(LargeMLP, self).__init__()
        input_dim = channels * input_size * input_size  # Flatten the image
        
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, num_classes)
        )
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.layers(x)
        return x

# ------------------------- Optimized Training Loop -------------------------

def optimized_training_loop(model, train_loader, val_loader=None, 
                          num_epochs=5, device='cuda', batch_size=None,
                          use_amp=False, use_channels_last=False, 
                          fixed_batch=False, iterations_per_epoch=None):
    """
    An optimized training loop designed to maximize GPU utilization.
    
    Args:
        model: PyTorch model
        train_loader: DataLoader for training data
        val_loader: Optional DataLoader for validation
        num_epochs: Number of epochs to train
        device: Device to train on
        batch_size: Batch size (for information only)
        use_amp: Whether to use automatic mixed precision
        use_channels_last: Whether to use channels_last memory format
        fixed_batch: Whether to use a single fixed batch for training
        iterations_per_epoch: Number of iterations per epoch when using fixed batch
    """
    # Move model to device and set memory format if needed
    if use_channels_last and 'cuda' in str(device):
        model = model.to(device, memory_format=torch.channels_last)
        print("Using channels_last memory format")
    else:
        model = model.to(device)
    
    # Count model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters: {trainable_params:,} trainable, {total_params:,} total")
    
    # Set up loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Set up learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
    
    # Enable cuDNN benchmarking and deterministic algorithms
    if 'cuda' in str(device):
        torch.backends.cudnn.benchmark = True
        
    # Set up AMP scaler if using mixed precision
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp) if use_amp and torch.cuda.is_available() else None
    
    # Preload a fixed batch if specified
    fixed_inputs = None
    fixed_targets = None
    
    if fixed_batch:
        print("Loading a fixed batch for training...")
        for inputs, targets in train_loader:
            # Convert to channels_last if needed
            if use_channels_last:
                inputs = inputs.to(device, memory_format=torch.channels_last, non_blocking=True)
            else:
                inputs = inputs.to(device, non_blocking=True)
            
            targets = targets.to(device, non_blocking=True)
            
            fixed_inputs = inputs
            fixed_targets = targets
            print(f"Fixed batch loaded: inputs shape {fixed_inputs.shape}, targets shape {fixed_targets.shape}")
            break  # Only load the first batch
    
    # Print configuration 
    print(f"\nTraining Configuration:")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Device: {device}")
    print(f"  Mixed Precision: {use_amp}")
    print(f"  Memory Format: {'channels_last' if use_channels_last else 'channels_first'}")
    print(f"  Fixed Batch Mode: {fixed_batch}")
    if fixed_batch:
        print(f"  Iterations per Epoch: {iterations_per_epoch}")
    if 'cuda' in str(device):
        print(f"  CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"  Memory Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"  Memory Reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
    
    # Start GPU monitoring
    gpu_monitor = GPUMonitor() if 'cuda' in str(device) else None
    if gpu_monitor:
        gpu_monitor.start()
    
    # Training loop
    print("\nStarting training...")
    total_batches = iterations_per_epoch if fixed_batch else len(train_loader)
    
    try:
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # Training phase
            model.train()
            train_loss = 0.0
            correct = 0
            total = 0
            batch_times = deque(maxlen=20)
            compute_times = deque(maxlen=20)
            
            # Use the progress meter for clean output
            print(f"\nEpoch {epoch+1}/{num_epochs}:")
            
            # For fixed batch mode, we iterate a fixed number of times
            if fixed_batch:
                for batch_idx in range(iterations_per_epoch):
                    batch_start = time.time()
                    
                    # Forward and backward pass on the fixed batch
                    compute_start = time.time()
                    optimizer.zero_grad()
                    
                    if use_amp:
                        # Use automatic mixed precision
                        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                            outputs = model(fixed_inputs)
                            loss = criterion(outputs, fixed_targets)
                        
                        # Scale gradients and update
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        # Standard precision training
                        outputs = model(fixed_inputs)
                        loss = criterion(outputs, fixed_targets)
                        loss.backward()
                        optimizer.step()
                    
                    compute_end = time.time()
                    compute_times.append(compute_end - compute_start)
                    
                    # Update statistics
                    train_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += fixed_targets.size(0)
                    correct += predicted.eq(fixed_targets).sum().item()
                    
                    # Calculate timing
                    batch_end = time.time()
                    batch_times.append(batch_end - batch_start)
                    
                    # Print progress periodically
                    if batch_idx % max(1, iterations_per_epoch // 10) == 0 or batch_idx == iterations_per_epoch - 1:
                        avg_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0
                        avg_compute_time = sum(compute_times) / len(compute_times) if compute_times else 0
                        
                        percent_complete = 100. * (batch_idx + 1) / iterations_per_epoch
                        current_loss = train_loss / (batch_idx + 1)
                        current_acc = 100. * correct / total
                        
                        print(f"  Train: [{batch_idx+1}/{iterations_per_epoch} ({percent_complete:.1f}%)] "
                              f"Loss: {current_loss:.4f}, Acc: {current_acc:.2f}%, "
                              f"Batch: {avg_batch_time:.4f}s (Compute: {avg_compute_time:.4f}s)")
                        
                        # Check memory usage (optional)
                        if 'cuda' in str(device) and (batch_idx % 50 == 0):
                            print(f"    Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated, "
                                  f"{torch.cuda.memory_reserved() / 1e9:.2f} GB reserved")
            
            # Standard training loop when not using fixed batch
            else:
                # Loop through training batches
                for batch_idx, (inputs, targets) in enumerate(train_loader):
                    batch_start = time.time()
                    
                    # Move data to device efficiently with non-blocking transfer
                    if use_channels_last:
                        inputs = inputs.to(device, non_blocking=True, memory_format=torch.channels_last)
                    else:
                        inputs = inputs.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                    
                    # Forward and backward pass
                    compute_start = time.time()
                    optimizer.zero_grad()
                    
                    if use_amp:
                        # Use automatic mixed precision
                        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                            outputs = model(inputs)
                            loss = criterion(outputs, targets)
                        
                        # Scale gradients and update
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        # Standard precision training
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        loss.backward()
                        optimizer.step()
                    
                    compute_end = time.time()
                    compute_times.append(compute_end - compute_start)
                    
                    # Update statistics
                    train_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
                    
                    # Calculate timing
                    batch_end = time.time()
                    batch_times.append(batch_end - batch_start)
                    
                    # Print progress periodically
                    if batch_idx % max(1, total_batches // 10) == 0 or batch_idx == total_batches - 1:
                        avg_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0
                        avg_compute_time = sum(compute_times) / len(compute_times) if compute_times else 0
                        
                        percent_complete = 100. * (batch_idx + 1) / total_batches
                        current_loss = train_loss / (batch_idx + 1)
                        current_acc = 100. * correct / total
                        
                        print(f"  Train: [{batch_idx+1}/{total_batches} ({percent_complete:.1f}%)] "
                              f"Loss: {current_loss:.4f}, Acc: {current_acc:.2f}%, "
                              f"Batch: {avg_batch_time:.4f}s (Compute: {avg_compute_time:.4f}s)")
                        
                        # Check memory usage (optional)
                        if 'cuda' in str(device) and (batch_idx % 50 == 0):
                            print(f"    Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated, "
                                  f"{torch.cuda.memory_reserved() / 1e9:.2f} GB reserved")
            
            # End-of-epoch training stats
            train_loss = train_loss / (iterations_per_epoch if fixed_batch else len(train_loader))
            train_acc = 100. * correct / total
            
            # Validation phase - skip if using fixed batch
            val_loss = 0.0
            val_acc = 0.0
            
            if val_loader and not fixed_batch:
                model.eval()
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for inputs, targets in val_loader:
                        inputs, targets = inputs.to(device), targets.to(device)
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        
                        val_loss += loss.item()
                        _, predicted = outputs.max(1)
                        val_total += targets.size(0)
                        val_correct += predicted.eq(targets).sum().item()
                
                val_loss = val_loss / len(val_loader)
                val_acc = 100. * val_correct / val_total
                
                # Update scheduler based on validation loss
                scheduler.step(val_loss)
            elif fixed_batch:
                # When using fixed batch, we can still validate on that batch
                model.eval()
                with torch.no_grad():
                    outputs = model(fixed_inputs)
                    loss = criterion(outputs, fixed_targets)
                    val_loss = loss.item()
                    _, predicted = outputs.max(1)
                    val_acc = 100. * predicted.eq(fixed_targets).sum().item() / fixed_targets.size(0)
                
                # Update scheduler based on fixed batch loss
                scheduler.step(val_loss)
            
            # End of epoch time
            epoch_time = time.time() - epoch_start_time
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            if val_loader or fixed_batch:
                print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"  Time: {epoch_time:.2f}s, LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Print GPU stats if available
            if gpu_monitor:
                stats = gpu_monitor.get_stats()
                if stats:
                    print(f"  GPU Utilization: {stats['util_avg']:.1f}% avg, {stats['util_min']:.1f}% min, {stats['util_max']:.1f}% max")
                    print(f"  Recent GPU Util: {stats['util_recent_avg']:.1f}% avg (last 20 samples)")
    
        # Final Results
        if gpu_monitor:
            stats = gpu_monitor.stop()
            if stats:
                print("\nGPU Utilization Statistics:")
                print(f"  Average: {stats['util_avg']:.1f}%")
                print(f"  Min: {stats['util_min']:.1f}%")
                print(f"  Max: {stats['util_max']:.1f}%")
                print(f"  Memory Used: {stats['memory_avg'] / 1024:.2f} GB average")
                
                # Count percentage of time with high utilization
                high_util = sum(1 for u in stats['util_values'] if u > 80)
                percent_high = (high_util / len(stats['util_values'])) * 100
                print(f"  High Utilization (>80%): {percent_high:.1f}% of the time")
                
                # Count percentage of time with low utilization
                low_util = sum(1 for u in stats['util_values'] if u < 20)
                percent_low = (low_util / len(stats['util_values'])) * 100
                print(f"  Low Utilization (<20%): {percent_low:.1f}% of the time")
    
    except KeyboardInterrupt:
        print("\nTraining interrupted!")
    
    finally:
        # Make sure we stop the GPU monitor
        if gpu_monitor:
            gpu_monitor.stop()
    
    print("\nTraining complete!")
    return model

# ------------------------- Dataset Loading Utilities -------------------------

def get_dataset(dataset_name, batch_size=64, num_workers=4, pin_memory=True, prefetch_factor=2):
    """Loads a dataset and returns train/test loaders"""
    
    if dataset_name.lower() == 'cifar10':
        # Data transformations
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        # Load CIFAR-10 dataset
        train_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train)
        
        test_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)
        
        num_classes = 10
        channels = 3
        input_size = 32
        
    elif dataset_name.lower() == 'cifar100':
        # Data transformations
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        
        # Load CIFAR-100 dataset
        train_dataset = torchvision.datasets.CIFAR100(
            root='./data', train=True, download=True, transform=transform_train)
        
        test_dataset = torchvision.datasets.CIFAR100(
            root='./data', train=False, download=True, transform=transform_test)
        
        num_classes = 100
        channels = 3
        input_size = 32
        
    elif dataset_name.lower() == 'mnist':
        # Data transformations
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # Load MNIST dataset
        train_dataset = torchvision.datasets.MNIST(
            root='./data', train=True, download=True, transform=transform)
        
        test_dataset = torchvision.datasets.MNIST(
            root='./data', train=False, download=True, transform=transform)
        
        num_classes = 10
        channels = 1
        input_size = 28
        
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else 2,
        persistent_workers=True if num_workers > 0 else False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else 2,
        persistent_workers=True if num_workers > 0 else False
    )
    
    return train_loader, test_loader, num_classes, channels, input_size

# ------------------------- Main Function -------------------------

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='GPU Utilization Test for PyTorch Training')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'mnist'],
                        help='Dataset to use (cifar10, cifar100, mnist)')
    parser.add_argument('--model', type=str, default='simple', choices=['simple', 'large'],
                        help='Model architecture to use (simple, large)')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of epochs to train')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disable CUDA training')
    parser.add_argument('--amp', action='store_true', default=False,
                        help='Use automatic mixed precision')
    parser.add_argument('--channels-last', action='store_true', default=False,
                        help='Use channels_last memory format')
    parser.add_argument('--pin-memory', action='store_true', default=True,
                        help='Use pin memory for faster GPU transfer')
    parser.add_argument('--no-pin-memory', action='store_false', dest='pin_memory',
                        help='Disable pin memory')
    parser.add_argument('--prefetch-factor', type=int, default=2,
                        help='Number of batches to prefetch (default: 2)')
    # Add new arguments for fixed batch mode
    parser.add_argument('--fixed-batch', action='store_true', default=False,
                        help='Use a single fixed batch for training')
    parser.add_argument('--iterations-per-epoch', type=int, default=100,
                        help='Number of iterations per epoch when using fixed batch')
    
    args = parser.parse_args()
    
    # Set device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    if use_cuda:
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")
    
    # Load dataset
    print(f"Loading {args.dataset} dataset...")
    train_loader, test_loader, num_classes, channels, input_size = get_dataset(
        args.dataset, 
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=args.pin_memory,
        prefetch_factor=args.prefetch_factor
    )
    
    # Create model
    print(f"Creating {args.model} MLP model...")
    if args.model == 'simple':
        model = SimpleMLP(num_classes=num_classes, channels=channels, input_size=input_size)
    else:
        model = LargeMLP(num_classes=num_classes, channels=channels, input_size=input_size)
    
    # Train model with optimized loop
    optimized_training_loop(
        model, 
        train_loader,
        test_loader,
        num_epochs=args.epochs,
        device=device,
        batch_size=args.batch_size,
        use_amp=args.amp,
        use_channels_last=args.channels_last,
        fixed_batch=args.fixed_batch,
        iterations_per_epoch=args.iterations_per_epoch
    )

if __name__ == "__main__":
    main()