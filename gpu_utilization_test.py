#!/usr/bin/env python
# full_gpu_training_profiled.py - With PyTorch profiling to diagnose performance issues

import os
import time
import argparse
import threading
import subprocess
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np

# Add profiling imports
from torch.profiler import profile, record_function, ProfilerActivity

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

# ------------------------- Fixed Batch Implementation -------------------------

def fixed_batch_training(model, train_loader, val_loader=None, 
                      num_epochs=5, device='cuda', batch_size=None,
                      use_amp=False, use_channels_last=False, 
                      iterations_per_epoch=100):
    """
    A training loop that uses a single fixed batch for the entire training.
    This approach can achieve very high GPU utilization.
    
    Args:
        model: PyTorch model
        train_loader: DataLoader for training data
        val_loader: Optional DataLoader for validation
        num_epochs: Number of epochs to train
        device: Device to train on
        batch_size: Batch size (for information only)
        use_amp: Whether to use automatic mixed precision
        use_channels_last: Whether to use channels_last memory format
        iterations_per_epoch: Number of iterations per epoch
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
    
    # Enable cuDNN benchmarking
    if 'cuda' in str(device):
        torch.backends.cudnn.benchmark = True
        
    # Set up AMP scaler if using mixed precision
    scaler = torch.amp.GradScaler() if use_amp and torch.cuda.is_available() else None
    
    # Performance tracking
    batch_times = deque(maxlen=20)
    compute_times = deque(maxlen=20)
    
    # Preload a fixed batch
    print("Loading a fixed batch for training...")
    fixed_inputs = None
    fixed_targets = None
    
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
    print(f"  Fixed Batch Mode: True")
    print(f"  Iterations per Epoch: {iterations_per_epoch}")
    
    # Start GPU monitoring
    gpu_monitor = GPUMonitor() if 'cuda' in str(device) else None
    if gpu_monitor:
        gpu_monitor.start()
    
    # Setup profiler
    profile_enabled = True  # Set to True to enable profiling
    
    # Training loop
    print("\nStarting training...")
    
    try:
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # Training phase
            model.train()
            train_loss = 0.0
            correct = 0
            total = 0
            
            # Use the progress meter for clean output
            print(f"\nEpoch {epoch+1}/{num_epochs}:")
            
            # Run profiler for the first 10 iterations of the first epoch
            if epoch == 0 and profile_enabled:
                print("Profiling the first 10 iterations...")
                with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=True
                ) as prof:
                    for batch_idx in range(10):  # Profile first 10 iterations
                        with record_function(f"iteration_{batch_idx}"):
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
                            
                # Print profiling results
                print("\nProfiling Results (Top 10 time-consuming operations):")
                print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
                
                # Export trace if needed
                # prof.export_chrome_trace("fixed_batch_trace.json")
            
            # Continue with regular training (without profiling)
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
            
            # End-of-epoch training stats
            train_loss = train_loss / iterations_per_epoch
            train_acc = 100. * correct / total
            
            # Validation phase - we can still validate on the fixed batch
            val_loss = 0.0
            val_acc = 0.0
            
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

# ------------------------- Full GPU Dataset Trainer -------------------------

class FullGPUTrainer:
    """
    Trainer that loads the entire dataset to GPU memory once and processes it in batches.
    This implementation includes profiling to diagnose performance issues.
    """
    def __init__(self, model, device='cuda', use_amp=False, use_channels_last=False):
        """
        Initialize the trainer with a model and training parameters.
        
        Args:
            model: PyTorch model to train
            device: Device to use ('cuda' or 'cpu')
            use_amp: Whether to use automatic mixed precision
            use_channels_last: Whether to use channels_last memory format
        """
        self.device = device
        self.use_amp = use_amp and torch.cuda.is_available()
        self.use_channels_last = use_channels_last
        
        # Set up the model
        if use_channels_last and 'cuda' in str(device):
            self.model = model.to(device, memory_format=torch.channels_last)
            print("Using channels_last memory format")
        else:
            self.model = model.to(device)
            
        # Set up loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Count model parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model Parameters: {trainable_params:,} trainable, {total_params:,} total")
        
        # Set up AMP scaler if using mixed precision
        self.scaler = torch.amp.GradScaler() if self.use_amp else None
        
        # Performance tracking
        self.batch_times = deque(maxlen=20)
        self.compute_times = deque(maxlen=20)
        
        # Configure cuDNN for better performance
        if 'cuda' in str(device):
            torch.backends.cudnn.benchmark = True
            
        # GPU monitor
        self.gpu_monitor = GPUMonitor(device_id=0) if 'cuda' in str(device) else None
    
    def load_full_dataset(self, train_loader, apply_transforms=True):
        """
        Load the entire dataset into GPU memory.
        
        Args:
            train_loader: DataLoader containing the training data
            apply_transforms: Whether to apply transformations now
            
        Returns:
            Tuple of (all_data, all_targets) tensors in GPU memory
        """
        print("Loading the entire dataset to GPU memory...")
        all_data = []
        all_targets = []
        
        # Use a with statement to handle CUDA memory better
        with torch.no_grad():
            for inputs, targets in train_loader:
                # If apply_transforms is False, assumes the loader already applied them
                if self.use_channels_last:
                    inputs = inputs.to(self.device, memory_format=torch.channels_last, non_blocking=True)
                else:
                    inputs = inputs.to(self.device, non_blocking=True)
                    
                targets = targets.to(self.device, non_blocking=True)
                
                all_data.append(inputs)
                all_targets.append(targets)
        
        # Concatenate all batches into single tensors
        all_data = torch.cat(all_data)
        all_targets = torch.cat(all_targets)
        
        print(f"Dataset loaded to GPU. Shape: {all_data.shape}, Labels: {all_targets.shape}")
        
        # Report memory usage
        if 'cuda' in str(self.device):
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            print(f"GPU Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
            
        return all_data, all_targets
    
    def create_optimizer(self, learning_rate=0.001, weight_decay=0):
        """Create optimizer for the model"""
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=learning_rate,
            weight_decay=weight_decay
        )
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'min', patience=2, factor=0.5
        )
        
    def profile_batch_processing(self, all_data, all_targets, batch_size, num_batches=10):
        """
        Profile batch processing to identify performance bottlenecks.
        
        Args:
            all_data: Tensor containing all training inputs
            all_targets: Tensor containing all training targets
            batch_size: Size of mini-batches to use
            num_batches: Number of batches to profile
            
        Returns:
            Profiler object with results
        """
        print(f"\nProfiling batch processing for {num_batches} batches...")
        
        # Get the total number of samples and create indices
        num_samples = all_data.size(0)
        indices = torch.randperm(num_samples).to(self.device)
        
        self.model.train()
        
        # Run profiler
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof:
            for batch_idx in range(num_batches):
                with record_function(f"batch_{batch_idx}"):
                    # Get batch indices
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, num_samples)
                    batch_indices = indices[start_idx:end_idx]
                    
                    # Get data slices
                    with record_function("data_slicing"):
                        inputs = all_data[batch_indices]
                        targets = all_targets[batch_indices]
                    
                    # Forward pass
                    self.optimizer.zero_grad()
                    
                    with record_function("forward_pass"):
                        if self.use_amp:
                            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                                outputs = self.model(inputs)
                                loss = self.criterion(outputs, targets)
                        else:
                            outputs = self.model(inputs)
                            loss = self.criterion(outputs, targets)
                    
                    # Backward pass
                    with record_function("backward_pass"):
                        if self.use_amp:
                            self.scaler.scale(loss).backward()
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            loss.backward()
                            self.optimizer.step()
        
        # Print profiling results
        print("\nProfiling Results (Top 10 time-consuming operations):")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        
        # You can uncomment this to save trace for Chrome tracing
        # prof.export_chrome_trace("full_gpu_trace.json")
        
        return prof
        
    def train_epoch(self, all_data, all_targets, batch_size, shuffle=True, profile_first_epoch=True):
        """
        Train for one epoch on pre-loaded data in GPU memory.
        
        Args:
            all_data: Tensor containing all training inputs
            all_targets: Tensor containing all training targets
            batch_size: Size of mini-batches to use
            shuffle: Whether to shuffle the data indices
            profile_first_epoch: Whether to profile the first epoch
            
        Returns:
            Dictionary of training statistics
        """
        self.model.train()
        
        # Get the total number of samples
        num_samples = all_data.size(0)
        
        # Generate indices and shuffle if needed
        indices = torch.randperm(num_samples).to(self.device) if shuffle else torch.arange(num_samples).to(self.device)
        
        # Calculate total number of batches
        total_batches = (num_samples + batch_size - 1) // batch_size  # Ceiling division
        
        # Initialize statistics
        train_loss = 0.0
        correct = 0
        total = 0
        
        # Start training loop
        for batch_idx in range(total_batches):
            batch_start = time.time()
            
            # Get batch indices - handles last batch which might be smaller
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            batch_indices = indices[start_idx:end_idx]
            
            # Get data and targets for this batch by indexing
            inputs = all_data[batch_indices]
            targets = all_targets[batch_indices]
            
            # Forward and backward pass
            compute_start = time.time()
            self.optimizer.zero_grad()
            
            if self.use_amp:
                # Use automatic mixed precision
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                
                # Scale gradients and update
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard precision training
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
            
            compute_end = time.time()
            self.compute_times.append(compute_end - compute_start)
            
            # Update statistics
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Calculate timing
            batch_end = time.time()
            self.batch_times.append(batch_end - batch_start)
            
            # Print progress periodically
            if batch_idx % max(1, total_batches // 10) == 0 or batch_idx == total_batches - 1:
                avg_batch_time = sum(self.batch_times) / len(self.batch_times) if self.batch_times else 0
                avg_compute_time = sum(self.compute_times) / len(self.compute_times) if self.compute_times else 0
                
                percent_complete = 100. * (batch_idx + 1) / total_batches
                current_loss = train_loss / (batch_idx + 1)
                current_acc = 100. * correct / total
                
                print(f"  Train: [{batch_idx+1}/{total_batches} ({percent_complete:.1f}%)] "
                      f"Loss: {current_loss:.4f}, Acc: {current_acc:.2f}%, "
                      f"Batch: {avg_batch_time:.4f}s (Compute: {avg_compute_time:.4f}s)")
                
                # Check memory usage (optional)
                if 'cuda' in str(self.device) and (batch_idx % 50 == 0):
                    allocated = torch.cuda.memory_allocated() / 1e9
                    reserved = torch.cuda.memory_reserved() / 1e9
                    print(f"    Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
        
        # Calculate overall statistics
        train_loss = train_loss / total_batches
        train_acc = 100. * correct / total
        
        return {
            'loss': train_loss,
            'accuracy': train_acc,
            'avg_batch_time': sum(self.batch_times) / max(1, len(self.batch_times)),
            'avg_compute_time': sum(self.compute_times) / max(1, len(self.compute_times))
        }
    
    def validate(self, val_data, val_targets, batch_size):
        """
        Validate on pre-loaded validation data in GPU memory.
        
        Args:
            val_data: Tensor containing all validation inputs
            val_targets: Tensor containing all validation targets
            batch_size: Size of mini-batches to use
            
        Returns:
            Dictionary of validation statistics
        """
        self.model.eval()
        
        # Get the total number of validation samples
        num_samples = val_data.size(0)
        
        # Calculate total number of batches
        total_batches = (num_samples + batch_size - 1) // batch_size  # Ceiling division
        
        # Initialize statistics
        val_loss = 0.0
        correct = 0
        total = 0
        
        # Validation loop
        with torch.no_grad():
            for batch_idx in range(total_batches):
                # Get batch indices - handles last batch which might be smaller
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, num_samples)
                
                # Get data and targets for this batch by simple slicing
                inputs = val_data[start_idx:end_idx]
                targets = val_targets[start_idx:end_idx]
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # Update statistics
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        # Calculate overall statistics
        val_loss = val_loss / total_batches
        val_acc = 100. * correct / total
        
        # Update scheduler based on validation loss
        self.scheduler.step(val_loss)
        
        return {
            'loss': val_loss,
            'accuracy': val_acc
        }
    
    def train(self, all_data, all_targets, val_data=None, val_targets=None, 
              batch_size=128, num_epochs=5, shuffle=True):
        """
        Train the model for multiple epochs using pre-loaded data.
        
        Args:
            all_data: Tensor containing all training inputs
            all_targets: Tensor containing all training targets
            val_data: Optional tensor containing all validation inputs
            val_targets: Optional tensor containing all validation targets
            batch_size: Size of mini-batches to use
            num_epochs: Number of epochs to train
            shuffle: Whether to shuffle the data each epoch
            
        Returns:
            Dictionary of training statistics
        """
        # Create optimizer if not already created
        if not hasattr(self, 'optimizer'):
            self.create_optimizer()
        
        print("\nStarting training...")
        print(f"Configuration: batch_size={batch_size}, num_epochs={num_epochs}")
        print(f"Mixed Precision: {self.use_amp}, Memory Format: {'channels_last' if self.use_channels_last else 'channels_first'}")
        
        # Profile batch processing first
        if 'cuda' in str(self.device):
            self.profile_batch_processing(all_data, all_targets, batch_size, num_batches=50)
        
        # Start GPU monitoring
        if self.gpu_monitor:
            self.gpu_monitor.start()
        
        try:
            for epoch in range(num_epochs):
                epoch_start_time = time.time()
                print(f"\nEpoch {epoch+1}/{num_epochs}:")
                
                # Train for one epoch
                train_stats = self.train_epoch(all_data, all_targets, batch_size, shuffle)
                
                # Validate if validation data is provided
                val_stats = None
                if val_data is not None and val_targets is not None:
                    val_stats = self.validate(val_data, val_targets, batch_size)
                
                # Print epoch summary
                epoch_time = time.time() - epoch_start_time
                print(f"\nEpoch {epoch+1} Summary:")
                print(f"  Train Loss: {train_stats['loss']:.4f}, Train Acc: {train_stats['accuracy']:.2f}%")
                
                if val_stats:
                    print(f"  Val Loss: {val_stats['loss']:.4f}, Val Acc: {val_stats['accuracy']:.2f}%")
                    
                print(f"  Time: {epoch_time:.2f}s, LR: {self.optimizer.param_groups[0]['lr']:.6f}")
                print(f"  Avg Batch Time: {train_stats['avg_batch_time']:.4f}s")
                print(f"  Avg Compute Time: {train_stats['avg_compute_time']:.4f}s")
                
                # Print GPU stats if available
                if self.gpu_monitor:
                    stats = self.gpu_monitor.get_stats()
                    if stats:
                        print(f"  GPU Utilization: {stats['util_avg']:.1f}% avg, {stats['util_min']:.1f}% min, {stats['util_max']:.1f}% max")
                        print(f"  Recent GPU Util: {stats['util_recent_avg']:.1f}% avg (last 20 samples)")
                        
                        
            # Final results
            if self.gpu_monitor:
                stats = self.gpu_monitor.stop()
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
            if self.gpu_monitor:
                self.gpu_monitor.stop()
                
        print("\nTraining complete!")
        return self.model

# ------------------------- Dataset Loading Utilities -------------------------

def get_dataset(dataset_name, batch_size=64):
    """Loads a dataset and returns train/test loaders"""
    
    # Use large batch size for initial loading, but not so large that it causes OOM errors
    initial_load_batch_size = min(10000, batch_size)
    
    # Define the transforms - apply these once during loading
    if dataset_name.lower() == 'cifar10':
        # Data transformations (same as your original script)
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
        # CIFAR-100 transformations
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
        
        train_dataset = torchvision.datasets.CIFAR100(
            root='./data', train=True, download=True, transform=transform_train)
        
        test_dataset = torchvision.datasets.CIFAR100(
            root='./data', train=False, download=True, transform=transform_test)
        
        num_classes = 100
        channels = 3
        input_size = 32
        
    elif dataset_name.lower() == 'mnist':
        # MNIST transformations
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_dataset = torchvision.datasets.MNIST(
            root='./data', train=True, download=True, transform=transform)
        
        test_dataset = torchvision.datasets.MNIST(
            root='./data', train=False, download=True, transform=transform)
        
        num_classes = 10
        channels = 1
        input_size = 28
        
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # Create data loaders - using as few workers as possible since we're loading everything at once
    train_loader = DataLoader(
        train_dataset, 
        batch_size=initial_load_batch_size,
        shuffle=False,  # No need to shuffle during loading as we'll shuffle in GPU memory
        num_workers=1,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=initial_load_batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )
    
    return train_loader, test_loader, num_classes, channels, input_size

# ------------------------- Main Function -------------------------

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='PyTorch Training Performance Comparison')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'mnist'],
                        help='Dataset to use (cifar10, cifar100, mnist)')
    parser.add_argument('--model', type=str, default='simple', choices=['simple', 'large'],
                        help='Model architecture to use (simple, large)')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for processing')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of epochs to train')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disable CUDA training')
    parser.add_argument('--amp', action='store_true', default=False,
                        help='Use automatic mixed precision')
    parser.add_argument('--channels-last', action='store_true', default=False,
                        help='Use channels_last memory format')
    parser.add_argument('--method', type=str, default='both', choices=['fixed', 'fullgpu', 'both'],
                        help='Training method to use: fixed batch, full GPU, or both for comparison')
    parser.add_argument('--iterations-per-epoch', type=int, default=100,
                        help='Number of iterations per epoch for fixed batch training')
    
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
        batch_size=args.batch_size
    )
    
    # Check if doing fixed batch training
    if args.method in ['fixed', 'both']:
        print("\n" + "="*80)
        print("FIXED BATCH TRAINING")
        print("="*80)
        
        # Create model for fixed batch training
        print(f"Creating {args.model} MLP model...")
        if args.model == 'simple':
            model_fixed = SimpleMLP(num_classes=num_classes, channels=channels, input_size=input_size)
        else:
            model_fixed = LargeMLP(num_classes=num_classes, channels=channels, input_size=input_size)
        
        # Train with fixed batch
        fixed_batch_training(
            model_fixed,
            train_loader,
            test_loader,
            num_epochs=args.epochs,
            device=device,
            batch_size=args.batch_size,
            use_amp=args.amp,
            use_channels_last=args.channels_last,
            iterations_per_epoch=args.iterations_per_epoch
        )
    
    # Check if doing full GPU training
    if args.method in ['fullgpu', 'both']:
        print("\n" + "="*80)
        print("FULL GPU DATASET TRAINING")
        print("="*80)
        
        # Create model for full GPU training
        print(f"Creating {args.model} MLP model...")
        if args.model == 'simple':
            model_full = SimpleMLP(num_classes=num_classes, channels=channels, input_size=input_size)
        else:
            model_full = LargeMLP(num_classes=num_classes, channels=channels, input_size=input_size)
        
        # Create trainer
        trainer = FullGPUTrainer(
            model=model_full,
            device=device,
            use_amp=args.amp,
            use_channels_last=args.channels_last
        )
        
        # Load training data to GPU
        print("\nLoading training data to GPU memory...")
        train_data, train_targets = trainer.load_full_dataset(train_loader)
        
        # Load validation data if provided
        print("\nLoading validation data to GPU memory...")
        val_data, val_targets = trainer.load_full_dataset(test_loader)
        
        # Start training
        print("\nStarting training with fully GPU-loaded dataset...")
        trainer.create_optimizer(learning_rate=0.001)
        trained_model = trainer.train(
            train_data, 
            train_targets,
            val_data,
            val_targets,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            shuffle=True
        )
    
    print("\nAll training complete!")

if __name__ == "__main__":
    main()