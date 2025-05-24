import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
# from torch.cuda.amp import GradScaler, autocast
# torch.amp.autocast
from torch.amp import autocast, GradScaler
from torchvision.models import resnet18, resnet34, resnet50
from torch.utils.data import Dataset, DataLoader

torch.set_float32_matmul_precision('high')


class FastDataLoader:
    """
    A data loader that loads the entire dataset to GPU memory at initialization
    and then efficiently serves batches by slicing tensors.
    
    This loader is much faster than the standard DataLoader when:
    1. The dataset fits in GPU memory
    2. The bottleneck is data transfer from CPU to GPU
    
    Args:
        dataset (Dataset): Dataset to load
        batch_size (int): Batch size for data loading
        shuffle (bool): Whether to shuffle data during iteration
        drop_last (bool): Whether to drop the last incomplete batch
        device (torch.device): Device to load data to (e.g., 'cuda', 'cpu')
        collate_fn (callable, optional): Collate function to apply when loading data
    """
    def __init__(self, dataset, batch_size=1, shuffle=False, drop_last=True, 
                 device=torch.device('cuda'), collate_fn=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.device = device
        self.collate_fn = collate_fn if collate_fn is not None else self._default_collate
        
        # Load all data to device memory at initialization
        self._load_all_data()
        
        # Calculate actual number of batches
        self.num_samples = len(self.all_targets)
        if self.drop_last:
            self.num_batches = self.num_samples // self.batch_size
        else:
            self.num_batches = (self.num_samples + self.batch_size - 1) // self.batch_size
    
    def _default_collate(self, batch):
        """Default collate function similar to PyTorch's default_collate"""
        data = torch.stack([item[0] for item in batch])
        targets = torch.tensor([item[1] for item in batch])
        return data, targets
    
    def _load_all_data(self):
        """Load entire dataset into device memory"""
        print(f"Loading dataset ({len(self.dataset)} samples) to {self.device} memory...")
        start_time = time.time()
        
        # Process all samples through collate function in a single batch
        all_samples = [self.dataset[i] for i in range(len(self.dataset))]
        all_data, all_targets = self.collate_fn(all_samples)
        
        # Move data to device
        self.all_data = all_data.to(self.device, non_blocking=True)
        self.all_targets = all_targets.to(self.device, non_blocking=True)
        
        # Report stats
        load_time = time.time() - start_time
        data_size_mb = self.all_data.element_size() * self.all_data.nelement() / (1024 * 1024)
        print(f"Dataset loaded to {self.device} in {load_time:.2f}s "
              f"({data_size_mb:.1f} MB, {len(self.dataset)} samples)")
        
        if torch.cuda.is_available() and 'cuda' in str(self.device):
            allocated = torch.cuda.memory_allocated() / (1024 * 1024 * 1024)
            reserved = torch.cuda.memory_reserved() / (1024 * 1024 * 1024)
            print(f"GPU Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
    
    def __iter__(self):
        """Return iterator over the dataset"""
        self.current_idx = 0
        
        # Create shuffled indices if needed
        if self.shuffle:
            self.indices = torch.randperm(self.num_samples, device=self.device)
        else:
            self.indices = torch.arange(self.num_samples, device=self.device)
            
        return self
    
    def __next__(self):
        """Get next batch"""
        if self.current_idx >= self.num_batches:
            raise StopIteration
            
        # Get batch indices
        start_idx = self.current_idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.num_samples)
        
        # If drop_last is True and this is an incomplete batch, stop iteration
        if self.drop_last and end_idx - start_idx < self.batch_size:
            raise StopIteration
            
        # Get indices for this batch
        batch_indices = self.indices[start_idx:end_idx]
        
        # Get data slices using indexing
        batch_data = self.all_data[batch_indices]
        batch_targets = self.all_targets[batch_indices]
        
        self.current_idx += 1
        return batch_data, batch_targets
    
    def __len__(self):
        """Return the number of batches"""
        return self.num_batches

# ----------------------- GPU Utilities -----------------------

class GPUStats:
    """Simple utility to track GPU stats during training"""
    def __init__(self):
        self.enabled = torch.cuda.is_available()
        self.reset()
    
    def reset(self):
        """Reset stats"""
        self.start_time = time.time()
        if self.enabled:
            torch.cuda.reset_peak_memory_stats()
            self.start_memory = torch.cuda.memory_allocated()
    
    def get_stats(self):
        """Get current GPU stats"""
        if not self.enabled:
            return {}
        
        current_memory = torch.cuda.memory_allocated()
        peak_memory = torch.cuda.max_memory_allocated()
        time_elapsed = time.time() - self.start_time
        
        return {
            'current_memory_gb': current_memory / 1e9,
            'peak_memory_gb': peak_memory / 1e9,
            'memory_increase_gb': (current_memory - self.start_memory) / 1e9,
            'time_elapsed': time_elapsed
        }
    
    def print_stats(self, prefix=""):
        """Print current GPU stats"""
        if not self.enabled:
            return
            
        stats = self.get_stats()
        print(f"{prefix} GPU Memory: {stats['current_memory_gb']:.2f} GB current, "
              f"{stats['peak_memory_gb']:.2f} GB peak")


# ----------------------- Model Definitions -----------------------

class SimpleMLP(nn.Module):
    """A simple MLP for CIFAR-10 classification"""
    def __init__(self, num_classes=10):
        super(SimpleMLP, self).__init__()
        self.features = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, 512),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.features(x)


class CNN(nn.Module):
    """A simple CNN for CIFAR-10"""
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def create_model(model_name, num_classes=10, pretrained=False):
    """Create model based on model_name"""
    if model_name == 'mlp':
        return SimpleMLP(num_classes=num_classes)
    elif model_name == 'cnn':
        return CNN(num_classes=num_classes)
    elif model_name == 'resnet18':
        model = resnet18(weights='DEFAULT' if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif model_name == 'resnet34':
        model = resnet34(weights='DEFAULT' if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif model_name == 'resnet50':
        model = resnet50(weights='DEFAULT' if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ----------------------- Training Functions -----------------------

def train_epoch(model, train_loader, criterion, optimizer, device, 
                epoch, use_amp=False, use_channels_last=False, scaler=None, use_standard_loader=False):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Track batch times for performance monitoring
    batch_times = []
    compute_times = []
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        batch_start = time.time()
        
        # Move data to device if using standard DataLoader
        if use_standard_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
        
        # Change memory format if requested
        if use_channels_last:
            inputs = inputs.to(memory_format=torch.channels_last)
        
        # Compute and timing for compute portion
        compute_start = time.time()
        
        # Forward + backward + optimize
        optimizer.zero_grad()
        
        if use_amp:
            # Use automatic mixed precision
            with autocast('cuda', dtype=torch.float16):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            # Scale gradients and optimize
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard precision training
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        # Compute time measurement
        compute_time = time.time() - compute_start
        compute_times.append(compute_time)
        
        # Update statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        batch_total = targets.size(0)
        batch_correct = predicted.eq(targets).sum().item()
        total += batch_total
        correct += batch_correct
        
        # Batch time measurement
        batch_time = time.time() - batch_start
        batch_times.append(batch_time)
        
        # Print progress every 10% of the batches
        if (batch_idx + 1) % max(1, len(train_loader) // 10) == 0:
            avg_loss = running_loss / (batch_idx + 1)
            avg_acc = 100. * correct / total
            avg_batch_time = sum(batch_times[-10:]) / min(len(batch_times), 10)
            avg_compute_time = sum(compute_times[-10:]) / min(len(compute_times), 10)
            
            print(f"Epoch {epoch} [{batch_idx+1}/{len(train_loader)}] "
                  f"Loss: {avg_loss:.4f} | Acc: {avg_acc:.1f}% | "
                  f"Batch: {avg_batch_time*1000:.1f}ms | Compute: {avg_compute_time*1000:.1f}ms")
    
    return {
        'loss': running_loss / len(train_loader),
        'accuracy': 100. * correct / total,
        'avg_batch_time': sum(batch_times) / len(batch_times),
        'avg_compute_time': sum(compute_times) / len(compute_times),
    }


def validate(model, val_loader, criterion, device, epoch, use_standard_loader=False):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            # Move data to device if using standard DataLoader
            if use_standard_loader:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Update statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    val_loss = running_loss / len(val_loader)
    val_acc = 100. * correct / total
    
    print(f"Validation: Loss: {val_loss:.4f} | Acc: {val_acc:.1f}%")
    
    return {
        'loss': val_loss,
        'accuracy': val_acc
    }


# ----------------------- Main Training Loop -----------------------

def train_model(model, train_loader, val_loader, device, num_epochs, use_amp=False, 
                use_channels_last=False, use_compile=False, compile_mode="default", use_standard_loader=False):
    """Main training loop"""
    # Setup for channels_last memory format
    if use_channels_last and 'cuda' in str(device):
        model = model.to(device, memory_format=torch.channels_last)
        print("Using channels_last memory format")
    else:
        model = model.to(device)
    
    # Apply torch.compile if requested and available
    if use_compile:
        if hasattr(torch, 'compile'):
            print(f"Compiling model with mode: {compile_mode}")
            model = torch.compile(model, mode=compile_mode)
        else:
            print("Warning: torch.compile is not available in your PyTorch version")
    
    # Setup loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=1, patience=50)
    
    # Setup mixed precision scaler if needed
    scaler = GradScaler(device) if use_amp else None
    
    # Enable cuDNN benchmarking for better performance
    if 'cuda' in str(device):
        torch.backends.cudnn.benchmark = True
    
    # Track best validation accuracy for model saving
    best_acc = 0.0
    
    # Training loop
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Train for one epoch
        train_stats = train_epoch(
            model, train_loader, criterion, optimizer, device, 
            epoch + 1, use_amp, use_channels_last, scaler, use_standard_loader
        )
        
        # Validate the model
        val_stats = validate(model, val_loader, criterion, device, epoch + 1, use_standard_loader)
        
        # Update learning rate based on validation loss
        scheduler.step(val_stats['loss'])
        
        # Save model if it's the best so far
        if val_stats['accuracy'] > best_acc:
            best_acc = val_stats['accuracy']
            # Uncomment to save the model
            # torch.save(model.state_dict(), 'best_model.pth')
        
        # Print epoch summary
        epoch_time = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Time: {epoch_time:.2f}s | Train Loss: {train_stats['loss']:.4f} | "
              f"Val Loss: {val_stats['loss']:.4f} | "
              f"Train Acc: {train_stats['accuracy']:.2f}% | "
              f"Val Acc: {val_stats['accuracy']:.2f}% | "
              f"LR: {current_lr:.6f}")
        
        # Print GPU memory stats if available
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            print(f"GPU Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
        
        print("-" * 80)
    
    return model, best_acc


# ----------------------- Data Loading -----------------------

def load_cifar10(batch_size=128, device=torch.device('cuda'), use_standard_loader=False, num_workers=4):
    """Load CIFAR-10 dataset using either FastDataLoader or standard DataLoader"""
    
    # Define transformations
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
    
    # Load train dataset
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train if use_standard_loader else None)
    
    # Load test dataset
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test if use_standard_loader else None)
    
    if use_standard_loader:
        print("Using standard PyTorch DataLoader")
        # Create standard DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False
        )
    else:
        print("Using FastDataLoader with data preloaded to GPU")
        # Define collate functions for FastDataLoader
        def train_collate(batch):
            data = torch.stack([transform_train(item[0]) for item in batch])
            targets = torch.tensor([item[1] for item in batch])
            return data, targets
        
        def test_collate(batch):
            data = torch.stack([transform_test(item[0]) for item in batch])
            targets = torch.tensor([item[1] for item in batch])
            return data, targets
        
        # Create FastDataLoaders
        train_loader = FastDataLoader(
            train_dataset, 
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            device=device,
            collate_fn=train_collate
        )
        
        val_loader = FastDataLoader(
            test_dataset, 
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            device=device,
            collate_fn=test_collate
        )
    
    return train_loader, val_loader


# ----------------------- Main Function -----------------------

def main():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training with FastDataLoader')
    parser.add_argument('--model', default='cnn', type=str, 
                        choices=['mlp', 'cnn', 'resnet18', 'resnet34', 'resnet50'],
                        help='model architecture')
    parser.add_argument('--batch-size', default=128, type=int,
                        help='mini-batch size')
    parser.add_argument('--epochs', default=10, type=int,
                        help='number of epochs')
    parser.add_argument('--no-cuda', action='store_true',
                        help='disable CUDA training')
    parser.add_argument('--amp', action='store_true',
                        help='use automatic mixed precision')
    parser.add_argument('--channels-last', action='store_true',
                        help='use channels_last memory format')
    parser.add_argument('--compile', action='store_true',
                        help='use torch.compile (PyTorch 2.0+)')
    parser.add_argument('--compile-mode', default='default', type=str,
                        choices=['default', 'reduce-overhead', 'max-autotune'],
                        help='compilation mode for torch.compile')
    parser.add_argument('--pretrained', action='store_true',
                        help='use pretrained model')
    parser.add_argument('--standard-loader', action='store_true',
                        help='use standard PyTorch DataLoader instead of FastDataLoader')
    parser.add_argument('--num-workers', default=4, type=int,
                        help='number of data loading workers (for standard DataLoader only)')
    
    args = parser.parse_args()
    
    # Set device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    print(f"Training Configuration:")
    print(f"Model: {args.model}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Device: {device}")
    print(f"Mixed Precision: {args.amp}")
    print(f"Channels Last: {args.channels_last}")
    print(f"Using torch.compile: {args.compile}")
    print(f"Compile Mode: {args.compile_mode}")
    print(f"Pretrained: {args.pretrained}")
    print(f"DataLoader: {'Standard' if args.standard_loader else 'Fast'}")
    if args.standard_loader:
        print(f"Number of Workers: {args.num_workers}")
    print("-" * 80)
    
    # Get datasets
    gpu_stats = GPUStats()
    gpu_stats.reset()
    
    train_loader, val_loader = load_cifar10(
        batch_size=args.batch_size, 
        device=device,
        use_standard_loader=args.standard_loader,
        num_workers=args.num_workers
    )
    
    gpu_stats.print_stats("After loading datasets:")
    
    # Create model
    model = create_model(args.model, num_classes=10, pretrained=args.pretrained)
    print(f"Created {args.model} model with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Train the model
    print("\nStarting training...")
    model, best_acc = train_model(
        model,
        train_loader,
        val_loader,
        device,
        args.epochs,
        use_amp=args.amp,
        use_channels_last=args.channels_last,
        use_compile=args.compile,
        compile_mode=args.compile_mode,
        use_standard_loader=args.standard_loader
    )
    
    print(f"\nTraining complete! Best validation accuracy: {best_acc:.2f}%")


if __name__ == '__main__':
    main()