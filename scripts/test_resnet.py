import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np

# Import standard PyTorch models
import torchvision.models as torch_models

import argparse
import gc

# Our optimized ResNet implementation
class BasicBlock(nn.Module):
    """Basic ResNet block with activation and normalization."""
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1, activation='relu', 
                 use_batchnorm=True, norm_after_activation=False, downsample=None,
                 normalization_affine=True):
        super(BasicBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, 
                              padding=1, bias=not use_batchnorm)
        
        self.bn1 = None
        if use_batchnorm:
            self.bn1 = nn.BatchNorm2d(planes, affine=normalization_affine)
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, 
                              padding=1, bias=not use_batchnorm)
        
        self.bn2 = None
        if use_batchnorm:
            self.bn2 = nn.BatchNorm2d(planes, affine=normalization_affine)
        
        self.downsample = downsample
        self.activation = nn.ReLU(inplace=True) if activation == 'relu' else nn.Identity()
        self.norm_after_activation = norm_after_activation
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        
        if self.bn1 is not None and not self.norm_after_activation:
            out = self.bn1(out)
        
        out = self.activation(out)
        
        if self.bn1 is not None and self.norm_after_activation:
            out = self.bn1(out)
            
        out = self.conv2(out)
        
        if self.bn2 is not None and not self.norm_after_activation:
            out = self.bn2(out)
            
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.activation(out)
        
        if self.bn2 is not None and self.norm_after_activation:
            out = self.bn2(out)
            
        return out


class Bottleneck(nn.Module):
    """Bottleneck ResNet block with 1x1 -> 3x3 -> 1x1 conv layers."""
    expansion = 4
    
    def __init__(self, in_planes, planes, stride=1, activation='relu', 
                 use_batchnorm=True, norm_after_activation=False, downsample=None,
                 normalization_affine=True):
        super(Bottleneck, self).__init__()
        
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=not use_batchnorm)
        self.bn1 = None
        if use_batchnorm:
            self.bn1 = nn.BatchNorm2d(planes, affine=normalization_affine)
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, 
                              padding=1, bias=not use_batchnorm)
        self.bn2 = None
        if use_batchnorm:
            self.bn2 = nn.BatchNorm2d(planes, affine=normalization_affine)
        
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=not use_batchnorm)
        self.bn3 = None
        if use_batchnorm:
            self.bn3 = nn.BatchNorm2d(planes * self.expansion, affine=normalization_affine)
        
        self.downsample = downsample
        self.activation = nn.ReLU(inplace=True) if activation == 'relu' else nn.Identity()
        self.norm_after_activation = norm_after_activation
        
    def forward(self, x):
        identity = x
        
        # First block
        out = self.conv1(x)
        if self.bn1 is not None and not self.norm_after_activation:
            out = self.bn1(out)
        out = self.activation(out)
        if self.bn1 is not None and self.norm_after_activation:
            out = self.bn1(out)
        
        # Second block
        out = self.conv2(out)
        if self.bn2 is not None and not self.norm_after_activation:
            out = self.bn2(out)
        out = self.activation(out)
        if self.bn2 is not None and self.norm_after_activation:
            out = self.bn2(out)
        
        # Third block
        out = self.conv3(out)
        if self.bn3 is not None and not self.norm_after_activation:
            out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.activation(out)
        
        if self.bn3 is not None and self.norm_after_activation:
            out = self.bn3(out)
        
        return out


class ResNet(nn.Module):
    """Optimized ResNet architecture."""
    
    def __init__(self, 
                 block=BasicBlock,
                 layers=[2, 2, 2, 2],
                 num_classes=10,
                 in_channels=3,
                 base_channels=64,
                 activation='relu',
                 dropout_p=0.0,
                 use_batchnorm=True,
                 norm_after_activation=False,
                 normalization_affine=True):
        super(ResNet, self).__init__()
        
        self.use_batchnorm = use_batchnorm
        self.norm_after_activation = norm_after_activation
        self.inplanes = base_channels
        
        # Initial convolution layer
        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=3, 
                              stride=1, padding=1, bias=not use_batchnorm)
        
        self.bn1 = None
        if use_batchnorm:
            self.bn1 = nn.BatchNorm2d(base_channels, affine=normalization_affine)
        
        self.activation = nn.ReLU(inplace=True) if activation == 'relu' else nn.Identity()
        
        # Create layer groups
        self.layer1 = self._make_layer(block, base_channels, layers[0], stride=1,
                                     activation=activation, use_batchnorm=use_batchnorm,
                                     norm_after_activation=norm_after_activation,
                                     normalization_affine=normalization_affine)
        
        self.layer2 = self._make_layer(block, base_channels*2, layers[1], stride=2,
                                     activation=activation, use_batchnorm=use_batchnorm,
                                     norm_after_activation=norm_after_activation,
                                     normalization_affine=normalization_affine)
        
        self.layer3 = self._make_layer(block, base_channels*4, layers[2], stride=2,
                                     activation=activation, use_batchnorm=use_batchnorm,
                                     norm_after_activation=norm_after_activation,
                                     normalization_affine=normalization_affine)
        
        self.layer4 = self._make_layer(block, base_channels*8, layers[3], stride=2,
                                     activation=activation, use_batchnorm=use_batchnorm,
                                     norm_after_activation=norm_after_activation,
                                     normalization_affine=normalization_affine)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.dropout = None
        if dropout_p > 0:
            self.dropout = nn.Dropout(dropout_p)
        
        self.fc = nn.Linear(base_channels * 8 * block.expansion, num_classes)
        
        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:  # Check if affine=True
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        
        # Store for compatibility with original interface
        self.num_layers = len(layers)
        self.blocks_per_layer = layers
    
    def _make_layer(self, block, planes, blocks, stride=1, activation='relu',
                   use_batchnorm=True, norm_after_activation=False,
                   normalization_affine=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            layers = [nn.Conv2d(self.inplanes, planes * block.expansion,
                               kernel_size=1, stride=stride, bias=not use_batchnorm)]
            
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(planes * block.expansion, affine=normalization_affine))
            
            downsample = nn.Sequential(*layers)
        
        layers = []
        # First block with potential downsampling
        layers.append(block(self.inplanes, planes, stride, activation,
                          use_batchnorm, norm_after_activation, downsample,
                          normalization_affine=normalization_affine))
        
        self.inplanes = planes * block.expansion
        
        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, activation,
                              use_batchnorm, norm_after_activation,
                              normalization_affine=normalization_affine))
        
        return nn.Sequential(*layers)
    
    def _forward_impl(self, x):
        # Initial convolution block
        x = self.conv1(x)
        
        if self.bn1 is not None and not self.norm_after_activation:
            x = self.bn1(x)
        
        x = self.activation(x)
        
        if self.bn1 is not None and self.norm_after_activation:
            x = self.bn1(x)
        
        # Forward through layer groups
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global pooling and classification
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        if self.dropout is not None:
            x = self.dropout(x)
            
        x = self.fc(x)
        
        return x
    
    def forward(self, x):
        return self._forward_impl(x)


# Our model factory functions
def resnet18(num_classes=10, **kwargs):
    """Constructs a ResNet-18 model."""
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, **kwargs)


def resnet34(num_classes=10, **kwargs):
    """Constructs a ResNet-34 model."""
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, **kwargs)


def resnet50(num_classes=10, **kwargs):
    """Constructs a ResNet-50 model."""
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, **kwargs)


def resnet101(num_classes=10, **kwargs):
    """Constructs a ResNet-101 model."""
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, **kwargs)


def resnet152(num_classes=10, **kwargs):
    """Constructs a ResNet-152 model."""
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, **kwargs)


# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load and prepare CIFAR-10 dataset
def get_cifar10_loaders(batch_size=128):
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
    
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, test_loader

# Function to train for one epoch and measure time
def train_epoch(model, train_loader, criterion, optimizer):
    model.train()
    batch_times = []
    total_correct = 0
    total_samples = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Clear gradients
        optimizer.zero_grad()
        
        # Synchronize before timing (for accurate GPU measurements)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Start timing
        start_time = time.time()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Synchronize before stop timing
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # End timing
        end_time = time.time()
        batch_time = end_time - start_time
        batch_times.append(batch_time)
        
        # Calculate accuracy
        _, predicted = outputs.max(1)
        total_samples += targets.size(0)
        total_correct += predicted.eq(targets).sum().item()
        
        # Print progress
        if (batch_idx + 1) % 20 == 0:
            print(f'Batch: {batch_idx+1}/{len(train_loader)}, '
                  f'Loss: {loss.item():.4f}, '
                  f'Batch Time: {batch_time:.4f}s')
    
    avg_batch_time = np.mean(batch_times)
    accuracy = 100.0 * total_correct / total_samples
    
    return avg_batch_time, accuracy

# Function to evaluate model
def evaluate(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    test_loss /= len(test_loader)
    accuracy = 100.0 * correct / total
    
    return test_loss, accuracy

# Benchmark function
def benchmark_model(model_name, model_factory, batch_size=128, num_classes=10):
    print(f"\n{'='*50}")
    print(f"Benchmarking {model_name}")
    print(f"{'='*50}")
    
    # Get data loaders
    train_loader, test_loader = get_cifar10_loaders(batch_size)
    
    # Create model
    try:
        model = model_factory(num_classes=num_classes)
        model = model.to(device)
        
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        
        # Train for one epoch and measure time
        print(f"Training {model_name} for one epoch...")
        avg_batch_time, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        
        # Evaluate on validation set
        test_loss, test_acc = evaluate(model, test_loader, criterion)
        
        print(f"\nResults for {model_name}:")
        print(f"Average batch time: {avg_batch_time:.4f} seconds")
        print(f"Training accuracy: {train_acc:.2f}%")
        print(f"Validation accuracy: {test_acc:.2f}%")
        print(f"Validation loss: {test_loss:.4f}")
        
        return {
            'model_name': model_name,
            'avg_batch_time': avg_batch_time,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'test_loss': test_loss
        }
    except Exception as e:
        print(f"Error with {model_name}: {str(e)}")
        return {
            'model_name': model_name,
            'error': str(e)
        }

def main():
    BATCH_SIZE = 128
    NUM_CLASSES = 10  # CIFAR-10 has 10 classes
    
    # Function to adapt PyTorch models for CIFAR-10
    def adapt_pytorch_model_for_cifar10(model_fn):
        def wrapper(**kwargs):
            model = model_fn(pretrained=False)
            # Replace the first 7x7 conv with a 3x3 one (better for small CIFAR-10 images)
            model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            # Remove the maxpool layer after conv1 (preserves spatial dimensions for small images)
            model.maxpool = nn.Identity()
            # Adjust the fully connected layer to the number of classes
            model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
            return model
        return wrapper
    
    # Model configurations to benchmark (customize based on available resources)
    models_to_benchmark = [
        # Standard PyTorch models adapted for CIFAR-10
        ('PyTorch ResNet18', adapt_pytorch_model_for_cifar10(torch_models.resnet18)),
        ('PyTorch ResNet34', adapt_pytorch_model_for_cifar10(torch_models.resnet34)),
        ('PyTorch ResNet50', adapt_pytorch_model_for_cifar10(torch_models.resnet50)),
        # Only include the following if you have sufficient computational resources
        # ('PyTorch ResNet101', adapt_pytorch_model_for_cifar10(torch_models.resnet101)),
        # ('PyTorch ResNet152', adapt_pytorch_model_for_cifar10(torch_models.resnet152)),
        
        # Our optimized custom models
        ('Custom ResNet18', resnet18),
        ('Custom ResNet34', resnet34),
        ('Custom ResNet50', resnet50),
        # Only include the following if you have sufficient computational resources  
        # ('Custom ResNet101', resnet101),
        # ('Custom ResNet152', resnet152),
    ]
    
    results = []
    
    # Run benchmarks
    for model_name, model_factory in models_to_benchmark:
        try:
            result = benchmark_model(model_name, model_factory, BATCH_SIZE, NUM_CLASSES)
            results.append(result)
        except Exception as e:
            print(f"Error benchmarking {model_name}: {str(e)}")
            results.append({
                'model_name': model_name,
                'error': str(e)
            })
    
    # Print summary
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    print(f"{'Model':<20} | {'Avg Batch Time':<15} | {'Train Acc':<10} | {'Test Acc':<10}")
    print("-"*80)
    
    for result in results:
        if 'error' in result:
            print(f"{result['model_name']:<20} | ERROR: {result['error']}")
        else:
            print(f"{result['model_name']:<20} | {result['avg_batch_time']:.4f}s{' '*8} | "
                  f"{result['train_acc']:.2f}%{' '*5} | {result['test_acc']:.2f}%")


# Memory tracking functions
def print_gpu_memory():
    if torch.cuda.is_available():
        print(f"GPU Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"GPU Memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# CIFAR-10 data loaders
def get_cifar10_loaders(batch_size=128):
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
    
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, test_loader

# Training function with detailed batch timing
def train_epoch(model, train_loader, criterion, optimizer, max_batches=None):
    model.train()
    batch_times = []
    total_correct = 0
    total_samples = 0
    
    # For measuring memory during training
    peak_memory = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
            
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Clear gradients
        optimizer.zero_grad()
        
        # Synchronize before timing
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Start timing
        start_time = time.time()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Synchronize before stop timing
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # End timing
        end_time = time.time()
        batch_time = end_time - start_time
        batch_times.append(batch_time)
        
        # Calculate accuracy
        _, predicted = outputs.max(1)
        total_samples += targets.size(0)
        total_correct += predicted.eq(targets).sum().item()
        
        # Track memory usage
        if torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated() / 1024**2  # MB
            peak_memory = max(peak_memory, current_memory)
        
        # Print progress
        if (batch_idx + 1) % 10 == 0:
            print(f'Batch: {batch_idx+1}/{len(train_loader) if max_batches is None else max_batches}, '
                  f'Loss: {loss.item():.4f}, '
                  f'Batch Time: {batch_time:.4f}s, '
                  f'Current Memory: {torch.cuda.memory_allocated() / 1024**2:.2f}MB')
    
    avg_batch_time = np.mean(batch_times)
    accuracy = 100.0 * total_correct / total_samples
    
    # Compute detailed statistics
    min_time = np.min(batch_times)
    max_time = np.max(batch_times)
    std_time = np.std(batch_times)
    
    timing_stats = {
        'avg': avg_batch_time,
        'min': min_time,
        'max': max_time,
        'std': std_time,
        'peak_memory_mb': peak_memory
    }
    
    return timing_stats, accuracy

# Function to evaluate model
def evaluate(model, test_loader, criterion, max_batches=None):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
                
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    test_loss = test_loss / (max_batches if max_batches is not None else len(test_loader))
    accuracy = 100.0 * correct / total
    
    return test_loss, accuracy

# Create CIFAR-10 adapted PyTorch ResNet model
def adapt_pytorch_model_for_cifar10(model_fn, num_classes=10):
    model = model_fn(pretrained=False)
    # Replace the first 7x7 conv with a 3x3 one (better for small CIFAR-10 images)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # Remove the maxpool layer after conv1 (preserves spatial dimensions for small images)
    model.maxpool = nn.Identity()
    # Adjust the fully connected layer to the number of classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# Get the appropriate model
def get_model(model_name, num_classes=10):
    # PyTorch standard models
    if model_name == 'pytorch-resnet18':
        return adapt_pytorch_model_for_cifar10(torch_models.resnet18, num_classes)
    elif model_name == 'pytorch-resnet34':
        return adapt_pytorch_model_for_cifar10(torch_models.resnet34, num_classes)
    elif model_name == 'pytorch-resnet50':
        return adapt_pytorch_model_for_cifar10(torch_models.resnet50, num_classes)
    elif model_name == 'pytorch-resnet101':
        return adapt_pytorch_model_for_cifar10(torch_models.resnet101, num_classes)
    elif model_name == 'pytorch-resnet152':
        return adapt_pytorch_model_for_cifar10(torch_models.resnet152, num_classes)
    
    # Our custom models
    elif model_name == 'custom-resnet18':
        return resnet18(num_classes=num_classes)
    elif model_name == 'custom-resnet34':
        return resnet34(num_classes=num_classes)
    elif model_name == 'custom-resnet50':
        return resnet50(num_classes=num_classes)
    elif model_name == 'custom-resnet101':
        return resnet101(num_classes=num_classes)
    elif model_name == 'custom-resnet152':
        return resnet152(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")

def main():
    parser = argparse.ArgumentParser(description='ResNet Benchmarking')
    parser.add_argument('--model', type=str, required=True, 
                        choices=['pytorch-resnet18', 'pytorch-resnet34', 'pytorch-resnet50', 
                                'pytorch-resnet101', 'pytorch-resnet152',
                                'custom-resnet18', 'custom-resnet34', 'custom-resnet50', 
                                'custom-resnet101', 'custom-resnet152'],
                        help='Model to benchmark')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--max-batches', type=int, default=None, 
                        help='Maximum number of batches to run (for quick testing)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Get data loaders
    train_loader, test_loader = get_cifar10_loaders(args.batch_size)
    
    # Clear GPU memory before starting
    clear_gpu_memory()
    
    # Create model
    print(f"Creating model: {args.model}")
    model = get_model(args.model, num_classes=10)
    model = model.to(device)
    
    # Print model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Print initial memory usage
    print("\nInitial GPU memory usage:")
    print_gpu_memory()
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    
    # Train for one epoch and measure time
    print(f"\nTraining {args.model} for one epoch...")
    timing_stats, train_acc = train_epoch(model, train_loader, criterion, optimizer, max_batches=args.max_batches)
    
    # Print training timing statistics
    print("\nTraining timing statistics:")
    print(f"Average batch time: {timing_stats['avg']:.4f} seconds")
    print(f"Min batch time: {timing_stats['min']:.4f} seconds")
    print(f"Max batch time: {timing_stats['max']:.4f} seconds")
    print(f"Std dev of batch time: {timing_stats['std']:.4f} seconds")
    print(f"Peak memory usage: {timing_stats['peak_memory_mb']:.2f} MB")
    print(f"Training accuracy: {train_acc:.2f}%")
    
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    test_loss, test_acc = evaluate(model, test_loader, criterion, max_batches=args.max_batches)
    print(f"Validation loss: {test_loss:.4f}")
    print(f"Validation accuracy: {test_acc:.2f}%")
    
    # Print final memory usage
    print("\nFinal GPU memory usage:")
    print_gpu_memory()

if __name__ == "__main__":
    main()