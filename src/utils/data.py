import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, Dataset
import random
import os
import time

class SubsetDataset(Dataset):
    """Dataset wrapper for class subset selection"""
    def __init__(self, dataset, class_indices):
        self.dataset = dataset
        self.class_indices = class_indices
        self.indices = self._get_indices()
        
    def _get_indices(self):
        indices = []
        for i in range(len(self.dataset)):
            _, label = self.dataset[i]
            if label in self.class_indices:
                indices.append(i)
        return indices
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        image, label = self.dataset[self.indices[idx]]
        return image, label


def prepare_continual_learning_data(dataset, class_sequence, batch_size=128, val_split=0.2):
    """
    Prepare dataloaders for continual learning on a sequence of class subsets.
    
    Args:
        dataset: The full dataset (e.g., CIFAR10)
        class_sequence: List of lists, where each inner list contains class indices for a task
        batch_size: Batch size for dataloaders
        val_split: Fraction of data to use for validation
    
    Returns:
        Dictionary mapping task_id -> (train_loader, val_loader, fixed_train_loader, fixed_val_loader)
    """
    dataloaders = {}
    all_seen_classes = set()
    
    for task_id, classes in enumerate(class_sequence):
        current_classes = set(classes)
        
        # Create current task dataset
        current_dataset = SubsetDataset(dataset, classes)
        
        # Split into training and validation
        dataset_size = len(current_dataset)
        val_size = int(val_split * dataset_size)
        train_size = dataset_size - val_size
        
        indices = list(range(dataset_size))
        random.shuffle(indices)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        train_subset = Subset(current_dataset, train_indices)
        val_subset = Subset(current_dataset, val_indices)
        
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        # Fixed batches for metrics
        fixed_train = Subset(train_subset, range(min(500, len(train_subset))))
        fixed_val = Subset(val_subset, range(min(500, len(val_subset))))
        
        fixed_train_loader = DataLoader(fixed_train, batch_size=batch_size, shuffle=False)
        fixed_val_loader = DataLoader(fixed_val, batch_size=batch_size, shuffle=False)
        
        # For previous tasks (old classes)
        old_loaders = {}
        if task_id > 0:
            old_classes = all_seen_classes - current_classes
            if old_classes:
                old_dataset = SubsetDataset(dataset, list(old_classes))
                old_size = len(old_dataset)
                old_indices = list(range(old_size))
                random.shuffle(old_indices)
                
                old_train_size = int((1 - val_split) * old_size)
                old_train_indices = old_indices[:old_train_size]
                old_val_indices = old_indices[old_train_size:]
                
                old_train_subset = Subset(old_dataset, old_train_indices)
                old_val_subset = Subset(old_dataset, old_val_indices)
                
                old_train_loader = DataLoader(old_train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
                old_val_loader = DataLoader(old_val_subset, batch_size=batch_size, shuffle=False, num_workers=2)
                
                # Fixed old batches for metrics
                fixed_old_train = Subset(old_train_subset, range(min(500, len(old_train_subset))))
                fixed_old_val = Subset(old_val_subset, range(min(500, len(old_val_subset))))
                
                fixed_old_train_loader = DataLoader(fixed_old_train, batch_size=batch_size, shuffle=False)
                fixed_old_val_loader = DataLoader(fixed_old_val, batch_size=batch_size, shuffle=False)
                
                old_loaders = {
                    'train': old_train_loader,
                    'val': old_val_loader,
                    'fixed_train': fixed_old_train_loader,
                    'fixed_val': fixed_old_val_loader
                }
        
        # Store the dataloaders for this task
        dataloaders[task_id] = {
            'current': {
                'train': train_loader,
                'val': val_loader,
                'fixed_train': fixed_train_loader,
                'fixed_val': fixed_val_loader,
                'classes': classes
            },
            'old': old_loaders
        }
        
        # Update the set of all seen classes
        all_seen_classes.update(current_classes)
    
    return dataloaders


def get_transforms(dataset_name, no_augment=False):
    """
    Get appropriate data transformations for the dataset.
    
    Args:
        dataset_name: Name of the dataset
        no_augment: Whether to disable data augmentation
        
    Returns:
        transform_train: Transformations for training data
        transform_test: Transformations for test data
    """
    if dataset_name.lower() == 'cifar10':
        # CIFAR-10 normalization stats
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        img_size = 32
    elif dataset_name.lower() == 'cifar100':
        # CIFAR-100 normalization stats
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
        img_size = 32
    elif dataset_name.lower() == 'tiny-imagenet':
        # Tiny ImageNet normalization stats (approximated ImageNet stats)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        img_size = 64
    elif dataset_name.lower() == 'mnist':
        mean = (0.1307,)
        std = (0.3081,)
        img_size = 28
    else:
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        img_size = 32

    # Create transform with or without augmentation
    if no_augment:
        print("Data augmentation disabled")
        if dataset_name.lower() == 'mnist':
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            transform_train = transforms.Compose([
                transforms.Resize(img_size) if img_size != 64 else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
    else:
        if dataset_name.lower() == 'tiny-imagenet':
            # Transforms for Tiny ImageNet
            transform_train = transforms.Compose([
                transforms.RandomCrop(64, padding=8),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        elif dataset_name.lower() == 'mnist':
            transform_train = transforms.Compose([
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            # Transforms for CIFAR datasets
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
    
    # Test transforms
    if dataset_name.lower() == 'tiny-imagenet':
        transform_test = transforms.Compose([
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    
    return transform_train, transform_test


def get_dataset(dataset_name, transform_train=None, transform_test=None, download=True):
    """
    Get the specified dataset.
    
    Args:
        dataset_name: Name of the dataset
        transform_train: Transformations for training data
        transform_test: Transformations for test data
        download: Whether to download the dataset
        
    Returns:
        train_dataset: Training dataset
        test_dataset: Test dataset
        num_classes: Number of classes in the dataset
    """
    # Generate default transforms if not provided
    if transform_train is None or transform_test is None:
        transform_train, transform_test = get_transforms(dataset_name)
    
    # Get the path to the data directory relative to the project root
    script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(script_dir, 'data')
    
    if dataset_name.lower() == 'cifar10':
        train_dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=download, transform=transform_train)
            
        test_dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=False, download=download, transform=transform_test)
        
        num_classes = 10
    
    elif dataset_name.lower() == 'cifar100':
        train_dataset = torchvision.datasets.CIFAR100(
            root=data_dir, train=True, download=download, transform=transform_train)
            
        test_dataset = torchvision.datasets.CIFAR100(
            root=data_dir, train=False, download=download, transform=transform_test)
        
        num_classes = 100
            
    elif dataset_name.lower() == 'mnist':
        train_dataset = torchvision.datasets.MNIST(
            root=data_dir, train=True, download=download, transform=transform_train)
            
        test_dataset = torchvision.datasets.MNIST(
            root=data_dir, train=False, download=download, transform=transform_test)
        
        num_classes = 10
    
    elif dataset_name.lower() == 'tiny-imagenet':
        # Check if Tiny ImageNet dataset exists, if not suggest downloading
        tiny_imagenet_path = os.path.join(data_dir, 'tiny-imagenet-200')
        if not os.path.exists(tiny_imagenet_path):
            script_path = os.path.join(script_dir, 'scripts', 'download_tiny_imagenet.py')
            print(f"Tiny ImageNet dataset not found at {tiny_imagenet_path}")
            print(f"Please run: python {script_path}")
            raise FileNotFoundError(f"Tiny ImageNet dataset not found at {tiny_imagenet_path}")
        
        # Use ImageFolder to load Tiny ImageNet
        train_dataset = torchvision.datasets.ImageFolder(
            root=os.path.join(tiny_imagenet_path, 'train'),
            transform=transform_train
        )
        test_dataset = torchvision.datasets.ImageFolder(
            root=os.path.join(tiny_imagenet_path, 'val'),
            transform=transform_test
        )
        num_classes = 200
        print(f"Loaded Tiny ImageNet with {len(train_dataset)} training samples and {len(test_dataset)} validation samples")
    
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    return train_dataset, test_dataset, num_classes


def create_class_partitions(dataset, partition_sizes):
    """
    Create partitions of the dataset based on class labels.
    Uses an optimized algorithm that processes the dataset only once,
    creating a dictionary mapping from class label to indices to handle
    large datasets efficiently.
    
    Args:
        dataset: The full dataset
        partition_sizes: List of tuples defining the classes in each partition
                         e.g. [(0,1), (2,3), (4,5), (6,7,8,9)]
    
    Returns:
        List of dataset subsets, one for each partition
    """
    # Initialize empty lists for each class
    class_to_indices = {}
    
    # Group indices by class with a single pass through the dataset
    print("Creating partitions (this may take a moment for large datasets)...")
    start_time = time.time()
    
    # Process dataset in batches for large datasets
    for i, (_, label) in enumerate(dataset):
        label_int = int(label)
        if label_int not in class_to_indices:
            class_to_indices[label_int] = []
        class_to_indices[label_int].append(i)
    
    # Create partitions using the collected indices
    partitions = []
    for class_list in partition_sizes:
        partition_indices = []
        for cls in class_list:
            if cls in class_to_indices:
                partition_indices.extend(class_to_indices[cls])
        partitions.append(Subset(dataset, partition_indices))
    
    print(f"Partitioning completed in {time.time() - start_time:.2f} seconds")
    return partitions
