"""
Dataset management utilities for neural network training and continual learning experiments.
"""
import os
import random
import time
from typing import Dict, List, Tuple, Any

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, Dataset
from omegaconf import DictConfig

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
        
        fixed_train_loader = DataLoader(fixed_train, batch_size=batch_size, shuffle=False, num_workers=2)
        fixed_val_loader = DataLoader(fixed_val, batch_size=batch_size, shuffle=False, num_workers=2)
        
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
        
        # Check if validation set is properly restructured
        val_images_dir = os.path.join(tiny_imagenet_path, 'val', 'images')
        if os.path.exists(val_images_dir):
            print(f"Tiny ImageNet validation set needs restructuring.")
            # Import the restructure function directly to avoid module import issues
            import sys
            sys.path.append(os.path.join(script_dir, 'scripts'))
            from download_tiny_imagenet import restructure_val_set
            restructure_val_set(tiny_imagenet_path)
        
        # Get class to idx mapping from train folder to ensure consistency
        train_folder = os.path.join(tiny_imagenet_path, 'train')
        train_classes = sorted([d.name for d in os.scandir(train_folder) if d.is_dir()])
        class_to_idx = {cls: i for i, cls in enumerate(train_classes)}
        
        # Use ImageFolder to load Tiny ImageNet with consistent class mapping
        train_dataset = torchvision.datasets.ImageFolder(
            root=train_folder,
            transform=transform_train,
            target_transform=None
        )
        
        # Ensure validation set uses the same class mapping as training set
        test_dataset = torchvision.datasets.ImageFolder(
            root=os.path.join(tiny_imagenet_path, 'val'),
            transform=transform_test,
            target_transform=None
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


# Dataset Manager Functions from dataset_manager.py

def generate_class_sequence(cfg: DictConfig, num_classes: int) -> List[List[int]]:
    """
    Generate a sequence of class partitions for continual learning tasks.
    
    Args:
        cfg: Configuration object
        num_classes: Total number of classes in the dataset
        
    Returns:
        List of class lists, one for each task
    """
    # Parse partitions if provided
    if cfg.training.partitions is not None:
        # Use provided partitions
        return [list(p) for p in cfg.training.partitions]
    else:
        # Auto-generate partitions
        tasks = cfg.training.tasks
        classes_per_task = cfg.training.classes_per_task
        
        # if tasks and classes_per_task are not provided, assume we have one task and all classes in it 
        if tasks is None and classes_per_task is None:
            tasks = 1
            classes_per_task = num_classes
        elif tasks is None:
            # set tasks to the ceiling of num_classes / classes_per_task
            tasks = (num_classes + classes_per_task - 1) // classes_per_task
        elif classes_per_task is None:
            # set classes_per_task to the ceiling of num_classes / tasks
            classes_per_task = (num_classes + tasks - 1) // tasks
        
        # Ensure we have enough tasks for all classes
        if tasks is None:
            tasks = num_classes // classes_per_task
        
        # Get all class IDs
        all_class_ids = list(range(num_classes))
        
        # Shuffle class IDs if this is a continual learning scenario
        if tasks is not None or classes_per_task is not None:
            random.seed(cfg.training.seed)
            random.shuffle(all_class_ids)
            print(f"Shuffled class IDs with seed {cfg.training.seed}: {all_class_ids}")
        
        # Create class sequence
        return [
            all_class_ids[i * classes_per_task:min((i + 1) * classes_per_task, num_classes)]
            for i in range(tasks)
        ]

def create_task_dataloaders(
    partitioned_train_datasets: List[Subset],
    partitioned_val_datasets: List[Subset],
    class_sequence: List[List[int]],
    batch_size: int
) -> Dict[int, Dict[str, Any]]:
    """
    Create data loaders for each task.
    
    Args:
        partitioned_train_datasets: List of training dataset partitions
        partitioned_val_datasets: List of validation dataset partitions
        class_sequence: List of class lists, one for each task
        batch_size: Batch size for the data loaders
        
    Returns:
        Dictionary mapping task IDs to data loaders
    """
    task_dataloaders = {}
    
    for task_id, (train_subset, val_subset) in enumerate(zip(partitioned_train_datasets, partitioned_val_datasets)):
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2, multiprocessing_context='fork')
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2, multiprocessing_context='fork')
        
        # Fixed batches for metrics
        fixed_train = Subset(train_subset, range(min(500, len(train_subset))))
        fixed_val = Subset(val_subset, range(min(500, len(val_subset))))
        
        fixed_train_loader = DataLoader(fixed_train, batch_size=batch_size,  num_workers=2, shuffle=False)
        fixed_val_loader = DataLoader(fixed_val, batch_size=batch_size,  num_workers=2, shuffle=False)
        
        task_dataloaders[task_id] = {
            'train': train_loader,
            'val': val_loader,
            'fixed_train': fixed_train_loader,
            'fixed_val': fixed_val_loader,
            'classes': class_sequence[task_id]
        }
    
    return task_dataloaders

def update_dataset_config(cfg: DictConfig, dataset_name: str, num_classes: int) -> None:
    """
    Update dataset configuration parameters based on the dataset name.
    
    Args:
        cfg: Configuration object
        dataset_name: Name of the dataset
        num_classes: Number of classes in the dataset
    """
    # Set dataset specific parameters
    if dataset_name.lower() == 'mnist':
        cfg.dataset.input_size = 784  # 28x28
        cfg.dataset.img_size = 28
        cfg.dataset.in_channels = 1
    elif dataset_name.lower() == 'cifar10':
        cfg.dataset.input_size = 3072  # 32x32x3
        cfg.dataset.img_size = 32
        cfg.dataset.in_channels = 3
    elif dataset_name.lower() == 'cifar100':
        cfg.dataset.input_size = 3072  # 32x32x3
        cfg.dataset.img_size = 32
        cfg.dataset.in_channels = 3
    elif dataset_name.lower() == 'tiny-imagenet':
        cfg.dataset.input_size = 12288  # 64x64x3
        cfg.dataset.img_size = 64
        cfg.dataset.in_channels = 3
    
    # Update number of classes
    cfg.dataset.num_classes = num_classes

def prepare_continual_learning_dataloaders(cfg: DictConfig) -> Tuple[Dict[int, Dict[str, Any]], int, List[List[int]]]:
    """
    Prepare dataloaders for continual learning experiments.
    
    Args:
        cfg: Configuration object
        
    Returns:
        task_dataloaders: Dictionary mapping task IDs to data loaders
        num_classes: Number of classes in the dataset
        class_sequence: List of class lists, one for each task
    """
    # Create transforms with or without augmentation
    transform_train, transform_test = get_transforms(cfg.dataset.name, cfg.training.no_augment)
    
    # Get dataset
    train_dataset, val_dataset, num_classes = get_dataset(cfg.dataset.name, transform_train, transform_test)
    
    # Update dataset config parameters
    update_dataset_config(cfg, cfg.dataset.name, num_classes)
    
    # Generate class sequence
    class_sequence = generate_class_sequence(cfg, num_classes)
    print(f"Class sequence: {class_sequence}")
    
    # Create partition datasets
    partitioned_train_datasets = create_class_partitions(
        train_dataset, [tuple(cls_list) for cls_list in class_sequence])
    
    partitioned_val_datasets = create_class_partitions(
        val_dataset, [tuple(cls_list) for cls_list in class_sequence])
    
    # Create dataloader dict
    task_dataloaders = create_task_dataloaders(
        partitioned_train_datasets, 
        partitioned_val_datasets, 
        class_sequence, 
        cfg.training.batch_size
    )
    
    return task_dataloaders, num_classes, class_sequence