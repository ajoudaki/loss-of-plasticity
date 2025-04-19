"""
Dataset management utilities for continual learning experiments.
"""
import torch
from typing import Dict, List, Tuple, Any
from torch.utils.data import DataLoader, Subset
from omegaconf import DictConfig

from ..utils.data import get_dataset, get_transforms, create_class_partitions

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
    if cfg.task.partitions is not None:
        # Use provided partitions
        return [list(p) for p in cfg.task.partitions]
    else:
        # Auto-generate partitions
        tasks = cfg.task.tasks
        classes_per_task = cfg.task.classes_per_task
        
        # Ensure we have enough tasks for all classes
        if tasks is None:
            tasks = num_classes // classes_per_task
        
        # Create class sequence
        return [
            list(range(i * classes_per_task, min((i + 1) * classes_per_task, num_classes)))
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
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        # Fixed batches for metrics
        fixed_train = Subset(train_subset, range(min(500, len(train_subset))))
        fixed_val = Subset(val_subset, range(min(500, len(val_subset))))
        
        fixed_train_loader = DataLoader(fixed_train, batch_size=batch_size, shuffle=False)
        fixed_val_loader = DataLoader(fixed_val, batch_size=batch_size, shuffle=False)
        
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