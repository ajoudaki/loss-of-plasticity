import torch
from torch.utils.data import DataLoader, Subset, Dataset
import random

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
