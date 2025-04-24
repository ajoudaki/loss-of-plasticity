import os
import sys
import torch
import random
import numpy as np
from tqdm import tqdm

# Add project root to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

from src.utils.data import prepare_continual_learning_dataloaders
from src.models.layers import set_seed
from omegaconf import OmegaConf

def verify_tiny_imagenet():
    """Verify Tiny ImageNet dataset with proper validation structure works in continual learning setup"""
    print("Verifying Tiny ImageNet with continual learning setup")
    
    # Create a basic config for testing
    config_str = """
    dataset:
      name: tiny-imagenet
    training:
      batch_size: 128
      no_augment: false
      seed: 42
    task:
      tasks: 2
      classes_per_task: 10
      partitions: null
    metrics:
      metrics_frequency: 5
      log_activation_histograms: false
    logging:
      use_wandb: false
    """
    
    cfg = OmegaConf.create(config_str)
    
    # Set random seed for reproducibility
    set_seed(cfg.training.seed)
    
    print("Loading data...")
    # Load datasets
    task_dataloaders, num_classes, class_sequence = prepare_continual_learning_dataloaders(cfg)
    
    print(f"Number of tasks: {len(task_dataloaders)}")
    print(f"Class sequence: {class_sequence}")
    
    # Verify both tasks load correctly
    total_train_samples = 0
    total_val_samples = 0
    
    for task_id, task_data in task_dataloaders.items():
        train_loader = task_data['train']
        val_loader = task_data['val']
        classes = task_data['classes']
        
        print(f"\nTask {task_id} with classes {classes}:")
        
        # Count train samples
        train_samples = 0
        labels_seen = set()
        
        for images, labels in train_loader:
            train_samples += len(images)
            labels_seen.update(labels.numpy())
            if train_samples >= 1000:  # Just check a portion for speed
                break
        
        print(f"  Train loader: {train_samples} samples checked")
        print(f"  Train labels seen: {sorted(list(labels_seen))}")
        
        # Count validation samples
        val_samples = 0
        val_labels_seen = set()
        
        for images, labels in val_loader:
            val_samples += len(images)
            val_labels_seen.update(labels.numpy())
            if val_samples >= 1000:  # Just check a portion for speed
                break
        
        print(f"  Val loader: {val_samples} samples checked")
        print(f"  Val labels seen: {sorted(list(val_labels_seen))}")
        
        # Check if the classes match expectations
        expected_classes = set(classes)
        print(f"  Expected classes: {sorted(list(expected_classes))}")
        
        total_train_samples += train_samples
        total_val_samples += val_samples
    
    print(f"\nTotal samples checked: {total_train_samples} train, {total_val_samples} validation")
    print("Validation complete!")

if __name__ == "__main__":
    verify_tiny_imagenet()