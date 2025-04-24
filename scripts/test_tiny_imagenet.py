import os
import sys
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tqdm import tqdm

# Add project root to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

from src.utils.data import get_dataset, get_transforms

def test_tiny_imagenet_loading():
    """Test that Tiny ImageNet loads correctly with class-based structure"""
    print("Testing Tiny ImageNet dataset loading...")
    
    # Get paths
    data_dir = os.path.join(project_root, 'data')
    tiny_imagenet_path = os.path.join(data_dir, 'tiny-imagenet-200')
    
    if not os.path.exists(tiny_imagenet_path):
        print("Tiny ImageNet dataset not found. Run download_tiny_imagenet.py first.")
        return False
    
    # Check the val directory structure
    val_dir = os.path.join(tiny_imagenet_path, 'val')
    val_images_dir = os.path.join(val_dir, 'images')
    
    if os.path.exists(val_images_dir):
        print("ERROR: Validation set is not properly restructured. 'images' subfolder still exists.")
        return False
    
    # Verify class directories exist in val folder
    class_dirs = [d for d in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, d))]
    print(f"Found {len(class_dirs)} class directories in validation set")
    
    if len(class_dirs) != 200:  # Tiny ImageNet has 200 classes
        print(f"WARNING: Expected 200 class directories, found {len(class_dirs)}")
    
    # Test using get_dataset function
    print("Loading datasets using get_dataset function...")
    transform_train, transform_test = get_transforms('tiny-imagenet')
    
    try:
        train_dataset, test_dataset, num_classes = get_dataset(
            'tiny-imagenet', 
            transform_train, 
            transform_test,
            download=False
        )
        
        print(f"Successfully loaded datasets:")
        print(f"  Training dataset: {len(train_dataset)} images")
        print(f"  Test dataset: {len(test_dataset)} images")
        print(f"  Number of classes: {num_classes}")
        
        # Create data loaders to verify batch loading works
        batch_size = 64
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=2
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=2
        )
        
        # Iterate through a few batches to verify loading works
        print("Testing batch loading from train loader...")
        for i, (images, labels) in enumerate(train_loader):
            if i == 0:
                print(f"First batch shape: {images.shape}, labels shape: {labels.shape}")
                print(f"Label range: {labels.min().item()} to {labels.max().item()}")
            if i >= 2:  # Just test a few batches
                break
        
        print("Testing batch loading from test loader...")
        for i, (images, labels) in enumerate(test_loader):
            if i == 0:
                print(f"First batch shape: {images.shape}, labels shape: {labels.shape}")
                print(f"Label range: {labels.min().item()} to {labels.max().item()}")
            if i >= 2:  # Just test a few batches
                break
        
        # Verify consistency of class labels
        train_labels = set()
        test_labels = set()
        
        print("Checking class label consistency...")
        for i, (_, labels) in enumerate(tqdm(train_loader)):
            train_labels.update(labels.numpy())
            if i >= 10:  # Check enough batches to likely see all classes
                break
        
        for i, (_, labels) in enumerate(tqdm(test_loader)):
            test_labels.update(labels.numpy())
            if i >= 10:  # Check enough batches to likely see all classes
                break
        
        print(f"Train dataset classes: {len(train_labels)}")
        print(f"Test dataset classes: {len(test_labels)}")
        
        if train_labels != test_labels:
            diff = train_labels.symmetric_difference(test_labels)
            print(f"WARNING: Train and test datasets have different class labels. Difference: {diff}")
        else:
            print("Train and test datasets have consistent class labels.")
        
        print("Dataset validation completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return False

if __name__ == "__main__":
    success = test_tiny_imagenet_loading()
    if success:
        print("SUCCESS: Tiny ImageNet dataset is properly configured!")
    else:
        print("FAILED: Tiny ImageNet dataset test failed.")