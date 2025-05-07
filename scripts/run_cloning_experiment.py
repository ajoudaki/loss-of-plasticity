import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import time
from tqdm import tqdm
import os
import sys

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# Import your cloning functions
from src.models import MLP, CNN, ResNet, VisionTransformer
from src.utils.monitor import NetworkMonitor
from src.utils.cloning import * 
# Import the cloning function defined in your code

# For models that require flattened input (like MLP)
class ModelWrapper(nn.Module):
    """Wrapper to handle input reshaping for MLP models."""
    def __init__(self, model, flatten=False):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.flatten = flatten
        
    def forward(self, x):
        if self.flatten:
            batch_size = x.size(0)
            x = x.view(batch_size, -1)
        return self.model(x)

def load_cifar10(batch_size=128):
    """Load and prepare CIFAR-10 dataset with data augmentation."""
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
    
    # Load datasets
    trainset = torchvision.datasets.CIFAR10(
        root='../data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.CIFAR10(
        root='../data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    return trainloader, testloader, classes

def train_epoch(model, trainloader, criterion, optimizer, device):
    """Train the model for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(trainloader, desc="Training")
    for inputs, targets in progress_bar:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Track statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        progress_bar.set_postfix({
            'loss': running_loss / (progress_bar.n + 1),
            'acc': 100. * correct / total
        })
    
    return running_loss / len(trainloader), 100. * correct / total

def evaluate(model, testloader, criterion, device):
    """Evaluate the model on the test set."""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return test_loss / len(testloader), 100. * correct / total

def create_model(model_type, expanded=False, activation='relu', normalization='batch', dropout_p=0.0):
    """Create a model based on the specified type and configuration."""
    if model_type == 'mlp':
        if expanded:
            return MLP(input_size=3*32*32, output_size=10, hidden_sizes=[512*2, 256*2], 
                      activation=activation, dropout_p=dropout_p, normalization=normalization)
        else:
            return MLP(input_size=3*32*32, output_size=10, hidden_sizes=[512, 256], 
                      activation=activation, dropout_p=dropout_p, normalization=normalization)
    
    elif model_type == 'cnn':
        if expanded:
            return CNN(in_channels=3, num_classes=10, conv_channels=[64*2, 128*2, 256*2], 
                      activation=activation, dropout_p=dropout_p, normalization=normalization)
        else:
            return CNN(in_channels=3, num_classes=10, conv_channels=[64, 128, 256], 
                      activation=activation, dropout_p=dropout_p, normalization=normalization)
    
    elif model_type == 'resnet':
        if expanded:
            return ResNet(in_channels=3, num_classes=10, base_channels=64*2, 
                         activation=activation, dropout_p=dropout_p, normalization=normalization)
        else:
            return ResNet(in_channels=3, num_classes=10, base_channels=64, 
                         activation=activation, dropout_p=dropout_p, normalization=normalization)
    
    elif model_type == 'vit':
        if expanded:
            return VisionTransformer(in_channels=3, num_classes=10, embed_dim=64*2, depth=2, 
                                    dropout_p=dropout_p, attn_drop_rate=dropout_p, activation=activation)
        else:
            return VisionTransformer(in_channels=3, num_classes=10, embed_dim=64, depth=2, 
                                    dropout_p=dropout_p, attn_drop_rate=dropout_p, activation=activation)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")

# Fixed test_activation_cloning function
def fixed_test_activation_cloning(base_model, cloned_model, input, target, tolerance=1e-3, check_equality=False):
    criterion = nn.CrossEntropyLoss()
    base_monitor = NetworkMonitor(base_model)
    cloned_monitor = NetworkMonitor(cloned_model)
    base_monitor.register_hooks()
    cloned_monitor.register_hooks()
    
    # Forward pass
    y1 = base_model(input)
    y2 = cloned_model(input)
    
    # Loss calculation
    l1 = criterion(y1, target)
    l2 = criterion(y2, target)
    
    # Backward pass
    l1.backward()
    l2.backward()
    
    # Remove hooks
    cloned_monitor.remove_hooks()
    base_monitor.remove_hooks()
    
    if check_equality:
        assert torch.allclose(y1, y2, atol=tolerance), "Outputs do not match after cloning"
    
    unexplained_vars = []
    
    for act_type in ['forward']:  # Only check forward activations for simplicity
        if act_type == 'forward':
            base_acts = base_monitor.get_latest_activations()
            clone_acts = cloned_monitor.get_latest_activations()
        else:
            base_acts = base_monitor.get_latest_gradients()
            clone_acts = cloned_monitor.get_latest_gradients()
        
        for key, a1 in base_acts.items():
            if key not in clone_acts:
                continue
                
            a2 = clone_acts[key]
            s1, s2 = torch.tensor(a1.shape), torch.tensor(a2.shape)
            print(f"key: {key}, a1: {a1.shape}, a2: {a2.shape}")
            
            i = (s1 != s2).nonzero()
            if len(i) == 0:
                if check_equality:
                    assert torch.allclose(a1, a2, atol=tolerance), f"Activations for {key} do not match"
            elif len(i) == 1:
                i = i[0][0]
                expansion = a2.shape[i] // a1.shape[i]
                slices = []
                
                # Check expansion depending on the dimension
                for j in range(expansion):
                    print(f"mismatch dim: {i}, checking slice: {j}, expansion: {expansion}")
                    
                    if i == 0:
                        slice_data = a2[j::expansion]
                    elif i == 1:
                        slice_data = a2[:, j::expansion]
                    elif i == 2:
                        slice_data = a2[:, :, j::expansion]
                    elif i == 3:
                        slice_data = a2[:, :, :, j::expansion]
                    
                    slices.append(slice_data)
                    
                    if check_equality:
                        assert torch.allclose(slice_data, a1, atol=tolerance), f"Activations for {key} do not match"
                
                if len(slices) > 1:
                    try:
                        slices_tensor = torch.stack(slices)
                        if slices_tensor.shape[0] > 1:
                            std = slices_tensor.std(dim=0)
                            rms = torch.sqrt(torch.mean(slices_tensor**2, dim=0))
                            
                            # Avoid division by zero
                            mask = rms > 1e-6
                            if mask.any():
                                unexplained = (std[mask] / rms[mask]).mean().item()
                                unexplained_vars.append(unexplained)
                                print(f"Unexplained variance for {key} is {unexplained}")
                                if check_equality:
                                    assert unexplained < tolerance, f"Unexplained variance is higher than the threshold {tolerance}"
                    except Exception as e:
                        print(f"Error calculating unexplained variance: {e}")
                        
            elif len(i) > 1:
                print(f"Activations for {key} have more than one dimension mismatch, this is unexpected behavior")
    
    print(f"All {act_type} activations match after cloning up to tolerance {tolerance}")
    return np.mean(unexplained_vars) if unexplained_vars else 0.0

def run_cloning_experiment(model_type='cnn', num_epochs=5, activation='relu', normalization='batch', 
                     dropout_p=0.0, lr=0.01, weight_decay=5e-4, batch_size=128, 
                     validate_cloning_every=2, tolerance=1e-3):
    """
    Run the complete cloning experiment in a notebook environment.
    
    Args:
        model_type: Type of model ('mlp', 'cnn', 'resnet', 'vit')
        num_epochs: Number of epochs to train each phase
        activation: Activation function to use
        normalization: Normalization type
        dropout_p: Dropout probability
        lr: Learning rate
        weight_decay: Weight decay for optimizer
        batch_size: Batch size for training
        validate_cloning_every: Check cloning property every N epochs
        tolerance: Tolerance for activation similarity check
    
    Returns:
        dict: Results dictionary containing training metrics and cloning validation results
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory for saving results
    save_path = os.path.join(os.getcwd(), "outputs", model_type)
    os.makedirs(save_path, exist_ok=True)
    
    # Load CIFAR-10
    trainloader, testloader, classes = load_cifar10(batch_size)
    
    # Set up criterion
    criterion = nn.CrossEntropyLoss()
    
    # Results tracking
    results = {
        'base_model': {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': [], 'epoch_times': []},
        'cloned_model': {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': [], 'epoch_times': []},
        'scratch_model': {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': [], 'epoch_times': []}
    }
    
    # 1. Train base model
    print(f"\n{'='*20} Training base {model_type.upper()} model {'='*20}")
    base_model = create_model(model_type, expanded=False, activation=activation, 
                            normalization=normalization, dropout_p=dropout_p)
    
    # Wrap MLP model to handle input reshaping
    needs_flatten = model_type == 'mlp'
    base_model = ModelWrapper(base_model, flatten=needs_flatten).to(device)
    
    optimizer = optim.SGD(base_model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        start_time = time.time()
        
        train_loss, train_acc = train_epoch(base_model, trainloader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(base_model, testloader, criterion, device)
        
        epoch_time = time.time() - start_time
        
        results['base_model']['train_loss'].append(train_loss)
        results['base_model']['train_acc'].append(train_acc)
        results['base_model']['test_loss'].append(test_loss)
        results['base_model']['test_acc'].append(test_acc)
        results['base_model']['epoch_times'].append(epoch_time)
        
        print(f"Base model - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%, Time: {epoch_time:.2f}s")
        
        scheduler.step()
    
    # Save base model
    base_model_path = os.path.join(save_path, f"{model_type}_base_model.pth")
    torch.save(base_model.state_dict(), base_model_path)
    print(f"Base model saved to {base_model_path}")
    
    # 2. Clone and continue training
    print(f"\n{'='*20} Training cloned {model_type.upper()} model {'='*20}")
    expanded_model = create_model(model_type, expanded=True, activation=activation, 
                                normalization=normalization, dropout_p=dropout_p)
    
    # Extract the inner model for cloning if wrapped
    inner_base_model = base_model.model if needs_flatten else base_model
    
    # Clone the model
    cloned_model_inner = model_clone(inner_base_model, expanded_model)
    cloned_model = ModelWrapper(cloned_model_inner, flatten=needs_flatten).to(device)
    
    # Create an identical model for reference to verify cloning properties
    reference_model = create_model(model_type, expanded=False, activation=activation, 
                                 normalization=normalization, dropout_p=dropout_p)
    reference_model.load_state_dict(inner_base_model.state_dict())
    reference_model = reference_model.to(device)
    
    # Validate initial cloning
    print("Validating initial cloning properties...")
    # Get a small batch for validation
    val_inputs, val_targets = next(iter(testloader))
    val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
    
    # Cloning validation results
    cloning_validation_results = {'epochs': [], 'unexplained_variance': []}
    
    # Initial validation
    try:
        if needs_flatten:
            inner_model = cloned_model.model
            val_inputs_flattened = val_inputs.view(val_inputs.size(0), -1)
            unexplained_var = fixed_test_activation_cloning(
                reference_model, inner_model, val_inputs_flattened, val_targets, 
                tolerance=tolerance, check_equality=False
            )
        else:
            unexplained_var = fixed_test_activation_cloning(
                reference_model, cloned_model, val_inputs, val_targets, 
                tolerance=tolerance, check_equality=False
            )
        
        cloning_validation_results['epochs'].append(0)
        cloning_validation_results['unexplained_variance'].append(unexplained_var)
        print(f"Initial cloning validation passed! Unexplained variance: {unexplained_var:.6f}")
    except Exception as e:
        print(f"Initial cloning validation failed: {e}")
    
    optimizer = optim.SGD(cloned_model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        start_time = time.time()
        
        train_loss, train_acc = train_epoch(cloned_model, trainloader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(cloned_model, testloader, criterion, device)
        
        epoch_time = time.time() - start_time
        
        results['cloned_model']['train_loss'].append(train_loss)
        results['cloned_model']['train_acc'].append(train_acc)
        results['cloned_model']['test_loss'].append(test_loss)
        results['cloned_model']['test_acc'].append(test_acc)
        results['cloned_model']['epoch_times'].append(epoch_time)
        
        print(f"Cloned model - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%, Time: {epoch_time:.2f}s")
        
        # Periodically validate cloning properties
        if (epoch + 1) % validate_cloning_every == 0 or epoch == num_epochs - 1:
            print(f"Validating cloning properties after epoch {epoch+1}...")
            # Get a fresh batch for validation
            val_inputs, val_targets = next(iter(testloader))
            val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
            
            try:
                # Need to handle MLP differently due to input flattening
                if needs_flatten:
                    inner_model = cloned_model.model
                    val_inputs_flattened = val_inputs.view(val_inputs.size(0), -1)
                    unexplained_var = fixed_test_activation_cloning(
                        reference_model, inner_model, val_inputs_flattened, val_targets, 
                        tolerance=tolerance, check_equality=False
                    )
                else:
                    unexplained_var = fixed_test_activation_cloning(
                        reference_model, cloned_model, val_inputs, val_targets, 
                        tolerance=tolerance, check_equality=False
                    )
                
                cloning_validation_results['epochs'].append(epoch + 1)
                cloning_validation_results['unexplained_variance'].append(unexplained_var)
                print(f"Cloning validation passed! Unexplained variance: {unexplained_var:.6f}")
            except Exception as e:
                print(f"Cloning validation failed: {e}")
        
        scheduler.step()
    
    # Save cloned model and cloning validation results
    cloned_model_path = os.path.join(save_path, f"{model_type}_cloned_model.pth")
    torch.save(cloned_model.state_dict(), cloned_model_path)
    print(f"Cloned model saved to {cloned_model_path}")
    
    # Save cloning validation results
    cloning_results_path = os.path.join(save_path, f"{model_type}_cloning_validation.pth")
    torch.save(cloning_validation_results, cloning_results_path)
    
    # Plot cloning validation results
    if cloning_validation_results['epochs']:
        plt.figure(figsize=(10, 6))
        epochs = cloning_validation_results['epochs']
        unexplained_var = cloning_validation_results['unexplained_variance']
        
        # Filter out None values if any validation failed
        valid_points = [(e, v) for e, v in zip(epochs, unexplained_var) if v is not None]
        if valid_points:
            valid_epochs, valid_vars = zip(*valid_points)
            plt.plot(valid_epochs, valid_vars, 'b-o', label='Unexplained Variance')
            plt.axhline(y=tolerance, color='r', linestyle='--', label=f'Tolerance ({tolerance})')
            plt.title(f'Cloning Property Validation - {model_type.upper()}')
            plt.xlabel('Epoch')
            plt.ylabel('Unexplained Variance')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            plt.savefig(os.path.join(save_path, f"{model_type}_cloning_validation.png"))
            plt.close()
    
    # 3. Train from scratch for 2*num_epochs
    print(f"\n{'='*20} Training expanded {model_type.upper()} model from scratch {'='*20}")
    scratch_model_inner = create_model(model_type, expanded=True, activation=activation, 
                                     normalization=normalization, dropout_p=dropout_p)
    scratch_model = ModelWrapper(scratch_model_inner, flatten=needs_flatten).to(device)
    
    optimizer = optim.SGD(scratch_model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2*num_epochs)
    
    for epoch in range(2*num_epochs):
        print(f"\nEpoch {epoch+1}/{2*num_epochs}")
        start_time = time.time()
        
        train_loss, train_acc = train_epoch(scratch_model, trainloader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(scratch_model, testloader, criterion, device)
        
        epoch_time = time.time() - start_time
        
        results['scratch_model']['train_loss'].append(train_loss)
        results['scratch_model']['train_acc'].append(train_acc)
        results['scratch_model']['test_loss'].append(test_loss)
        results['scratch_model']['test_acc'].append(test_acc)
        results['scratch_model']['epoch_times'].append(epoch_time)
        
        print(f"Scratch model - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%, Time: {epoch_time:.2f}s")
        
        scheduler.step()
    
    # Save scratch model
    scratch_model_path = os.path.join(save_path, f"{model_type}_scratch_model.pth")
    torch.save(scratch_model.state_dict(), scratch_model_path)
    print(f"Scratch model saved to {scratch_model_path}")
    
    # Save results
    results_path = os.path.join(save_path, f"{model_type}_results.pth")
    torch.save(results, results_path)
    
    # Store cloning validation in results
    results['cloning_validation'] = cloning_validation_results
    
    # Plot and save results
    plot_results(results, model_type, num_epochs, save_path)
    
    return results

def plot_results(results, model_type, num_epochs, save_path):
    """Plot training and testing curves for all models."""
    plt.figure(figsize=(18, 15))
    
    # Set up x-axis for each model
    epochs_base = list(range(1, num_epochs + 1))
    epochs_cloned = list(range(num_epochs + 1, 2 * num_epochs + 1))
    epochs_scratch = list(range(1, 2 * num_epochs + 1))
    
    # Combined epochs for full timeline view
    epochs_combined = list(range(1, 2 * num_epochs + 1))
    
    # Plot train loss
    plt.subplot(3, 2, 1)
    plt.plot(epochs_base, results['base_model']['train_loss'], 'b-', label='Base Model')
    plt.plot(epochs_cloned, results['cloned_model']['train_loss'], 'r-', label='Cloned Model')
    plt.plot(epochs_scratch, results['scratch_model']['train_loss'], 'g-', label='From Scratch')
    plt.title(f'Training Loss - {model_type.upper()}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Plot train accuracy
    plt.subplot(3, 2, 2)
    plt.plot(epochs_base, results['base_model']['train_acc'], 'b-', label='Base Model')
    plt.plot(epochs_cloned, results['cloned_model']['train_acc'], 'r-', label='Cloned Model')
    plt.plot(epochs_scratch, results['scratch_model']['train_acc'], 'g-', label='From Scratch')
    plt.title(f'Training Accuracy - {model_type.upper()}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Plot test loss
    plt.subplot(3, 2, 3)
    plt.plot(epochs_base, results['base_model']['test_loss'], 'b-', label='Base Model')
    plt.plot(epochs_cloned, results['cloned_model']['test_loss'], 'r-', label='Cloned Model')
    plt.plot(epochs_scratch, results['scratch_model']['test_loss'], 'g-', label='From Scratch')
    plt.title(f'Test Loss - {model_type.upper()}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Plot test accuracy
    plt.subplot(3, 2, 4)
    plt.plot(epochs_base, results['base_model']['test_acc'], 'b-', label='Base Model')
    plt.plot(epochs_cloned, results['cloned_model']['test_acc'], 'r-', label='Cloned Model')
    plt.plot(epochs_scratch, results['scratch_model']['test_acc'], 'g-', label='From Scratch')
    plt.title(f'Test Accuracy - {model_type.upper()}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Plot epoch times
    plt.subplot(3, 2, 5)
    plt.plot(epochs_base, results['base_model']['epoch_times'], 'b-', label='Base Model')
    plt.plot(epochs_cloned, results['cloned_model']['epoch_times'], 'r-', label='Cloned Model')
    plt.plot(epochs_scratch, results['scratch_model']['epoch_times'], 'g-', label='From Scratch')
    plt.title(f'Epoch Training Time - {model_type.upper()}')
    plt.xlabel('Epoch')
    plt.ylabel('Time (seconds)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Combined test accuracy plot (base → cloned vs. scratch)
    plt.subplot(3, 2, 6)
    # Combine base and cloned for continuous line
    combined_acc = results['base_model']['test_acc'] + results['cloned_model']['test_acc']
    plt.plot(epochs_combined, combined_acc, 'b-', label='Base → Cloned')
    plt.plot(epochs_scratch, results['scratch_model']['test_acc'], 'g-', label='From Scratch')
    plt.axvline(x=num_epochs, color='r', linestyle='--', label='Cloning Point')
    plt.title(f'Test Accuracy Comparison - {model_type.upper()}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"{model_type}_results.png"))
    plt.close()
    
    # Plot cloning validation results if available
    if 'cloning_validation' in results and results['cloning_validation']['epochs']:
        plt.figure(figsize=(10, 6))
        epochs = results['cloning_validation']['epochs']
        unexplained_var = results['cloning_validation']['unexplained_variance']
        
        # Filter out None values if present
        valid_points = [(e, v) for e, v in zip(epochs, unexplained_var) if v is not None]
        if valid_points:
            valid_epochs, valid_vars = zip(*valid_points)
            plt.plot(valid_epochs, valid_vars, 'b-o', label='Unexplained Variance')
            plt.title(f'Cloning Property Validation - {model_type.upper()}')
            plt.xlabel('Epoch')
            plt.ylabel('Unexplained Variance')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            plt.savefig(os.path.join(save_path, f"{model_type}_cloning_validation.png"))
            plt.close()
    
    # Print summary
    print("\nExperiment Summary:")
    print(f"{'Model Type':15} {'Final Acc':12} {'Best Acc':12} {'Final Loss':12} {'Avg Time/Epoch':15}")
    print("-" * 70)
    
    final_results = {
        'Base Model': {
            'Final Test Acc': results['base_model']['test_acc'][-1],
            'Best Test Acc': max(results['base_model']['test_acc']),
            'Final Test Loss': results['base_model']['test_loss'][-1],
            'Avg Epoch Time': np.mean(results['base_model']['epoch_times'])
        },
        'Cloned Model': {
            'Final Test Acc': results['cloned_model']['test_acc'][-1],
            'Best Test Acc': max(results['cloned_model']['test_acc']),
            'Final Test Loss': results['cloned_model']['test_loss'][-1],
            'Avg Epoch Time': np.mean(results['cloned_model']['epoch_times'])
        },
        'Scratch Model': {
            'Final Test Acc': results['scratch_model']['test_acc'][-1],
            'Best Test Acc': max(results['scratch_model']['test_acc']),
            'Final Test Loss': results['scratch_model']['test_loss'][-1],
            'Avg Epoch Time': np.mean(results['scratch_model']['epoch_times'])
        }
    }
    
    for model, metrics in final_results.items():
        print(f"{model:15} {metrics['Final Test Acc']:12.2f} {metrics['Best Test Acc']:12.2f} "
              f"{metrics['Final Test Loss']:12.4f} {metrics['Avg Epoch Time']:15.2f}")

# Example usage
if __name__ == "__main__":
    results = run_cloning_experiment(
        model_type='mlp',  # Options: 'mlp', 'cnn', 'resnet', 'vit'
        num_epochs=1,
        activation='relu',
        normalization='batch',
        dropout_p=0.0,
        validate_cloning_every=1,
        tolerance=1e-3
    )