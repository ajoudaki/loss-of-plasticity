import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import time
from typing import List, Tuple, Dict, Optional

# Set random seeds for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

def generate_permutation(n_pixels: int) -> np.ndarray:
    """
    Generate a random permutation of pixel indices.
    
    Args:
        n_pixels: Number of pixels in the image (e.g., 28*28=784 for MNIST)
    
    Returns:
        Array containing the permutation indices
    """
    perm = np.arange(n_pixels)
    np.random.shuffle(perm)
    return perm

def apply_permutation(images: torch.Tensor, permutation: np.ndarray) -> torch.Tensor:
    """
    Apply a permutation to a batch of images.
    
    Args:
        images: Tensor of shape (batch_size, n_pixels)
        permutation: Array containing the permutation indices
    
    Returns:
        Tensor of permuted images with the same shape as the input
    """
    batch_size = images.shape[0]
    n_pixels = images.shape[1]
    assert len(permutation) == n_pixels, "Permutation size doesn't match image size"
    
    # Apply the permutation
    permuted_images = images[:, permutation]
    return permuted_images

class FeedForwardNetwork(nn.Module):
    """
    Feed-forward network for the Permuted MNIST experiment.
    """
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int):
        super(FeedForwardNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        
        # Create layers
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        
        # Hidden layers
        for i in range(1, len(hidden_sizes)):
            self.layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
        
        # Output layer
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))
        
        # Initialize weights using Kaiming initialization
        for layer in self.layers:
            nn.init.kaiming_uniform_(layer.weight, a=0, mode='fan_in', nonlinearity='relu')
            nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        # Flatten input if necessary
        if len(x.shape) > 2:
            x = x.view(x.shape[0], -1)
        
        # Apply hidden layers with ReLU activation
        for i in range(len(self.hidden_sizes)):
            x = F.relu(self.layers[i](x))
        
        # Apply output layer
        x = self.layers[-1](x)
        return x
    
    def effective_rank(self, activations: torch.Tensor) -> float:
        """
        Calculate the effective rank of hidden layer activations.
        
        Args:
            activations: Tensor of activations from a hidden layer
        
        Returns:
            Effective rank value
        """
        # Calculate singular values
        u, s, v = torch.svd(activations)
        
        # Normalize singular values
        p = s / torch.sum(s)
        
        # Calculate entropy
        entropy = -torch.sum(p * torch.log(p + 1e-10))
        
        # Effective rank is exp(entropy)
        return torch.exp(entropy).item()
    
    def count_dead_units(self, activations: torch.Tensor) -> int:
        """
        Count the number of dead units (ReLU units that are always 0).
        
        Args:
            activations: Tensor of activations from a hidden layer
        
        Returns:
            Number of dead units
        """
        # A unit is dead if it's always 0 across all samples
        return torch.sum(torch.max(activations, dim=0)[0] == 0).item()
    
    def average_weight_magnitude(self) -> float:
        """
        Calculate the average magnitude of all weights in the network.
        
        Returns:
            Average weight magnitude
        """
        total_mag = 0.0
        total_weights = 0
        
        for layer in self.layers:
            total_mag += torch.sum(torch.abs(layer.weight)).item()
            total_weights += layer.weight.numel()
        
        return total_mag / total_weights

def l2_regularization_step(model, weight_decay):
    """Apply L2 regularization (weight decay) manually."""
    with torch.no_grad():
        for param in model.parameters():
            param.mul_(1 - weight_decay)

def shrink_and_perturb_step(model, weight_decay, noise_variance):
    """Apply shrink-and-perturb: L2 regularization + Gaussian noise."""
    with torch.no_grad():
        for param in model.parameters():
            # Shrink step (L2 regularization)
            param.mul_(1 - weight_decay)
            # Perturb step (add Gaussian noise)
            param.add_(torch.randn_like(param) * np.sqrt(noise_variance))

class OnlineNormLayer(nn.Module):
    """
    Online normalization layer that normalizes activations online.
    """
    def __init__(self, num_features, momentum=0.1, eps=1e-5):
        super(OnlineNormLayer, self).__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
    
    def forward(self, x):
        # Update running stats if training
        if self.training:
            mean = x.mean(0)
            var = x.var(0, unbiased=False)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        
        # Normalize
        x_norm = (x - self.running_mean) / (self.running_var + self.eps).sqrt()
        
        # Scale and shift
        return self.weight * x_norm + self.bias

def continual_backpropagation_step(
    model: FeedForwardNetwork, 
    x: torch.Tensor,
    replacement_rate: float, 
    maturity_threshold: int,
    utilities: List[torch.Tensor],
    avg_activations: List[torch.Tensor],
    ages: List[torch.Tensor],
    decay_rate: float = 0.99
):
    """
    Perform one step of continual backpropagation.
    
    Args:
        model: The neural network model
        x: Current input batch
        replacement_rate: Fraction of eligible units to replace
        maturity_threshold: Minimum age before a unit is eligible for replacement
        utilities: List of utility values for hidden units
        avg_activations: List of average activations for hidden units
        ages: List of ages for hidden units
        decay_rate: Decay rate for running averages
    """
    with torch.no_grad():
        # Forward pass to collect activations
        activations = []
        h = x.view(x.shape[0], -1)
        
        for i in range(len(model.hidden_sizes)):
            h = model.layers[i](h)
            activations.append(h.clone())
            h = F.relu(h)
        
        # Update ages for all hidden layers
        for i in range(len(ages)):
            ages[i] += 1
        
        # Process each hidden layer
        for i in range(len(model.hidden_sizes)):
            layer_size = model.hidden_sizes[i]
            
            # Update running average of activations
            h_act = F.relu(activations[i]).mean(0)  # Average over batch
            avg_activations[i] = decay_rate * avg_activations[i] + (1 - decay_rate) * h_act
            bias_corrected_avg = avg_activations[i] / (1 - decay_rate ** ages[i])
            
            # Calculate contribution utility (mean-corrected)
            if i < len(model.hidden_sizes) - 1:
                # For hidden layers except the last one
                outgoing_weights_mag = torch.sum(torch.abs(model.layers[i+1].weight), dim=0)
            else:
                # For the last hidden layer
                outgoing_weights_mag = torch.sum(torch.abs(model.layers[-1].weight), dim=0)
            
            contribution = torch.abs(h_act - bias_corrected_avg) * outgoing_weights_mag
            
            # Calculate adaptation utility (inverse of input weight magnitude)
            incoming_weights_mag = torch.sum(torch.abs(model.layers[i].weight), dim=1) + 1e-10
            adaptation = 1.0 / incoming_weights_mag
            
            # Overall utility
            utility = contribution * adaptation
            utilities[i] = decay_rate * utilities[i] + (1 - decay_rate) * utility
            bias_corrected_utility = utilities[i] / (1 - decay_rate ** ages[i])
            
            # Find eligible units to replace
            eligible_units = (ages[i] > maturity_threshold).nonzero().squeeze(-1)
            
            if len(eligible_units) > 0 and replacement_rate > 0:
                # Number of units to replace
                num_units_to_replace = max(1, int(layer_size * replacement_rate))
                
                # Get utilities of eligible units
                eligible_utilities = bias_corrected_utility[eligible_units]
                
                # Choose units with lowest utility
                _, indices = torch.topk(eligible_utilities, min(len(eligible_units), layer_size), 
                                       largest=False)
                units_to_replace = eligible_units[indices[:num_units_to_replace]]
                
                if len(units_to_replace) > 0:
                    # Reinitialize input weights
                    gain = nn.init.calculate_gain('relu')
                    fan_in = model.layers[i].weight.size(1)
                    bound = gain * np.sqrt(3.0 / fan_in)
                    nn.init.uniform_(model.layers[i].weight[units_to_replace], -bound, bound)
                    
                    # Reset biases to zero
                    model.layers[i].bias[units_to_replace] = 0
                    
                    # Initialize output weights to zero
                    if i < len(model.hidden_sizes) - 1:
                        model.layers[i+1].weight[:, units_to_replace] = 0
                    else:
                        model.layers[-1].weight[:, units_to_replace] = 0
                    
                    # Reset utility, activation average, and age
                    utilities[i][units_to_replace] = 0
                    avg_activations[i][units_to_replace] = 0
                    ages[i][units_to_replace] = 0

def run_online_permuted_mnist_experiment(
    hidden_sizes: List[int] = [2000, 2000, 2000],
    num_tasks: int = 10,
    examples_per_task: int = 60000,
    step_size: float = 0.003,
    optimizer_type: str = 'sgd',  # 'sgd' or 'adam'
    use_dropout: bool = False,
    dropout_prob: float = 0.0,
    use_l2_reg: bool = False,
    weight_decay: float = 0.0,
    use_shrink_perturb: bool = False,
    noise_variance: float = 0.0,
    use_online_norm: bool = False,
    online_norm_momentum: float = 0.1,
    use_continual_backprop: bool = False,
    replacement_rate: float = 0.0,
    maturity_threshold: int = 100,
    device: str = 'cpu'
):
    """
    Run the Online Permuted MNIST experiment.
    
    Args:
        hidden_sizes: List of hidden layer sizes
        num_tasks: Number of permuted MNIST tasks
        examples_per_task: Number of examples per task
        step_size: Learning rate
        optimizer_type: Type of optimizer ('sgd' or 'adam')
        use_dropout: Whether to use dropout
        dropout_prob: Dropout probability
        use_l2_reg: Whether to use L2 regularization
        weight_decay: L2 regularization coefficient
        use_shrink_perturb: Whether to use Shrink-and-Perturb
        noise_variance: Variance of the noise for Shrink-and-Perturb
        use_online_norm: Whether to use Online Normalization
        online_norm_momentum: Momentum for Online Normalization
        use_continual_backprop: Whether to use Continual Backpropagation
        replacement_rate: Replacement rate for Continual Backpropagation
        maturity_threshold: Maturity threshold for Continual Backpropagation
        device: Device to run the experiment on ('cpu' or 'cuda')
    
    Returns:
        Dictionary containing experiment results
    """
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    mnist_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    # Combine train and test sets
    all_images = torch.cat([mnist_train.data.float(), mnist_test.data.float()])
    all_labels = torch.cat([mnist_train.targets, mnist_test.targets])
    
    # Normalize and flatten images
    all_images = all_images.view(-1, 28*28) / 255.0
    
    # Create model
    input_size = 28 * 28
    output_size = 10
    model = FeedForwardNetwork(input_size, hidden_sizes, output_size).to(device)
    
    # Add dropout if needed
    if use_dropout:
        dropout_layers = nn.ModuleList([nn.Dropout(dropout_prob) for _ in range(len(hidden_sizes))])
    
    # Add online normalization if needed
    if use_online_norm:
        norm_layers = nn.ModuleList([OnlineNormLayer(size, momentum=online_norm_momentum) for size in hidden_sizes])
    
    # Initialize optimizer
    if optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=step_size, weight_decay=0.0)  # We'll handle weight decay manually
    else:
        optimizer = optim.SGD(model.parameters(), lr=step_size, weight_decay=0.0)  # We'll handle weight decay manually
    
    # Initialize loss function
    criterion = nn.CrossEntropyLoss()
    
    # Arrays to store results
    accuracy_per_task = np.zeros(num_tasks)
    weight_magnitude_per_task = np.zeros(num_tasks)
    dead_units_per_task = np.zeros((num_tasks, len(hidden_sizes)))
    effective_rank_per_task = np.zeros((num_tasks, len(hidden_sizes)))
    
    # For continual backpropagation
    if use_continual_backprop:
        utilities = [torch.zeros(size, device=device) for size in hidden_sizes]
        avg_activations = [torch.zeros(size, device=device) for size in hidden_sizes]
        ages = [torch.zeros(size, device=device) for size in hidden_sizes]
    
    # Generate permutations for each task
    permutations = [generate_permutation(input_size) for _ in range(num_tasks)]
    
    # Run experiment for each task
    for task_idx in range(num_tasks):
        print(f"Task {task_idx+1}/{num_tasks}")
        permutation = permutations[task_idx]
        
        # Shuffle the dataset for this task
        indices = torch.randperm(len(all_images))
        task_images = all_images[indices]
        task_labels = all_labels[indices]
        
        # Apply permutation to the images
        permuted_images = apply_permutation(task_images, permutation)
        
        # Evaluate the network before training
        if task_idx > 0:
            # Collect activations and calculate metrics before training
            with torch.no_grad():
                # Sample 2000 random examples for evaluation
                eval_idx = torch.randperm(len(permuted_images))[:2000]
                eval_images = permuted_images[eval_idx].to(device)
                
                # Forward pass to collect activations
                activations = []
                h = eval_images
                
                if len(h.shape) > 2:
                    h = h.view(h.shape[0], -1)
                
                for i in range(len(hidden_sizes)):
                    h = model.layers[i](h)
                    activations.append(h.clone())
                    
                    if use_online_norm:
                        h = norm_layers[i](h)
                    
                    h = F.relu(h)
                    
                    if use_dropout:
                        h = dropout_layers[i](h)
                
                # Calculate metrics
                for i in range(len(hidden_sizes)):
                    dead_units_per_task[task_idx, i] = model.count_dead_units(activations[i])
                    effective_rank_per_task[task_idx, i] = model.effective_rank(activations[i])
                
                weight_magnitude_per_task[task_idx] = model.average_weight_magnitude()
        
        # Training loop
        correct = 0
        total = 0
        
        for i in tqdm(range(examples_per_task)):
            # Get current example
            x = permuted_images[i].unsqueeze(0).to(device)
            y = task_labels[i].unsqueeze(0).to(device)
            
            # Forward pass
            outputs = model(x)
            
            # Calculate loss
            loss = criterion(outputs, y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Apply regularization or other techniques
            if use_l2_reg:
                l2_regularization_step(model, weight_decay)
            
            if use_shrink_perturb:
                shrink_and_perturb_step(model, weight_decay, noise_variance)
            
            if use_continual_backprop:
                continual_backpropagation_step(
                    model, x, replacement_rate, maturity_threshold,
                    utilities, avg_activations, ages
                )
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
        
        # Record accuracy for this task
        accuracy_per_task[task_idx] = 100 * correct / total
        
        # For the first task, calculate metrics after training
        if task_idx == 0:
            with torch.no_grad():
                # Sample 2000 random examples for evaluation
                eval_idx = torch.randperm(len(permuted_images))[:2000]
                eval_images = permuted_images[eval_idx].to(device)
                
                # Forward pass to collect activations
                activations = []
                h = eval_images
                
                if len(h.shape) > 2:
                    h = h.view(h.shape[0], -1)
                
                for i in range(len(hidden_sizes)):
                    h = model.layers[i](h)
                    activations.append(h.clone())
                    
                    if use_online_norm:
                        h = norm_layers[i](h)
                    
                    h = F.relu(h)
                    
                    if use_dropout:
                        h = dropout_layers[i](h)
                
                # Calculate metrics
                for i in range(len(hidden_sizes)):
                    dead_units_per_task[task_idx, i] = model.count_dead_units(activations[i])
                    effective_rank_per_task[task_idx, i] = model.effective_rank(activations[i])
                
                weight_magnitude_per_task[task_idx] = model.average_weight_magnitude()
        
        print(f"Task {task_idx+1} Accuracy: {accuracy_per_task[task_idx]:.2f}%")
        print(f"Weight Magnitude: {weight_magnitude_per_task[task_idx]:.6f}")
        print(f"Average Dead Units: {np.mean(dead_units_per_task[task_idx]):.2f}")
        print(f"Average Effective Rank: {np.mean(effective_rank_per_task[task_idx]):.2f}")
    
    return {
        'accuracy_per_task': accuracy_per_task,
        'weight_magnitude_per_task': weight_magnitude_per_task,
        'dead_units_per_task': dead_units_per_task,
        'effective_rank_per_task': effective_rank_per_task,
        'hidden_sizes': hidden_sizes
    }

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Experiment settings
    num_tasks = 50  # Reduced for faster execution
    hidden_sizes = [2000, 2000, 2000]
    examples_per_task = 60000
    
    # Example usage: vanilla backpropagation
    print("Running Online Permuted MNIST experiment with vanilla backpropagation")
    result_bp = run_online_permuted_mnist_experiment(
        hidden_sizes=hidden_sizes,
        num_tasks=num_tasks,
        examples_per_task=examples_per_task,
        step_size=0.003,
        device=device
    )
    
    # Example usage: with continual backpropagation
    print("Running Online Permuted MNIST experiment with continual backpropagation")
    result_cbp = run_online_permuted_mnist_experiment(
        hidden_sizes=hidden_sizes,
        num_tasks=num_tasks,
        examples_per_task=examples_per_task,
        step_size=0.003,
        use_continual_backprop=True,
        replacement_rate=1e-4,  # Replace approximately 0.01% of units per example
        maturity_threshold=100,
        device=device
    )
    
    # Plot accuracy comparison
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(1, num_tasks+1), result_bp['accuracy_per_task'], label='Backpropagation')
    plt.plot(np.arange(1, num_tasks+1), result_cbp['accuracy_per_task'], label='Continual Backpropagation')
    plt.xlabel('Task Number')
    plt.ylabel('Accuracy (%)')
    plt.title('Online Permuted MNIST: Accuracy Comparison')
    plt.grid(True)
    plt.legend()
    # plt.savefig('permuted_mnist_accuracy.png')
    
    # Plot weight magnitude comparison
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(1, num_tasks+1), result_bp['weight_magnitude_per_task'], label='Backpropagation')
    plt.plot(np.arange(1, num_tasks+1), result_cbp['weight_magnitude_per_task'], label='Continual Backpropagation')
    plt.xlabel('Task Number')
    plt.ylabel('Average Weight Magnitude')
    plt.title('Online Permuted MNIST: Weight Magnitude Comparison')
    plt.grid(True)
    plt.legend()
    # plt.savefig('permuted_mnist_weight_magnitude.png')
    
    # Average dead units across all layers
    dead_units_bp = np.mean(result_bp['dead_units_per_task'], axis=1)
    dead_units_cbp = np.mean(result_cbp['dead_units_per_task'], axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(1, num_tasks+1), dead_units_bp, label='Backpropagation')
    plt.plot(np.arange(1, num_tasks+1), dead_units_cbp, label='Continual Backpropagation')
    plt.xlabel('Task Number')
    plt.ylabel('Percent of Dead Units')
    plt.title('Online Permuted MNIST: Dead Units Comparison')
    plt.grid(True)
    plt.legend()
    # plt.savefig('permuted_mnist_dead_units.png')
    
    # Average effective rank across all layers
    effective_rank_bp = np.mean(result_bp['effective_rank_per_task'], axis=1)
    effective_rank_cbp = np.mean(result_cbp['effective_rank_per_task'], axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(1, num_tasks+1), effective_rank_bp, label='Backpropagation')
    plt.plot(np.arange(1, num_tasks+1), effective_rank_cbp, label='Continual Backpropagation')
    plt.xlabel('Task Number')
    plt.ylabel('Effective Rank')
    plt.title('Online Permuted MNIST: Effective Rank Comparison')
    plt.grid(True)
    plt.legend()
    # plt.savefig('permuted_mnist_effective_rank.png')
    
    print("Experiment complete.")

if __name__ == "__main__":
    main()