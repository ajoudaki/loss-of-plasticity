import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Subset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy import stats
import pandas as pd
from tqdm import tqdm
import os
from collections import defaultdict
import torchvision
import torchvision.transforms as transforms

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class MLP(nn.Module):
    """
    Multi-layer perceptron with configurable normalization layers
    """
    def __init__(self, 
                 input_dim=3072,  # 32x32x3 for CIFAR-10 flattened images
                 hidden_dims=[256, 128, 64], 
                 output_dim=10,
                 activation=nn.ReLU,
                 norm_type=None,  # 'bn', 'ln', None
                 affine=True,     # Whether to use learnable affine parameters in normalization
                 track_pre_activations=False):
        super(MLP, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.norm_type = norm_type
        self.affine = affine
        self.track_pre_activations = track_pre_activations
        
        # Create network
        layers = []
        prev_dim = input_dim
        
        # Hidden layers
        for i, dim in enumerate(hidden_dims):
            # Linear layer
            layers.append(nn.Linear(prev_dim, dim))
            
            # Normalization layer
            if norm_type == 'bn':
                layers.append(nn.BatchNorm1d(dim, affine=affine))
            elif norm_type == 'ln':
                layers.append(nn.LayerNorm(dim, elementwise_affine=affine))
            
            # Activation
            layers.append(activation())
            
            prev_dim = dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.layers = nn.ModuleList(layers)
        self.pre_activations = []  # For storing pre-activation values
        
    def forward(self, x):
        self.pre_activations = []  # Reset for each forward pass
        
        # Store each layer's output for rank analysis
        activations = []
        h = x
        
        layer_index = 0
        while layer_index < len(self.layers):
            # Apply linear layer
            linear_layer = self.layers[layer_index]
            h = linear_layer(h)
            
            # Store pre-activation (after linear, before norm and activation)
            if self.track_pre_activations:
                self.pre_activations.append(h.detach().clone())
            
            # Move to next layer (could be norm or activation)
            layer_index += 1
            
            # Apply normalization if present
            if layer_index < len(self.layers) and (
                isinstance(self.layers[layer_index], nn.BatchNorm1d) or 
                isinstance(self.layers[layer_index], nn.LayerNorm)):
                norm_layer = self.layers[layer_index]
                h = norm_layer(h)
                layer_index += 1
            
            # Apply activation if present
            if layer_index < len(self.layers) and (
                isinstance(self.layers[layer_index], nn.ReLU) or
                isinstance(self.layers[layer_index], nn.Tanh) or
                isinstance(self.layers[layer_index], nn.Sigmoid)):
                activation_layer = self.layers[layer_index]
                h = activation_layer(h)
                # Store post-activation
                activations.append(h.detach().clone())
                layer_index += 1
        
        return h, activations

def load_cifar10_subset(n_train=50000, n_val=10000):
    """
    Load a subset of the CIFAR-10 dataset with standardization
    as recommended for CIFAR-10
    """
    # Mean and std values for CIFAR-10 standardization
    cifar_mean = (0.4914, 0.4822, 0.4465)
    cifar_std = (0.2470, 0.2435, 0.2616)
    
    # Define the transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std)
    ])
    
    # Load CIFAR-10 dataset
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=True,
        download=True, 
        transform=transform
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=False,
        download=True, 
        transform=transform
    )
    
    # Take subsets if specified
    if n_train < len(train_dataset):
        train_indices = np.random.choice(len(train_dataset), n_train, replace=False)
        train_dataset = Subset(train_dataset, train_indices)
    
    if n_val < len(test_dataset):
        val_indices = np.random.choice(len(test_dataset), n_val, replace=False)
        test_dataset = Subset(test_dataset, val_indices)
    
    return train_dataset, test_dataset

def compute_effective_rank(matrix):
    """
    Compute the effective rank of a matrix using singular values
    """
    # Convert to numpy if it's a torch tensor
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.cpu().numpy()
    
    # Compute SVD
    s = np.linalg.svd(matrix, compute_uv=False)
    
    # Normalize singular values
    s_norm = s / np.sum(s)
    
    # Remove zeros to avoid log(0)
    s_norm = s_norm[s_norm > 1e-10]
    
    # Compute entropy
    entropy = -np.sum(s_norm * np.log(s_norm))
    
    # Effective rank
    return np.exp(entropy)

def test_gaussianity(data, alpha=0.05):
    """
    Test if each feature follows a Gaussian distribution
    Returns the percentage of features that pass the test
    """
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    
    n_features = data.shape[1]
    gaussian_count = 0
    
    for i in range(n_features):
        feature = data[:, i]
        # Use Shapiro-Wilk test for normality
        # Taking a random sample of 5000 points max (Shapiro-Wilk works best with smaller samples)
        if len(feature) > 5000:
            feature = np.random.choice(feature, size=5000, replace=False)
        _, p_value = stats.shapiro(feature)
        
        if p_value > alpha:
            gaussian_count += 1
    
    return gaussian_count / n_features

def test_distribution_properties(data, alpha=0.05):
    """
    Test distribution properties of each feature
    
    Returns:
    - gaussianity: percentage of features following a Gaussian distribution
    - standard_gaussianity: percentage of features following a standard Gaussian (mean≈0, std≈1)
    - mean_deviation: average absolute deviation from zero mean
    - std_deviation: average absolute deviation from unit std
    """
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    
    n_features = data.shape[1]
    gaussian_count = 0
    standard_gaussian_count = 0
    mean_devs = []
    std_devs = []
    
    for i in range(n_features):
        feature = data[:, i]
        
        # Calculate mean and std
        mean = np.mean(feature)
        std = np.std(feature)
        mean_devs.append(abs(mean))
        std_devs.append(abs(std - 1.0))
        
        # Shapiro-Wilk test for normality
        if len(feature) > 5000:
            sampled_feature = np.random.choice(feature, size=5000, replace=False)
            _, p_value = stats.shapiro(sampled_feature)
        else:
            _, p_value = stats.shapiro(feature)
        
        # Check if Gaussian
        if p_value > alpha:
            gaussian_count += 1
            
            # Check if standard Gaussian (mean close to 0, std close to 1)
            if abs(mean) < 0.1 and abs(std - 1.0) < 0.1:
                standard_gaussian_count += 1
    
    results = {
        'gaussianity': gaussian_count / n_features,
        'standard_gaussianity': standard_gaussian_count / n_features,
        'mean_deviation': np.mean(mean_devs),
        'std_deviation': np.mean(std_devs)
    }
    
    return results

def plot_feature_distributions(data, layer_name, norm_type, n_features=5):
    """
    Plot histograms of a few random features to visually inspect their distributions
    """
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    
    # Select random features
    feature_indices = np.random.choice(data.shape[1], size=min(n_features, data.shape[1]), replace=False)
    
    fig, axes = plt.subplots(1, len(feature_indices), figsize=(15, 3))
    if len(feature_indices) == 1:
        axes = [axes]
    
    for i, idx in enumerate(feature_indices):
        feature = data[:, idx]
        ax = axes[i]
        
        # Plot histogram
        ax.hist(feature, bins=30, alpha=0.7, density=True)
        
        # Plot normal distribution for comparison
        mu, std = np.mean(feature), np.std(feature)
        x = np.linspace(mu - 3*std, mu + 3*std, 100)
        ax.plot(x, stats.norm.pdf(x, mu, std), 'r-', alpha=0.7)
        
        # Shapiro-Wilk test
        if len(feature) > 5000:
            sampled_feature = np.random.choice(feature, size=5000, replace=False)
            _, p_value = stats.shapiro(sampled_feature)
        else:
            _, p_value = stats.shapiro(feature)
        
        ax.set_title(f"Feature {idx}\np-value: {p_value:.3f}")
        
    plt.suptitle(f"{layer_name} - {norm_type}")
    plt.tight_layout()
    return fig

def train_model(model, train_loader, val_loader, epochs=5):
    """
    Train the model and return training metrics
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Flatten the inputs for MLP
            inputs = inputs.view(inputs.size(0), -1)
            
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Flatten the inputs for MLP
                inputs = inputs.view(inputs.size(0), -1)
                
                outputs, _ = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    return {'train_losses': train_losses, 'val_losses': val_losses}

def analyze_layer_activations(model, data_loader, norm_type, affine, save_path):
    """
    Analyze pre and post activations of each layer in the model
    """
    model.eval()
    model.track_pre_activations = True
    
    # Collect pre and post activations
    all_pre_activations = []
    all_post_activations = []
    
    # First, do a single forward pass to determine the structure
    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            # Flatten the inputs for MLP
            inputs = inputs.view(inputs.size(0), -1)
            _, activations = model(inputs)
            break
    
    # Initialize lists with the correct number of layers
    all_pre_activations = [[] for _ in range(len(model.pre_activations))]
    all_post_activations = [[] for _ in range(len(activations))]
    
    # Now collect all activations
    with torch.no_grad():
        for inputs, _ in tqdm(data_loader, desc="Collecting activations"):
            inputs = inputs.to(device)
            # Flatten the inputs for MLP
            inputs = inputs.view(inputs.size(0), -1)
            _, activations = model(inputs)
            
            # Store pre activations
            for i, pre_act in enumerate(model.pre_activations):
                all_pre_activations[i].append(pre_act)
            
            # Store post activations
            for i, post_act in enumerate(activations):
                all_post_activations[i].append(post_act)
    
    # Concatenate batches
    all_pre_activations = [torch.cat(pre_acts, dim=0) for pre_acts in all_pre_activations]
    all_post_activations = [torch.cat(post_acts, dim=0) for post_acts in all_post_activations]
    
    # Analyze each layer
    results = []
    
    # Ensure we only analyze layers where we have both pre and post activations
    n_layers = min(len(all_pre_activations), len(all_post_activations))
    
    for layer_idx in range(n_layers):
        pre_act = all_pre_activations[layer_idx]
        post_act = all_post_activations[layer_idx]
        
        # Compute covariance matrices
        pre_cov = np.cov(pre_act.cpu().numpy(), rowvar=False)
        post_cov = np.cov(post_act.cpu().numpy(), rowvar=False)
        
        # Compute effective ranks
        pre_rank = compute_effective_rank(pre_cov)
        post_rank = compute_effective_rank(post_cov)
        rank_improvement = post_rank - pre_rank
        
        # Test Gaussianity
        pre_gaussianity = test_gaussianity(pre_act)
        post_gaussianity = test_gaussianity(post_act)
        
        # Layer info
        layer_info = {
            'layer_idx': layer_idx,
            'pre_rank': pre_rank,
            'post_rank': post_rank,
            'rank_improvement': rank_improvement,
            'pre_gaussianity': pre_gaussianity,
            'post_gaussianity': post_gaussianity,
            'norm_type': norm_type,
            'affine': affine
        }
        
        results.append(layer_info)
        
        # Create and save visualization plots
        plot_dir = f"{save_path}/norm_{norm_type}_affine_{affine}"
        os.makedirs(plot_dir, exist_ok=True)
        
        # Plot feature distributions
        pre_fig = plot_feature_distributions(
            pre_act, f"Layer {layer_idx} Pre-Activation", 
            f"{norm_type}, affine={affine}"
        )
        pre_fig.savefig(f"{plot_dir}/layer_{layer_idx}_pre_dist.png")
        plt.close(pre_fig)
        
        post_fig = plot_feature_distributions(
            post_act, f"Layer {layer_idx} Post-Activation", 
            f"{norm_type}, affine={affine}"
        )
        post_fig.savefig(f"{plot_dir}/layer_{layer_idx}_post_dist.png")
        plt.close(post_fig)
        
        # Plot covariance matrices
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        sns.heatmap(pre_cov, ax=axes[0], cmap='coolwarm')
        axes[0].set_title(f"Pre-Activation Covariance\nEff Rank: {pre_rank:.2f}")
        
        sns.heatmap(post_cov, ax=axes[1], cmap='coolwarm')
        axes[1].set_title(f"Post-Activation Covariance\nEff Rank: {post_rank:.2f}")
        
        plt.suptitle(f"Layer {layer_idx}, {norm_type}, affine={affine}")
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/layer_{layer_idx}_covariance.png")
        plt.close()
    
    return results

def run_experiment(input_dim=3072, hidden_dims=[128, 64, 32], output_dim=10, 
                  activation=nn.ReLU, batch_size=128, epochs=5, save_path="../outputs/rank_preservation_cifar10"):
    """
    Run the experiment for different normalization configurations using CIFAR-10
    """
    # Load CIFAR-10 data
    print("Loading CIFAR-10 dataset...")
    train_dataset, val_dataset = load_cifar10_subset(n_train=50000, n_val=10000)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Normalization configurations to test
    configs = [
        {'norm_type': None, 'affine': False, 'name': 'No Normalization'},
        {'norm_type': 'bn', 'affine': False, 'name': 'BN without affine'},
        {'norm_type': 'bn', 'affine': True, 'name': 'BN with affine'},
        {'norm_type': 'ln', 'affine': False, 'name': 'LN without affine'},
        {'norm_type': 'ln', 'affine': True, 'name': 'LN with affine'}
    ]
    
    all_results = []
    
    for config in configs:
        print(f"\n{'='*50}")
        print(f"Testing {config['name']}")
        print(f"{'='*50}")
        
        # Create model
        model = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            activation=activation,
            norm_type=config['norm_type'],
            affine=config['affine']
        ).to(device)
        
        # Train model
        train_metrics = train_model(model, train_loader, val_loader, epochs=epochs)
        
        # Analyze activations
        layer_results = analyze_layer_activations(
            model, val_loader, config['norm_type'], config['affine'], save_path
        )
        
        # Add to all results
        for res in layer_results:
            res['config_name'] = config['name']
            all_results.append(res)
    
    return all_results

def plot_combined_results(results, save_path="../outputs/rank_preservation_cifar10"):
    """
    Create summary plots comparing different normalization strategies
    """
    # Convert results to DataFrame for easier analysis
    df = pd.DataFrame(results)
    
    # Create plots directory
    os.makedirs(f"{save_path}/summary", exist_ok=True)
    
    # 1. Rank improvement by layer and normalization type
    plt.figure(figsize=(12, 8))
    sns.barplot(x='layer_idx', y='rank_improvement', hue='config_name', data=df)
    plt.title('Rank Improvement (Post - Pre) by Layer and Normalization')
    plt.xlabel('Layer Index')
    plt.ylabel('Rank Improvement')
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(f'{save_path}/summary/rank_improvement.png')
    plt.close()
    
    # 2. Pre-activation Gaussianity by layer and normalization type
    plt.figure(figsize=(12, 8))
    sns.barplot(x='layer_idx', y='pre_gaussianity', hue='config_name', data=df)
    plt.title('Pre-Activation Gaussianity by Layer and Normalization')
    plt.xlabel('Layer Index')
    plt.ylabel('Fraction of Features with Gaussian Distribution')
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(f'{save_path}/summary/pre_gaussianity.png')
    plt.close()
    
    # 3. Correlation: Gaussianity vs Rank Improvement
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='pre_gaussianity', y='rank_improvement', 
                   hue='config_name', style='layer_idx', s=100, data=df)
    plt.title('Correlation: Pre-Activation Gaussianity vs Rank Improvement')
    plt.xlabel('Pre-Activation Gaussianity')
    plt.ylabel('Rank Improvement')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{save_path}/summary/gaussianity_vs_improvement.png')
    plt.close()
    
    # 4. Pre and Post effective rank by layer and normalization
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    sns.barplot(x='layer_idx', y='pre_rank', hue='config_name', ax=axes[0], data=df)
    axes[0].set_title('Pre-Activation Effective Rank')
    axes[0].set_xlabel('Layer Index')
    axes[0].set_ylabel('Effective Rank')
    axes[0].grid(True, axis='y')
    
    sns.barplot(x='layer_idx', y='post_rank', hue='config_name', ax=axes[1], data=df)
    axes[1].set_title('Post-Activation Effective Rank')
    axes[1].set_xlabel('Layer Index')
    axes[1].set_ylabel('Effective Rank')
    axes[1].grid(True, axis='y')
    
    plt.suptitle('Effective Rank by Layer and Normalization', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{save_path}/summary/effective_ranks.png')
    plt.close()
    
    # 5. Average results table
    avg_results = df.groupby('config_name').agg({
        'pre_gaussianity': 'mean',
        'post_gaussianity': 'mean',
        'pre_rank': 'mean',
        'post_rank': 'mean',
        'rank_improvement': 'mean'
    }).reset_index()
    
    print("\nAverage results across all layers:")
    print(avg_results)
    
    # Save as CSV
    avg_results.to_csv(f'{save_path}/summary/average_results.csv', index=False)
    
    return avg_results

if __name__ == "__main__":
    # Create output directories
    save_path = "../outputs/rank_preservation_cifar10"
    os.makedirs(f"{save_path}", exist_ok=True)
    
    # Run experiments
    print("Starting normalization experiments with CIFAR-10...")
    all_results = run_experiment(
        input_dim=3072,  # 32x32x3 for CIFAR-10 flattened images
        hidden_dims=[128,]*5,
        output_dim=10,
        activation=nn.ReLU,
        batch_size=128,
        epochs=3,  # Fewer epochs for faster experiment
        save_path=save_path
    )
    
    # Plot summary results
    avg_results = plot_combined_results(all_results, save_path=save_path)
    
    print(f"Experiment completed! Results and visualizations are saved in the '{save_path}' directory.")