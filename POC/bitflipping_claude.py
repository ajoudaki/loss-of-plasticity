import torch
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F
from torch.optim import SGD, Adam
import argparse
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# Ensure reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create figures directory if it doesn't exist
os.makedirs("figures", exist_ok=True)

class TargetLTUNetwork(nn.Module):
    """Target network with Linear Threshold Units"""
    def __init__(self, input_size, hidden_size, beta):
        super(TargetLTUNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.beta = beta
        
        # Initialize weights to be +1 or -1
        self.weights = torch.randint(0, 2, (hidden_size, input_size), device=device).float() * 2 - 1
        self.output_weights = torch.randint(0, 2, (1, hidden_size), device=device).float() * 2 - 1
        
        # Calculate S_i: number of input weights with negative value for each hidden unit
        self.S = torch.sum(self.weights < 0, dim=1)
        # Calculate thresholds: θ_i = (m+1)·β - S_i
        self.thresholds = (input_size * beta) - self.S

    def forward(self, x):
        # Calculate weighted sum for each hidden unit
        hidden_activations = torch.matmul(x, self.weights.t())
        # Apply threshold activation function (LTU)
        hidden_outputs = (hidden_activations > self.thresholds).float()
        # Calculate output
        output = torch.matmul(hidden_outputs, self.output_weights.t())
        return output

class LinearModel(nn.Module):
    """Linear baseline model"""
    def __init__(self, input_size):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)
        
    def forward(self, x):
        return self.linear(x)

class MLPModel(nn.Module):
    """MLP learning model with configurable activation function"""
    def __init__(self, input_size, hidden_size, activation='relu'):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        
        # Set activation function
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid
        elif activation == 'elu':
            self.activation = F.elu
        elif activation == 'leaky_relu':
            self.activation = F.leaky_relu
        elif activation == 'swish':
            self.activation = lambda x: x * torch.sigmoid(x)
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
        
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x

def run_faster_experiment(
    m=21, 
    f=15, 
    T=10000, 
    n_target=100, 
    n_hidden=5, 
    total_examples=1000000, 
    bin_size=40000, 
    beta=0.7, 
    activation_funcs=None, 
    learning_rates=None, 
    optimizer_type='sgd', 
    run_linear=True
):
    """
    Optimized implementation that runs multiple experiments in parallel batches
    """
    if activation_funcs is None:
        activation_funcs = ['relu', 'tanh']
    if learning_rates is None:
        learning_rates = [0.01, 0.001]

    # Create target network
    target_net = TargetLTUNetwork(m + 1, n_target, beta)
    target_net.to(device)
    target_net.eval()  # Target network is fixed
    
    # Create models dictionary
    models = {}
    optimizers = {}
    
    # Initialize models and optimizers for each activation and learning rate
    for activation in activation_funcs:
        for lr in learning_rates:
            model_key = f"{activation}_{lr}"
            models[model_key] = MLPModel(m + 1, n_hidden, activation)
            models[model_key].to(device)
            
            if optimizer_type == 'sgd':
                optimizers[model_key] = SGD(models[model_key].parameters(), lr=lr)
            else:  # adam
                optimizers[model_key] = Adam(models[model_key].parameters(), lr=lr)
    
    # Create linear baseline if needed
    if run_linear:
        linear_model = LinearModel(m + 1)
        linear_model.to(device)
        if optimizer_type == 'sgd':
            linear_optimizer = SGD(linear_model.parameters(), lr=0.01)  # Fixed learning rate for linear
        else:
            linear_optimizer = Adam(linear_model.parameters(), lr=0.01)
    
    # Initialize fixed bits (first f bits that change slowly)
    fixed_bits = torch.bernoulli(torch.ones(f, device=device) * 0.5)
    
    # Create random number generator for reproducibility
    rng = np.random.RandomState(42)
    
    # Initialize error tracking
    num_bins = total_examples // bin_size + (1 if total_examples % bin_size > 0 else 0)
    errors = {model_key: np.zeros(num_bins) for model_key in models.keys()}
    if run_linear:
        linear_errors = np.zeros(num_bins)
    
    current_bin_errors = {model_key: 0.0 for model_key in models.keys()}
    if run_linear:
        current_bin_linear_error = 0.0
    
    examples_in_bin = 0
    current_bin = 0
    
    # Generate all inputs in advance (more efficient)
    print("Preparing experiment data...")
    
    # We'll generate inputs for each bit-flipping period
    num_periods = total_examples // T + 1
    inputs_per_period = min(T, total_examples)
    
    # Initialize bit flip schedule
    bit_flip_schedule = []
    current_fixed_bits = fixed_bits.clone()
    
    # For each period, determine which bit to flip
    for period in range(1, num_periods):
        bit_to_flip = rng.randint(0, f)
        bit_flip_schedule.append(bit_to_flip)
    
    # Main training loop
    print(f"Running experiment with {len(activation_funcs)} activations, {len(learning_rates)} learning rates")
    examples_processed = 0
    
    # Process data in chunks to avoid memory issues
    chunk_size = min(10000, T)  # Process data in chunks
    num_chunks = total_examples // chunk_size + (1 if total_examples % chunk_size > 0 else 0)
    
    for chunk in tqdm(range(num_chunks)):
        chunk_start = chunk * chunk_size
        chunk_end = min((chunk + 1) * chunk_size, total_examples)
        chunk_examples = chunk_end - chunk_start
        
        # Generate inputs and targets for this chunk
        inputs = torch.zeros(chunk_examples, m + 1, device=device)
        targets = torch.zeros(chunk_examples, 1, device=device)
        
        current_fixed_bits = fixed_bits.clone()
        
        for i in range(chunk_examples):
            global_i = chunk_start + i
            
            # Check if it's time to flip a bit
            if global_i > 0 and global_i % T == 0:
                period = global_i // T
                bit_to_flip = bit_flip_schedule[period-1]
                current_fixed_bits[bit_to_flip] = 1 - current_fixed_bits[bit_to_flip]
            
            # Set the first f bits to the current fixed bits
            inputs[i, :f] = current_fixed_bits
            
            # Set the next m-f bits randomly
            inputs[i, f:m] = torch.bernoulli(torch.ones(m-f, device=device) * 0.5)
            
            # Set the bias term to 1
            inputs[i, m] = 1.0
        
        # Get targets from target network (batch process)
        with torch.no_grad():
            targets = target_net(inputs)
        
        # Process the chunk in mini-batches
        mini_batch_size = 32
        num_mini_batches = chunk_examples // mini_batch_size + (1 if chunk_examples % mini_batch_size > 0 else 0)
        
        for mini_batch in range(num_mini_batches):
            mb_start = mini_batch * mini_batch_size
            mb_end = min((mini_batch + 1) * mini_batch_size, chunk_examples)
            
            if mb_start >= chunk_examples:
                break
                
            mb_inputs = inputs[mb_start:mb_end]
            mb_targets = targets[mb_start:mb_end]
            
            # Train each model
            for model_key, model in models.items():
                optimizer = optimizers[model_key]
                
                optimizer.zero_grad()
                outputs = model(mb_inputs)
                loss = F.mse_loss(outputs, mb_targets)
                loss.backward()
                optimizer.step()
                
                # Track error
                current_bin_errors[model_key] += loss.item() * (mb_end - mb_start)
            
            # Train linear model
            if run_linear:
                linear_optimizer.zero_grad()
                linear_outputs = linear_model(mb_inputs)
                linear_loss = F.mse_loss(linear_outputs, mb_targets)
                linear_loss.backward()
                linear_optimizer.step()
                
                # Track error
                current_bin_linear_error += linear_loss.item() * (mb_end - mb_start)
            
            examples_processed += (mb_end - mb_start)
            examples_in_bin += (mb_end - mb_start)
            
            # Check if we've completed a bin
            if examples_in_bin >= bin_size:
                for model_key in models.keys():
                    errors[model_key][current_bin] = current_bin_errors[model_key] / examples_in_bin
                    current_bin_errors[model_key] = 0.0
                
                if run_linear:
                    linear_errors[current_bin] = current_bin_linear_error / examples_in_bin
                    current_bin_linear_error = 0.0
                
                current_bin += 1
                examples_in_bin = 0
    
    # Add any remaining examples to the final bin
    if examples_in_bin > 0:
        for model_key in models.keys():
            errors[model_key][current_bin] = current_bin_errors[model_key] / examples_in_bin
        
        if run_linear:
            linear_errors[current_bin] = current_bin_linear_error / examples_in_bin
    
    # Format results
    results = {}
    for activation in activation_funcs:
        for lr in learning_rates:
            model_key = f"{activation}_{lr}"
            if run_linear:
                results[(activation, lr)] = (errors[model_key], linear_errors)
            else:
                results[(activation, lr)] = (errors[model_key], None)
    
    return results

def plot_results(results, activations, learning_rates, optimizer_type, bin_size, total_examples):
    """
    Plot the results of multiple experiments
    """
    plt.figure(figsize=(10, 6))
    
    # Plot each neural network result
    for activation in activations:
        for lr in learning_rates:
            label = f"{activation}, lr={lr}"
            mlp_errors, linear_errors = results[(activation, lr)]
            
            # Convert bin indices to example counts
            x_axis = np.arange(len(mlp_errors)) * bin_size
            
            # Plot MLP errors
            plt.plot(x_axis, mlp_errors, label=label)
    
    # Plot linear baseline (using the first experiment's linear result)
    activation, lr = list(results.keys())[0]
    _, linear_errors = results[(activation, lr)]
    plt.plot(x_axis, linear_errors, 'k--', label='Linear Baseline', linewidth=2)
    
    # Add labels and legend
    plt.xlabel('Examples')
    plt.ylabel('Mean Squared Error')
    plt.title(f'Loss of Plasticity with {optimizer_type.upper()} Optimizer')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save figure
    filename = f"figures/loss_of_plasticity_{optimizer_type}.pdf"
    plt.savefig(filename, format='pdf', bbox_inches='tight')
    print(f"Figure saved to {filename}")
    
    # Also create a figure focusing on the comparison with linear baseline
    plt.figure(figsize=(10, 6))
    
    # Choose a representative neural network result (first activation with lr=0.01)
    for activation in activations:
        if (activation, 0.01) in results:
            mlp_errors, linear_errors = results[(activation, 0.01)]
            plt.plot(x_axis, mlp_errors, label=f"Neural Network ({activation})")
            break
    
    plt.plot(x_axis, linear_errors, 'k--', label='Linear Baseline', linewidth=2)
    
    # Add horizontal lines showing when neural network becomes worse than linear
    crossover_points = np.where(mlp_errors > linear_errors)[0]
    if len(crossover_points) > 0:
        first_crossover = crossover_points[0]
        plt.axvline(x=first_crossover * bin_size, color='r', linestyle='--', 
                   label=f'First crossover at {first_crossover * bin_size} examples')
    
    plt.xlabel('Examples')
    plt.ylabel('Mean Squared Error')
    plt.title('Neural Network vs Linear Baseline')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save figure
    filename = f"figures/nn_vs_linear_{optimizer_type}.pdf"
    plt.savefig(filename, format='pdf', bbox_inches='tight')
    print(f"Figure saved to {filename}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Optimized Slowly Changing Regression Experiment')
    parser.add_argument('--m', type=int, default=21, help='Number of input bits')
    parser.add_argument('--f', type=int, default=15, help='Number of flipping bits')
    parser.add_argument('--T', type=int, default=10000, help='Duration between bit flips')
    parser.add_argument('--n_target', type=int, default=100, help='Number of hidden units in target network')
    parser.add_argument('--n_hidden', type=int, default=5, help='Number of hidden units in learning network')
    parser.add_argument('--total_examples', type=int, default=1000000, help='Total examples to run')
    parser.add_argument('--beta', type=float, default=0.7, help='Proportion used in LTU Threshold')
    parser.add_argument('--bin_size', type=int, default=40000, help='Bin size for error reporting')
    args = parser.parse_args()
    
    start_time = time.time()
    
    # Run the optimized experiment
    print("\nRunning experiment...")
    
    # Set experimental parameters
    activations = ['relu', 'tanh']
    learning_rates = [0.01, 0.001]
    
    # Run SGD experiment
    sgd_results = run_faster_experiment(
        m=args.m,
        f=args.f,
        T=args.T,
        n_target=args.n_target,
        n_hidden=args.n_hidden,
        total_examples=args.total_examples,
        bin_size=args.bin_size,
        beta=args.beta,
        activation_funcs=activations,
        learning_rates=learning_rates,
        optimizer_type='sgd'
    )
    
    # Plot SGD results
    plot_results(sgd_results, activations, learning_rates, 'sgd', args.bin_size, args.total_examples)
    
    # Run Adam experiment
    adam_results = run_faster_experiment(
        m=args.m,
        f=args.f,
        T=args.T,
        n_target=args.n_target,
        n_hidden=args.n_hidden,
        total_examples=args.total_examples,
        bin_size=args.bin_size,
        beta=args.beta,
        activation_funcs=activations,
        learning_rates=learning_rates,
        optimizer_type='adam'
    )
    
    # Plot Adam results
    plot_results(adam_results, activations, learning_rates, 'adam', args.bin_size, args.total_examples)
    
    # Run an experiment to show effect of network size
    hidden_sizes = [5, 20, 50]
    network_size_results = {}
    
    for n_hidden in hidden_sizes:
        # Only run with ReLU and fixed learning rate to save time
        size_results = run_faster_experiment(
            m=args.m,
            f=args.f,
            T=args.T,
            n_target=args.n_target,
            n_hidden=n_hidden,
            total_examples=args.total_examples,
            bin_size=args.bin_size,
            beta=args.beta,
            activation_funcs=['relu'],
            learning_rates=[0.01],
            optimizer_type='sgd'
        )
        network_size_results[(f"hidden={n_hidden}", 0.01)] = size_results[('relu', 0.01)]
    
    # Plot network size comparison
    plt.figure(figsize=(10, 6))
    
    # Get reference to linear baseline from first result
    _, linear_errors = list(network_size_results.values())[0]
    
    # Plot each neural network result
    x_axis = np.arange(len(linear_errors)) * args.bin_size
    
    for (config, _), (mlp_errors, _) in network_size_results.items():
        plt.plot(x_axis, mlp_errors, label=config)
    
    # Plot linear baseline
    plt.plot(x_axis, linear_errors, 'k--', label='Linear Baseline', linewidth=2)
    
    plt.xlabel('Examples')
    plt.ylabel('Mean Squared Error')
    plt.title('Effect of Network Size on Loss of Plasticity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save figure
    filename = "figures/network_size_effect.pdf"
    plt.savefig(filename, format='pdf', bbox_inches='tight')
    print(f"Figure saved to {filename}")
    
    # Create a comparison figure showing all activations in a single plot
    plt.figure(figsize=(12, 8))
    
    # Standard activation functions
    all_activations = ['relu', 'tanh', 'sigmoid', 'elu', 'leaky_relu', 'swish']
    standard_lr = 0.01
    
    # Run experiment with all activations
    all_activation_results = run_faster_experiment(
        m=args.m,
        f=args.f,
        T=args.T,
        n_target=args.n_target,
        n_hidden=args.n_hidden,
        total_examples=args.total_examples,
        bin_size=args.bin_size,
        beta=args.beta,
        activation_funcs=all_activations,
        learning_rates=[standard_lr],
        optimizer_type='sgd'
    )
    
    # Plot each neural network result
    for activation in all_activations:
        mlp_errors, linear_errors = all_activation_results[(activation, standard_lr)]
        plt.plot(x_axis, mlp_errors, label=f"{activation}")
    
    # Plot linear baseline
    plt.plot(x_axis, linear_errors, 'k--', label='Linear Baseline', linewidth=2)
    
    plt.xlabel('Examples')
    plt.ylabel('Mean Squared Error')
    plt.title('Loss of Plasticity Across Different Activation Functions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save figure
    filename = "figures/all_activations_comparison.pdf"
    plt.savefig(filename, format='pdf', bbox_inches='tight')
    print(f"Figure saved to {filename}")
    
    elapsed_time = time.time() - start_time
    print(f"Total experiment time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()