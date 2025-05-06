import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import time
from typing import List, Tuple, Callable, Dict, Optional

# Set random seeds for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

class LTUActivation(torch.autograd.Function):
    """
    Linear Threshold Unit activation function as described in the paper.
    Output is 1 if input is above threshold, 0 otherwise.
    """
    @staticmethod
    def forward(ctx, input, threshold):
        ctx.save_for_backward(input, threshold)
        return (input > threshold).float()
    
    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator for backward pass
        input, threshold = ctx.saved_tensors
        grad_input = grad_output.clone()
        # Zero gradient where input is far from threshold
        grad_input = grad_input * (torch.abs(input - threshold) < 0.1).float()
        return grad_input, None

class TargetNetwork(nn.Module):
    """
    Target network with LTU activation as described in the paper.
    This network generates the target outputs for the Slowly-Changing Regression problem.
    """
    def __init__(self, input_size: int, hidden_size: int, beta: float = 0.7):
        super(TargetNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.beta = beta
        
        # Initialize weights as -1 or 1 as specified in the paper
        self.weights = torch.randint(0, 2, (hidden_size, input_size)).float() * 2 - 1
        self.output_weights = torch.randint(0, 2, (1, hidden_size)).float() * 2 - 1
        
        # Calculate thresholds for LTU units
        self.S = torch.sum((self.weights < 0).float(), dim=1)
        self.thresholds = (input_size * beta) - self.S
    
    def forward(self, x):
        # First layer with LTU activation
        pre_activation = torch.matmul(x, self.weights.t())
        hidden = torch.zeros_like(pre_activation)
        for i in range(self.hidden_size):
            hidden[:, i] = LTUActivation.apply(pre_activation[:, i], self.thresholds[i])
        
        # Output layer (linear)
        output = torch.matmul(hidden, self.output_weights.t())
        return output

class LearningNetwork(nn.Module):
    """
    Learning network that tries to approximate the target function.
    """
    def __init__(self, input_size: int, hidden_size: int, activation: str = 'relu'):
        super(LearningNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation_name = activation
        
        # Initialize layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        
        # Initialize weights using Kaiming initialization
        if activation == 'relu':
            gain = nn.init.calculate_gain('relu')
            nn.init.kaiming_uniform_(self.fc1.weight, a=0, mode='fan_in', nonlinearity='relu')
        elif activation == 'tanh':
            gain = nn.init.calculate_gain('tanh')
            nn.init.kaiming_uniform_(self.fc1.weight, a=0, mode='fan_in', nonlinearity='tanh')
        elif activation == 'sigmoid':
            gain = 1.0
            nn.init.kaiming_uniform_(self.fc1.weight, a=0, mode='fan_in', nonlinearity='sigmoid')
        elif activation == 'leaky_relu':
            gain = nn.init.calculate_gain('leaky_relu', 0.01)
            nn.init.kaiming_uniform_(self.fc1.weight, a=0.01, mode='fan_in', nonlinearity='leaky_relu')
        elif activation == 'elu':
            gain = nn.init.calculate_gain('relu')  # Approximation
            nn.init.kaiming_uniform_(self.fc1.weight, a=0, mode='fan_in', nonlinearity='relu')
        elif activation == 'swish':
            gain = nn.init.calculate_gain('relu')  # Approximation
            nn.init.kaiming_uniform_(self.fc1.weight, a=0, mode='fan_in', nonlinearity='relu')
        else:
            raise ValueError(f"Activation function {activation} not supported")
        
        nn.init.kaiming_uniform_(self.fc2.weight, a=0, mode='fan_in', nonlinearity='linear')
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
    
    def forward(self, x):
        # Apply first layer with specified activation
        if self.activation_name == 'relu':
            h = torch.relu(self.fc1(x))
        elif self.activation_name == 'tanh':
            h = torch.tanh(self.fc1(x))
        elif self.activation_name == 'sigmoid':
            h = torch.sigmoid(self.fc1(x))
        elif self.activation_name == 'leaky_relu':
            h = torch.nn.functional.leaky_relu(self.fc1(x), negative_slope=0.01)
        elif self.activation_name == 'elu':
            h = torch.nn.functional.elu(self.fc1(x))
        elif self.activation_name == 'swish':
            h = self.fc1(x) * torch.sigmoid(self.fc1(x))  # Swish activation
        
        # Apply output layer (linear)
        output = self.fc2(h)
        return output

def generate_bit_flipping_data(
    m: int,
    f: int,
    T: int,
    n_examples: int
) -> Tuple[torch.Tensor, List[int]]:
    """
    Generate data for the Slowly-Changing Regression problem.
    
    Args:
        m: Total number of input bits (excluding bias)
        f: Number of flipping (slow-changing) bits
        T: Duration between bit flips
        n_examples: Total number of examples to generate
    
    Returns:
        Tuple containing:
            - Tensor of input data (n_examples, m+1)
            - List of indices where flips occurred
    """
    # Initialize inputs
    inputs = torch.zeros(n_examples, m + 1)
    
    # Initialize the flipping bits (first f bits)
    flipping_bits = torch.randint(0, 2, (f,)).float()
    
    # Track where flips occur
    flip_indices = []
    
    for i in range(n_examples):
        # Check if it's time to flip a bit
        if i % T == 0 and i > 0:
            flip_idx = random.randint(0, f - 1)
            flipping_bits[flip_idx] = 1 - flipping_bits[flip_idx]
            flip_indices.append(i)
        
        # Set the flipping bits
        inputs[i, :f] = flipping_bits
        
        # Set the random bits
        inputs[i, f:m] = torch.randint(0, 2, (m - f,)).float()
        
        # Set the bias bit (always 1)
        inputs[i, m] = 1.0
    
    return inputs, flip_indices

def run_slowly_changing_regression_experiment(
    m: int = 21,
    f: int = 15,
    hidden_size_target: int = 100,
    hidden_size_learner: int = 5,
    T: int = 10000,
    n_examples: int = 3000000,
    beta: float = 0.7,
    activation: str = 'relu',
    step_size: float = 0.01,
    bin_size: int = 40000,
    use_adam: bool = False,
    weight_decay: float = 0.0,
    use_shrink_perturb: bool = False,
    noise_variance: float = 0.0,
    continual_backprop: bool = False,
    rho: float = 0.0,
    maturity_threshold: int = 100,
):
    """
    Run the Slowly-Changing Regression experiment.
    
    Args:
        m: Number of input bits (excluding bias)
        f: Number of flipping bits
        hidden_size_target: Number of hidden units in the target network
        hidden_size_learner: Number of hidden units in the learning network
        T: Duration between bit flips
        n_examples: Total number of examples
        beta: Parameter for LTU threshold calculation
        activation: Activation function for the learning network
        step_size: Learning rate
        bin_size: Number of examples per bin for error calculation
        use_adam: Whether to use Adam optimizer
        weight_decay: L2 regularization coefficient
        use_shrink_perturb: Whether to use Shrink-and-Perturb
        noise_variance: Variance of the noise for Shrink-and-Perturb
        continual_backprop: Whether to use Continual Backpropagation
        rho: Replacement rate for Continual Backpropagation
        maturity_threshold: Maturity threshold for Continual Backpropagation
    
    Returns:
        Dictionary containing experiment results
    """
    # Create target network
    target_net = TargetNetwork(m + 1, hidden_size_target, beta)
    
    # Create learning network
    learning_net = LearningNetwork(m + 1, hidden_size_learner, activation)
    
    # Set up optimizer
    if use_adam:
        optimizer = optim.Adam(learning_net.parameters(), lr=step_size, weight_decay=weight_decay)
    else:
        optimizer = optim.SGD(learning_net.parameters(), lr=step_size, weight_decay=weight_decay)
    
    # Set up loss function
    criterion = nn.MSELoss()
    
    # Generate data
    print("Generating data...")
    inputs, flip_indices = generate_bit_flipping_data(m, f, T, n_examples)
    
    # Calculate targets using the target network
    with torch.no_grad():
        targets = target_net(inputs)
    
    # Arrays to store results
    n_bins = n_examples // bin_size
    squared_errors = np.zeros(n_bins)
    
    # Utility tracking for continual backpropagation
    if continual_backprop:
        # Initialize utility tracking
        fc1_utility = torch.zeros(hidden_size_learner)
        fc1_activation_avg = torch.zeros(hidden_size_learner)
        fc1_age = torch.zeros(hidden_size_learner)
        decay_rate = 0.99  # Decay rate for running averages
    
    # Training loop
    print("Starting training...")
    for i in tqdm(range(n_examples)):
        # Get current example
        x = inputs[i].unsqueeze(0)
        y = targets[i].unsqueeze(0)
        
        # Forward pass
        prediction = learning_net(x)
        loss = criterion(prediction, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Apply shrink-and-perturb if enabled
        if use_shrink_perturb:
            with torch.no_grad():
                for param in learning_net.parameters():
                    # Add Gaussian noise to the weights
                    if noise_variance > 0:
                        param.add_(torch.randn_like(param) * np.sqrt(noise_variance))
        
        # Apply continual backpropagation if enabled
        if continual_backprop and i > 0:
            with torch.no_grad():
                # Update age
                fc1_age += 1
                
                # Get activations
                h = learning_net.fc1(x)
                if learning_net.activation_name == 'relu':
                    h_act = torch.relu(h)
                elif learning_net.activation_name == 'tanh':
                    h_act = torch.tanh(h)
                elif learning_net.activation_name == 'sigmoid':
                    h_act = torch.sigmoid(h)
                elif learning_net.activation_name == 'leaky_relu':
                    h_act = torch.nn.functional.leaky_relu(h, negative_slope=0.01)
                elif learning_net.activation_name == 'elu':
                    h_act = torch.nn.functional.elu(h)
                elif learning_net.activation_name == 'swish':
                    h_act = h * torch.sigmoid(h)
                
                # Update running average of activations
                fc1_activation_avg = decay_rate * fc1_activation_avg + (1 - decay_rate) * h_act.squeeze(0)
                bias_corrected_avg = fc1_activation_avg / (1 - decay_rate ** fc1_age)
                
                # Calculate contribution utility (mean-corrected)
                contribution = torch.abs(h_act.squeeze(0) - bias_corrected_avg) * torch.sum(torch.abs(learning_net.fc2.weight), dim=0)
                
                # Calculate adaptation utility (inverse of input weight magnitude)
                adaptation = 1.0 / (torch.sum(torch.abs(learning_net.fc1.weight), dim=1) + 1e-10)
                
                # Overall utility
                utility = contribution * adaptation
                fc1_utility = decay_rate * fc1_utility + (1 - decay_rate) * utility
                bias_corrected_utility = fc1_utility / (1 - decay_rate ** fc1_age)
                
                # Find units to reinitialize
                eligible_units = (fc1_age > maturity_threshold).nonzero().squeeze(-1)
                if len(eligible_units) > 0 and rho > 0:
                    num_units_to_replace = max(1, int(hidden_size_learner * rho))
                    eligible_utilities = bias_corrected_utility[eligible_units]
                    _, indices = torch.topk(eligible_utilities, min(len(eligible_units), hidden_size_learner), 
                                           largest=False)
                    units_to_replace = eligible_units[indices[:num_units_to_replace]]
                    
                    if len(units_to_replace) > 0:
                        # Reinitialize input weights
                        if activation == 'relu':
                            gain = nn.init.calculate_gain('relu')
                            fan_in = learning_net.fc1.weight.size(1)
                            bound = gain * np.sqrt(3.0 / fan_in)
                            nn.init.uniform_(learning_net.fc1.weight[units_to_replace], -bound, bound)
                        elif activation == 'tanh':
                            gain = nn.init.calculate_gain('tanh')
                            fan_in = learning_net.fc1.weight.size(1)
                            bound = gain * np.sqrt(3.0 / fan_in)
                            nn.init.uniform_(learning_net.fc1.weight[units_to_replace], -bound, bound)
                        elif activation == 'sigmoid':
                            fan_in = learning_net.fc1.weight.size(1)
                            bound = 1 * np.sqrt(3.0 / fan_in)
                            nn.init.uniform_(learning_net.fc1.weight[units_to_replace], -bound, bound)
                        elif activation == 'leaky_relu':
                            gain = nn.init.calculate_gain('leaky_relu', 0.01)
                            fan_in = learning_net.fc1.weight.size(1)
                            bound = gain * np.sqrt(3.0 / fan_in)
                            nn.init.uniform_(learning_net.fc1.weight[units_to_replace], -bound, bound)
                        elif activation == 'elu':
                            gain = nn.init.calculate_gain('relu')  # Approximation
                            fan_in = learning_net.fc1.weight.size(1)
                            bound = gain * np.sqrt(3.0 / fan_in)
                            nn.init.uniform_(learning_net.fc1.weight[units_to_replace], -bound, bound)
                        elif activation == 'swish':
                            gain = nn.init.calculate_gain('relu')  # Approximation
                            fan_in = learning_net.fc1.weight.size(1)
                            bound = gain * np.sqrt(3.0 / fan_in)
                            nn.init.uniform_(learning_net.fc1.weight[units_to_replace], -bound, bound)
                        
                        # Initialize output weights to zero
                        learning_net.fc2.weight[:, units_to_replace] = 0
                        
                        # Reset utility, activation average, and age
                        fc1_utility[units_to_replace] = 0
                        fc1_activation_avg[units_to_replace] = 0
                        fc1_age[units_to_replace] = 0
        
        # Calculate squared error for current bin
        bin_idx = i // bin_size
        if bin_idx < n_bins:
            with torch.no_grad():
                squared_error = (prediction - y).pow(2).item()
                squared_errors[bin_idx] += squared_error / bin_size
    
    return {
        'squared_errors': squared_errors,
        'bins': np.arange(n_bins) * bin_size,
        'flip_indices': flip_indices
    }

def main():
    # Example usage
    print("Running Slowly-Changing Regression experiment with ReLU activation")
    result_relu = run_slowly_changing_regression_experiment(
        activation='relu',
        step_size=0.01,
        n_examples=500000,  # Reduced for faster execution
        bin_size=10000
    )
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(result_relu['bins'], result_relu['squared_errors'])
    plt.xlabel('Example Number')
    plt.ylabel('Mean Squared Error')
    plt.title('Slowly-Changing Regression with ReLU Activation')
    plt.grid(True)
    
    # Mark where bit flips occurred
    for flip_idx in result_relu['flip_indices']:
        if flip_idx < 500000:  # Only mark flips within the plot range
            plt.axvline(x=flip_idx, color='r', linestyle='--', alpha=0.3)
    
    # plt.savefig('bit_flipping_result.png')
    plt.show()
    
    print("Experiment complete. Results saved to bit_flipping_result.png")

if __name__ == "__main__":
    main()