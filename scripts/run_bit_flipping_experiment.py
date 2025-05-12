import sys
import os

# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import time
import wandb
from typing import List, Tuple, Callable, Dict, Optional
import hydra
from omegaconf import DictConfig, OmegaConf
import os
import collections

from src.config.utils import get_device, setup_wandb
from src.utils.metrics import analyze_fixed_batch
from src.utils.monitor import NetworkMonitor

def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

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
    def __init__(self, input_size: int, hidden_size: int, beta: float, device: torch.device):
        super(TargetNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.beta = beta
        self.device = device
        
        # Initialize weights as -1 or 1 as specified in the paper
        self.weights = (torch.randint(0, 2, (hidden_size, input_size), device=self.device).float() * 2 - 1)
        self.output_weights = (torch.randint(0, 2, (1, hidden_size), device=self.device).float() * 2 - 1)
        self.output_bias = torch.randint(0, 2, (1, ), dtype=torch.float, device=self.device)*2 - 1

        
        # Calculate thresholds for LTU units
        self.S = torch.sum((self.weights[:, :input_size-1] < 0).float(), dim=1) - 0.5*self.weights[:, -1]
        self.thresholds = (input_size * beta) - self.S
    
    def forward(self, x):
        # Ensure input x is on the same device as the network's weights
        x = x.to(self.device)
        # First layer with LTU activation
        pre_activation = torch.matmul(x, self.weights.t())
        hidden = torch.zeros_like(pre_activation)
        for i in range(self.hidden_size):
            hidden[:, i] = LTUActivation.apply(pre_activation[:, i], self.thresholds[i])
        
        # Output layer (linear)
        output = torch.matmul(hidden, self.output_weights.t()) + self.output_bias
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
    
    def forward(self, x, return_activation=False):
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
        if not return_activation: return output
        else: return output, h

def generate_bit_flipping_data(
    m: int,
    f: int,
    T: int,
    n_examples: int,
    device: torch.device
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
    inputs = torch.zeros(n_examples, m + 1, device=device)
    
    # Initialize the flipping bits (first f bits)
    flipping_bits = torch.randint(0, 2, (f,), device=device).float()
    
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
        inputs[i, f:m] = torch.randint(0, 2, (m - f,), device=device).float()
        
        # Set the bias bit (always 1)
        inputs[i, m] = 1.0
    
    return inputs, flip_indices



@hydra.main(version_base=None, config_path="../conf", config_name="config_bit_flipping")
def run_slowly_changing_regression_experiment_with_tracking(cfg: DictConfig) -> dict:
    """
    Run the Slowly-Changing Regression experiment using Hydra configuration.
    
    Args:
        cfg: Hydra configuration object (contains _global_ key)
    
    Returns:
        Dictionary containing experiment results
    """

    if hasattr(cfg, '_global'):
        cfg = cfg._global

    use_wandb = setup_wandb(cfg)

    print(OmegaConf.to_yaml(cfg))
    set_seed(cfg.training.seed)
    device = get_device(cfg.training.device)
    print(f"Using device: {device}")


    target_net = TargetNetwork(cfg.model.m + 1, cfg.model.hidden_size_target, cfg.model.beta, device)
    learning_net = LearningNetwork(cfg.model.m + 1, cfg.model.hidden_size_learner, cfg.model.activation).to(device)
    monitor = NetworkMonitor(learning_net)
    
    if cfg.optimizer.use_adam:
        optimizer = optim.Adam(learning_net.parameters(), lr=cfg.optimizer.step_size, weight_decay=cfg.optimizer.weight_decay)
    else:
        optimizer = optim.SGD(learning_net.parameters(), lr=cfg.optimizer.step_size, weight_decay=cfg.optimizer.weight_decay)
    
    criterion = nn.MSELoss()
    
    print("Generating data...")
    inputs, flip_indices_np = generate_bit_flipping_data(cfg.model.m, cfg.model.f, cfg.training.T, cfg.training.n_examples, device)
    # 'inputs' is already on the correct device from generate_bit_flipping_data

    with torch.no_grad():
        y_target = target_net(inputs) # y_target will be on device as inputs and target_net are

    # For analysis callback
    fixed_inputs_analysis, fixed_y_target_analysis = None, None
    if cfg.metrics.fixed_batch_size > 0:
        indices = torch.randperm(inputs.size(0), device=device)[:cfg.metrics.fixed_batch_size]
        fixed_inputs_analysis = inputs[indices] # Already on device
        fixed_y_target_analysis = y_target[indices] # Already on device
    else:
        fixed_inputs_analysis = inputs
        fixed_y_target_analysis = y_target


     # Utility tracking for continual backpropagation
    if cfg.training.continual_backprop:
        # Initialize utility tracking
        fc1_utility = torch.zeros(cfg.model.hidden_size_learner, device=device)
        fc1_activation_avg = torch.zeros(cfg.model.hidden_size_learner, dtype=torch.long, device=device)
        fc1_age =  torch.zeros(cfg.model.hidden_size_learner, device=device)
        decay_rate = 0.99  # Decay rate for running averages

    # Shrink and Perturb: noise scaling factor if needed
    if cfg.training.use_shrink_perturb:
        noise_scale = cfg.training.noise_variance # Direct use, assuming it's standard deviation

    # History and Metrics Storage
    history = {
        'global_metrics': {
            'steps': [],
            'loss_per_step': []
        },
        'binned_mean_squared_errors_log': [], 
        'flip_indices': flip_indices_np.tolist() if isinstance(flip_indices_np, np.ndarray) else flip_indices_np,
        'training_metrics_history': collections.defaultdict(lambda: collections.defaultdict(list))
    }

    # Define analyze_callback
    def analyze_callback(current_step_idx, completed_bin_idx, metrics_log_dict, model_to_analyze, net_monitor, 
                         fixed_inputs_for_analysis, fixed_targets_for_analysis, loss_criterion):
        nonlocal history # Allow modification of the history dict
        bin_start_step_inclusive = completed_bin_idx * cfg.training.bin_size
        bin_end_step_exclusive = (completed_bin_idx + 1) * cfg.training.bin_size
        
        losses_for_bin = history['global_metrics']['loss_per_step'][bin_start_step_inclusive:bin_end_step_exclusive]
        
        mse_for_bin = np.mean(losses_for_bin)
        history['binned_mean_squared_errors_log'].append(mse_for_bin)
        print(f"\nBin {completed_bin_idx} Report (step {current_step_idx}): MSE = {mse_for_bin:.6f}") 
        if metrics_log_dict is not None:
            metrics_log_dict[f"train/mse_bin"] = mse_for_bin
            # completed_bin_idx is already added to metrics_log_dict by the caller

        # Call analyze_fixed_batch for more detailed metrics
        detailed_metrics_output, activations_output, updated_metrics_log_dict = analyze_fixed_batch(
            model=model_to_analyze,
            monitor=net_monitor,
            fixed_batch=fixed_inputs_for_analysis,
            fixed_targets=fixed_targets_for_analysis,
            criterion=loss_criterion,
            dead_threshold=cfg.metrics.dead_threshold,
            corr_threshold=cfg.metrics.corr_threshold,
            saturation_threshold=cfg.metrics.saturation_threshold,
            saturation_percentage=cfg.metrics.saturation_percentage,
            gaussianity_method=cfg.metrics.gaussianity_method,
            use_wandb=use_wandb, 
            log_histograms=cfg.metrics.log_histograms,
            prefix="train/", 
            metrics_log=metrics_log_dict, # Pass the existing log dict to be updated
            device=device,
            seed=cfg.training.seed
        )
        return detailed_metrics_output, activations_output, updated_metrics_log_dict

    print("Starting training...")
    # Training loop (example structure)
    for i in tqdm(range(cfg.training.n_examples)):
        x = inputs[i].unsqueeze(0)  # x is already on device
        y_target_val = y_target[i].unsqueeze(0)  # y_target_val is already on device

        # Forward pass: learning network
        y_pred = learning_net(x)
        
        # Compute loss
        loss = criterion(y_pred, y_target_val)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log per-step metrics to history
        history['global_metrics']['steps'].append(i)
        history['global_metrics']['loss_per_step'].append(loss.item())
        
        # W&B per-step logging for basic loss (moved from end of loop)
        if use_wandb:
            wandb.log({"step": i, "train/loss_step": loss.item()}, step=i)

        if cfg.training.use_shrink_perturb:
            with torch.no_grad():
                for param in learning_net.parameters():
                    if noise_scale > 0:
                        param.add_(torch.randn_like(param) * np.sqrt(noise_scale))
        

        if cfg.training.continual_backprop and i > 0:
            with torch.no_grad():
                fc1_age += 1
                # h_fc1 = learning_net.fc1(x)
                _, h_act = learning_net(x, return_activation=True)
                
                # if cfg.model.activation == 'relu': h_act = torch.relu(h_fc1)
                # elif cfg.model.activation == 'tanh': h_act = torch.tanh(h_fc1)
                # elif cfg.model.activation == 'sigmoid': h_act = torch.sigmoid(h_fc1)
                # elif cfg.model.activation == 'leaky_relu': h_act = torch.nn.functional.leaky_relu(h_fc1, negative_slope=0.01)
                # elif cfg.model.activation == 'elu': h_act = torch.nn.functional.elu(h_fc1)
                # elif cfg.model.activation == 'swish': h_act = h_fc1 * torch.sigmoid(h_fc1)
                # else: raise ValueError(f"Unsupported activation for CBP: {cfg.model.activation}")
                

                # Update running average of activations
                fc1_activation_avg = decay_rate * fc1_activation_avg + (1 - decay_rate) * h_act.squeeze(0)
                bias_corrected_avg = fc1_activation_avg / (1 - decay_rate ** fc1_age)
                
                # Calculate contribution utility (mean-corrected)
                # adaptable contribution
                contribution = torch.abs(h_act.squeeze(0) - bias_corrected_avg) * torch.sum(torch.abs(learning_net.fc2.weight), dim=0)
                
                # Calculate adaptation utility (inverse of input weight magnitude)
                adaptation = 1.0 / (torch.sum(torch.abs(learning_net.fc1.weight), dim=1) + 1e-10)
                
                # Overall utility
                utility = contribution * adaptation
                fc1_utility = decay_rate * fc1_utility + (1 - decay_rate) * utility
                bias_corrected_utility = fc1_utility / (1 - decay_rate ** fc1_age)
                
                # Find units to reinitialize
                eligible_units = (fc1_age > cfg.training.maturity_threshold).nonzero().squeeze(-1)
                if len(eligible_units) > 0 and cfg.training.rho > 0:
                    num_units_to_replace = max(1, int(cfg.model.hidden_size_learner * cfg.training.rho))
                    
                    eligible_utilities = bias_corrected_utility[eligible_units]
                    _, indices = torch.topk(eligible_utilities, min(len(eligible_units), cfg.model.hidden_size_learner), largest=False)
                    
                    units_to_replace = eligible_units[indices[:num_units_to_replace]]
                    
                    if len(units_to_replace) > 0:
                        # Reinitialize input weights
                        if cfg.model.activation == 'relu':
                            gain = nn.init.calculate_gain('relu')
                            fan_in = learning_net.fc1.weight.size(1)
                            bound = gain * np.sqrt(3.0 / fan_in)
                            nn.init.uniform_(learning_net.fc1.weight[units_to_replace], -bound, bound)
                            nn.init.zeros_(learning_net.fc1.bias)
                            #  Update bias to correct for the removed features and set the outgoing weights and ages to zero
                            learning_net.fc2.bias += (learning_net.fc2.weight[:, units_to_replace] * \
                                                bias_corrected_avg[units_to_replace]).sum(dim=1)
                            learning_net.fc2.weight[:, units_to_replace] = 0
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


        if (i + 1) % cfg.training.bin_size == 0:
            completed_bin_idx = i // cfg.training.bin_size
            metrics_for_log = {
                "step": i,
                "completed_bin_idx": completed_bin_idx
            }
            detailed_metrics, train_act_stats, metrics_log = analyze_callback(current_step_idx=i, 
                             completed_bin_idx=completed_bin_idx, 
                             metrics_log_dict=metrics_for_log,
                             model_to_analyze=learning_net,
                             net_monitor=monitor,
                             fixed_inputs_for_analysis=fixed_inputs_analysis,
                             fixed_targets_for_analysis=fixed_y_target_analysis,
                             loss_criterion=criterion)
            
            # Store metrics in history
            for layer_name, metrics in detailed_metrics.items():
                for metric_name, value in metrics.items():
                    history['training_metrics_history'][layer_name][metric_name].append(value)
                

            # metrics_log_dict is updated in-place by analyze_fixed_batch if use_wandb=True and metrics_log is provided
            if use_wandb and metrics_log:
                wandb.log(metrics_log, step=i) 
        
    if use_wandb:
        wandb.finish()

    return history


if __name__ == '__main__':
    run_slowly_changing_regression_experiment_with_tracking()