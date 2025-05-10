"""
Cloning experiment for the Loss of Plasticity project.

This script implements the "Validate Constraint (No Escape)" scenario.
It demonstrates how cloned networks behave like smaller networks,
confining the parameter trajectory to lower-dimensional subspaces.

The experiment trains:
1. A base model ('base_ref_model').
2. A cloned model ('cloned_model'), initialized by cloning the trained base model.
3. An expanded model ('expanded_scratch_model'), same size as cloned, trained from scratch.

Metrics tracked:
- Training/Validation Loss and Accuracy.
- Rank-based metrics (Hard Rank, Effective Rank, Stable Rank) for key layers.
- Cloning Unexplained Variance (for cloned_model and expanded_scratch_model vs. base_ref_model).
"""

import os
import sys
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms
from datetime import datetime
from tqdm import tqdm

# Add the project root to the Python path
# Adjust based on the actual location of 'src' relative to this script.
# If script is in project_root/experiments/ and src is in project_root/src/
# then '..' is correct.
# If script is in project_root/notebooks/POC/ and src is in project_root/src/
# then '../..' is needed to get to project_root.
# Based on traceback: /local/home/ajoudaki/projects/loss-of-plasticity/notebooks/POC/cloning_experiment.py
# Assuming 'src' is in /local/home/ajoudaki/projects/loss-of-plasticity/src/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Try to import from src.utils, assuming standard project structure
try:
    from src.utils.cloning import model_clone, test_activation_cloning
    from src.utils.metrics import measure_effective_rank as measure_effective_rank_metricspy
    from src.utils.metrics import measure_stable_rank as measure_stable_rank_metricspy
except ImportError as e:
    print(f"ERROR: Could not import modules from src.utils: {e}")
    print("Please ensure __init__.py files are in 'src/' and 'src/utils/' directories.")
    print(f"PROJECT_ROOT is set to: {PROJECT_ROOT}")
    print(f"sys.path includes: {sys.path}")
    print("Attempting to use placeholders or local definitions if available (will likely fail for critical functions).")
    # Define placeholders if necessary, but this indicates a setup issue.
    # For critical functions like model_clone, this will not be sufficient.
    def model_clone(m1, m2): raise NotImplementedError("model_clone placeholder")
    def test_activation_cloning(*args, **kwargs): raise NotImplementedError("test_activation_cloning placeholder")
    def measure_effective_rank_metricspy(*args, **kwargs): return 0.0
    def measure_stable_rank_metricspy(*args, **kwargs): return 0.0


# Set up the output directory for figures
FIGURES_DIR = os.path.join(os.path.dirname(__file__), 'figures', 'no_escape')
os.makedirs(FIGURES_DIR, exist_ok=True)

# Set the random seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# MLP model
class MLP(nn.Module):
    def __init__(self,
                 input_size=784,
                 hidden_sizes=[512, 256, 128],
                 output_size=10,
                 activation='relu',
                 dropout_p=0.0, 
                 normalization=None):
        super().__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.layer_names_generated = [] # To store names for rank metric tracking

        if activation == 'relu':
            act_fn = nn.ReLU()
        elif activation == 'tanh':
            act_fn = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        layers = []
        prev_size = input_size
        
        for i, hidden_size in enumerate(hidden_sizes):
            linear_layer = nn.Linear(prev_size, hidden_size)
            layers.append(linear_layer)
            self.layer_names_generated.append(f'hidden_linear_{i}')
            
            if normalization == 'batch':
                layers.append(nn.BatchNorm1d(hidden_size))
                self.layer_names_generated.append(f'hidden_bn_{i}')
            elif normalization == 'layer':
                layers.append(nn.LayerNorm(hidden_size))
                self.layer_names_generated.append(f'hidden_ln_{i}')
            
            layers.append(act_fn)
            self.layer_names_generated.append(f'hidden_activation_{i}') 

            if dropout_p > 0: 
                layers.append(nn.Dropout(dropout_p))
                self.layer_names_generated.append(f'hidden_dropout_{i}')
            prev_size = hidden_size
        
        output_layer = nn.Linear(prev_size, output_size)
        layers.append(output_layer)
        self.layer_names_generated.append('output_linear') 

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.layers(x)


def create_dataset_loaders(dataset_name, batch_size=128, val_split=0.1, data_root='../data'):
    os.makedirs(data_root, exist_ok=True)
    if dataset_name.lower() == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root=data_root, train=False, download=True, transform=transform)
        input_size = 28 * 28
        num_classes = 10
    elif dataset_name.lower() == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        train_dataset = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform_test)
        input_size = 32 * 32 * 3
        num_classes = 10
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split_idx = int(np.floor(val_split * num_train))
    np.random.shuffle(indices) # Shuffling done here for train/val split consistency
    train_idx, val_idx = indices[split_idx:], indices[:split_idx]

    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=2, pin_memory=device.type=='cuda')
    val_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=2, pin_memory=device.type=='cuda')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=device.type=='cuda')
    
    fixed_batch_sampler_indices = train_idx[:min(500, len(train_idx))]
    fixed_batch_size = min(batch_size, len(fixed_batch_sampler_indices))
    if not fixed_batch_sampler_indices: # handle case where train_idx is too small
        fixed_train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True) # get at least one sample
        print("Warning: fixed_train_loader using a single shuffled sample due to small dataset partition.")
    else:
        fixed_train_loader = DataLoader(train_dataset, batch_size=fixed_batch_size, sampler=SubsetRandomSampler(fixed_batch_sampler_indices), shuffle=False)


    data_info = {'input_size': input_size, 'num_classes': num_classes}
    return {'train': train_loader, 'val': val_loader, 'test': test_loader, 'fixed_train': fixed_train_loader}, data_info


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    if total == 0: return 0.0, 0.0 # Avoid division by zero
    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    if total == 0: return 0.0, 0.0 # Avoid division by zero
    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc

# Rank metric calculations
def calculate_hard_rank(activations, threshold=1e-5):
    if activations.numel() == 0 or activations.shape[0] < 1: return 0.0
    act_np = activations.detach().cpu().numpy()
    if act_np.shape[0] < act_np.shape[1] and act_np.shape[0] > 0: 
        try:
            s = np.linalg.svd(act_np, compute_uv=False)
        except np.linalg.LinAlgError: # Handle cases where SVD might not converge
            return 0.0
    elif act_np.shape[0] >= act_np.shape[1] and act_np.shape[1] > 0:
        if act_np.shape[0] < 2: return float(act_np.shape[1]>0) # rank is 1 if there's one sample and features
        try:
            cov = np.cov(act_np.T)
            s = np.linalg.svd(cov, compute_uv=False)
        except np.linalg.LinAlgError:
             return 0.0
    else: # Zero features or zero samples
        return 0.0
    return (s > threshold).sum().item()


def calculate_effective_rank_local(activations):
    if activations.numel() == 0 or activations.shape[0] < 1 or activations.shape[1] < 1: return 0.0
    act_np = activations.detach().cpu().numpy()
    if act_np.shape[0] < 2: return 1.0 # Effective rank of 1 for a single sample or if covariance cannot be computed well
    
    try:
        cov = np.cov(act_np.T) 
        s = np.linalg.svd(cov, compute_uv=False)
    except np.linalg.LinAlgError:
        return 0.0
        
    s = np.maximum(s, 1e-10) 
    s_sum = np.sum(s)
    if s_sum < 1e-9: return 0.0
    p = s / s_sum
    entropy = -np.sum(p * np.log(p + 1e-12)) 
    return np.exp(entropy)

def calculate_stable_rank_local(activations):
    if activations.numel() == 0 or activations.shape[0] < 1 or activations.shape[1] < 1: return 0.0
    act_np = activations.detach().cpu().numpy() 
    try:
        frob_norm_sq = np.sum(act_np**2)
        spectral_norm = np.linalg.norm(act_np, 2) 
    except np.linalg.LinAlgError:
        return 0.0

    if spectral_norm < 1e-9 : return 0.0
    return frob_norm_sq / (spectral_norm**2)


def collect_all_metrics(model, fixed_loader, device, config):
    model.eval()
    metrics = {}
    
    try:
        inputs, _ = next(iter(fixed_loader))
    except StopIteration:
        print("Error: Fixed loader is empty. Cannot collect metrics.")
        return metrics # Return empty metrics
        
    inputs = inputs.to(device)
    if inputs.numel() == 0:
        print("Error: Input batch from fixed_loader is empty.")
        return metrics

    activations_dict = {}
    hooks = []
    
    act_idx_counter = 0
    linear_idx_counter = 0

    # Iterate through the modules in model.layers (which is an nn.Sequential)
    for layer_module_in_seq in model.layers:
        current_layer_name_for_config = None # This name is for matching with config['rank_metric_layers']
        
        if isinstance(layer_module_in_seq, (nn.ReLU, nn.Tanh)):
            # Check if this activation follows a hidden linear layer
            if linear_idx_counter <= len(model.hidden_sizes) and linear_idx_counter > 0 : # make sure it's not before first linear, or after output linear
                current_layer_name_for_config = f"hidden_activation_{act_idx_counter}"
                act_idx_counter += 1
        elif isinstance(layer_module_in_seq, nn.Linear):
            if linear_idx_counter < len(model.hidden_sizes): # It's a hidden linear layer
                linear_idx_counter +=1
            # We don't increment linear_idx_counter for the output layer here for this logic

        if current_layer_name_for_config and current_layer_name_for_config in config['rank_metric_layers']:
            # The actual name for activations_dict should be unique if multiple modules map to same config name (not an issue here)
            hook_name = current_layer_name_for_config 
            
            # Need a new scope for hook_fn per iteration to capture `hook_name` correctly
            def make_hook_fn(name_capture):
                def hook_fn(module, input, output):
                    activations_dict[name_capture] = output.detach().clone() # No .cpu() yet, do it before numpy
                return hook_fn
            
            hooks.append(layer_module_in_seq.register_forward_hook(make_hook_fn(hook_name)))

    if not hooks and config['rank_metric_layers']:
        print(f"Warning: No hooks registered for rank metrics. Check layer names in config: {config['rank_metric_layers']}")
        # print(f"Model layers structure: {model.layers}") # For debugging model structure
        # print(f"Model generated internal names: {model.layer_names_generated}")


    with torch.no_grad():
        _ = model(inputs) 

    for h in hooks:
        h.remove()

    for layer_name, layer_activations in activations_dict.items():
        if layer_activations.numel() == 0:
            print(f"Warning: Empty activations for layer {layer_name}")
            metrics[f"{layer_name}/hard_rank"] = 0.0
            metrics[f"{layer_name}/effective_rank"] = 0.0
            metrics[f"{layer_name}/stable_rank"] = 0.0
            continue
            
        if layer_activations.dim() > 2:
            layer_activations_flat = layer_activations.reshape(layer_activations.size(0), -1)
        else:
            layer_activations_flat = layer_activations
        
        if layer_activations_flat.shape[0] > 0 and layer_activations_flat.shape[1] > 0 : 
            metrics[f"{layer_name}/hard_rank"] = calculate_hard_rank(layer_activations_flat)
            metrics[f"{layer_name}/effective_rank"] = calculate_effective_rank_local(layer_activations_flat)
            metrics[f"{layer_name}/stable_rank"] = calculate_stable_rank_local(layer_activations_flat)
        else:
            metrics[f"{layer_name}/hard_rank"] = 0.0
            metrics[f"{layer_name}/effective_rank"] = 0.0
            metrics[f"{layer_name}/stable_rank"] = 0.0
    return metrics


def run_experiment(config):
    print(f"\n{'='*80}")
    print(f"Running '{config['scenario']}' scenario on {config['dataset_name'].upper()} with:")
    print(f"Base Hidden: {config['model_params']['hidden_sizes']}, Expansion: {config['expansion_factor']}x")
    print(f"Base Epochs: {config['epochs_base_train']}, Main Epochs: {config['epochs_main_train']}")
    print(f"{'='*80}\n")

    dataloaders, data_info = create_dataset_loaders(config['dataset_name'], config['batch_size'], data_root=config['data_root'])
    criterion = nn.CrossEntropyLoss()

    histories = {
        'base_ref': {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'test_loss': [], 'test_acc': [], 'metrics_trace': []},
        'cloned': {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'test_loss': [], 'test_acc': [], 'metrics_trace': [], 'unexplained_var': []},
        'expanded_scratch': {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'test_loss': [], 'test_acc': [], 'metrics_trace': [], 'unexplained_var': []}
    }

    # --- Phase 1: Train Base Reference Model ---
    print("--- Phase 1: Training Base Reference Model ---")
    base_ref_model_params_dict = {
        'input_size': data_info['input_size'],
        'output_size': data_info['num_classes'],
        **config['model_params'] 
    }
    base_ref_model = MLP(**base_ref_model_params_dict).to(device)
    base_optimizer = optim.Adam(base_ref_model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])

    for epoch in range(1, config['epochs_base_train'] + 1):
        train_loss, train_acc = train_epoch(base_ref_model, dataloaders['train'], criterion, base_optimizer, device)
        val_loss, val_acc = evaluate(base_ref_model, dataloaders['val'], criterion, device)
        
        histories['base_ref']['train_loss'].append(train_loss)
        histories['base_ref']['train_acc'].append(train_acc)
        histories['base_ref']['val_loss'].append(val_loss)
        histories['base_ref']['val_acc'].append(val_acc)

        if epoch % config['metrics_interval'] == 0 or epoch == config['epochs_base_train']:
            detailed_metrics = collect_all_metrics(base_ref_model, dataloaders['fixed_train'], device, config)
            histories['base_ref']['metrics_trace'].append({'epoch': epoch, **detailed_metrics})
            # Print first few detailed metrics for brevity
            print_metrics = {k: v for k, v in detailed_metrics.items() if v is not None}
            print(f"BaseRef Ep {epoch} | Metrics: {[(k, round(v,2)) for k,v in list(print_metrics.items())[:3]]}...")


        if epoch % 5 == 0 or epoch == 1 or epoch == config['epochs_base_train']:
            print(f"BaseRef Ep {epoch}/{config['epochs_base_train']} | TrL: {train_loss:.3f}, TrA: {train_acc:.2f}% | VaL: {val_loss:.3f}, VaA: {val_acc:.2f}%")

    base_ref_model.eval() 
    
    # --- Phase 2: Setup Cloned and Expanded-from-Scratch Models ---
    print("\n--- Phase 2: Setting up Cloned and Expanded-Scratch Models ---")
    expanded_hidden_sizes = [s * config['expansion_factor'] for s in config['model_params']['hidden_sizes']]
    expanded_model_params_dict = {
        'input_size': data_info['input_size'],
        'output_size': data_info['num_classes'],
        'hidden_sizes': expanded_hidden_sizes,
        'activation': config['model_params']['activation'],
        'normalization': config['model_params']['normalization'],
        'dropout_p': config['model_params']['dropout_p'] # Ensure cloned/expanded also get dropout if base had it (though 0 for this scenario)
    }

    cloned_model_shell = MLP(**expanded_model_params_dict) 
    cloned_model = model_clone(base_ref_model, cloned_model_shell).to(device) 
    cloned_optimizer = optim.Adam(cloned_model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    print(f"Cloned model created. Params: {sum(p.numel() for p in cloned_model.parameters())}")

    expanded_scratch_model = MLP(**expanded_model_params_dict).to(device)
    expanded_optimizer = optim.Adam(expanded_scratch_model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    print(f"Expanded-scratch model created. Params: {sum(p.numel() for p in expanded_scratch_model.parameters())}")

    fixed_batch_inputs, fixed_batch_targets = next(iter(dataloaders['fixed_train']))
    fixed_batch_inputs, fixed_batch_targets = fixed_batch_inputs.to(device), fixed_batch_targets.to(device)
    
    try:
        print("Testing initial cloning similarity for Cloned Model...")
        initial_success_cloned, initial_unexplained_var_cloned_dict = test_activation_cloning(
            base_ref_model, cloned_model, fixed_batch_inputs, fixed_batch_targets, model_name='mlp', tolerance=1e-3
        )
        avg_initial_unexplained_cloned = np.mean(list(initial_unexplained_var_cloned_dict.values())) if initial_unexplained_var_cloned_dict else 0.0
        print(f"Initial cloning for Cloned Model: Success: {initial_success_cloned}, Avg Unexplained Var: {avg_initial_unexplained_cloned:.4e}")
    except Exception as e:
        print(f"Error during initial cloning test for Cloned Model: {e}")

    # --- Phase 3: Train Cloned and Expanded-Scratch Models ---
    print("\n--- Phase 3: Training Cloned and Expanded-Scratch Models ---")
    for epoch in range(1, config['epochs_main_train'] + 1):
        cl_train_loss, cl_train_acc = train_epoch(cloned_model, dataloaders['train'], criterion, cloned_optimizer, device)
        histories['cloned']['train_loss'].append(cl_train_loss)
        histories['cloned']['train_acc'].append(cl_train_acc)

        ex_train_loss, ex_train_acc = train_epoch(expanded_scratch_model, dataloaders['train'], criterion, expanded_optimizer, device)
        histories['expanded_scratch']['train_loss'].append(ex_train_loss)
        histories['expanded_scratch']['train_acc'].append(ex_train_acc)

        cl_val_loss, cl_val_acc = evaluate(cloned_model, dataloaders['val'], criterion, device)
        histories['cloned']['val_loss'].append(cl_val_loss)
        histories['cloned']['val_acc'].append(cl_val_acc)

        ex_val_loss, ex_val_acc = evaluate(expanded_scratch_model, dataloaders['val'], criterion, device)
        histories['expanded_scratch']['val_loss'].append(ex_val_loss)
        histories['expanded_scratch']['val_acc'].append(ex_val_acc)

        print_epoch_stats = (epoch % 5 == 0 or epoch == 1 or epoch == config['epochs_main_train'])
        if print_epoch_stats:
            print(f"Main Ep {epoch}/{config['epochs_main_train']}")
            print(f"  Cloned: TrL: {cl_train_loss:.3f}, TrA: {cl_train_acc:.2f}% | VaL: {cl_val_loss:.3f}, VaA: {cl_val_acc:.2f}%")
            print(f"  Expand: TrL: {ex_train_loss:.3f}, TrA: {ex_train_acc:.2f}% | VaL: {ex_val_loss:.3f}, VaA: {ex_val_acc:.2f}%")

        if epoch % config['metrics_interval'] == 0 or epoch == config['epochs_main_train']:
            cl_detailed_metrics = collect_all_metrics(cloned_model, dataloaders['fixed_train'], device, config)
            histories['cloned']['metrics_trace'].append({'epoch': epoch, **cl_detailed_metrics})
            
            avg_unexplained_cl = float('nan')
            try:
                _, unexplained_var_cl_dict = test_activation_cloning(base_ref_model, cloned_model, fixed_batch_inputs, fixed_batch_targets, model_name='mlp', tolerance=1e-3)
                avg_unexplained_cl = np.mean(list(unexplained_var_cl_dict.values())) if unexplained_var_cl_dict else 0.0
                histories['cloned']['unexplained_var'].append(avg_unexplained_cl)
            except Exception as e:
                print(f"Error in test_activation_cloning for Cloned (Ep {epoch}): {e}")
                histories['cloned']['unexplained_var'].append(float('nan'))

            ex_detailed_metrics = collect_all_metrics(expanded_scratch_model, dataloaders['fixed_train'], device, config)
            histories['expanded_scratch']['metrics_trace'].append({'epoch': epoch, **ex_detailed_metrics})
            
            avg_unexplained_ex = float('nan')
            try:
                _, unexplained_var_ex_dict = test_activation_cloning(base_ref_model, expanded_scratch_model, fixed_batch_inputs, fixed_batch_targets, model_name='mlp', tolerance=1e-3)
                avg_unexplained_ex = np.mean(list(unexplained_var_ex_dict.values())) if unexplained_var_ex_dict else 0.0
                histories['expanded_scratch']['unexplained_var'].append(avg_unexplained_ex)
            except Exception as e:
                print(f"Error in test_activation_cloning for Expanded (Ep {epoch}): {e}")
                histories['expanded_scratch']['unexplained_var'].append(float('nan'))

            if print_epoch_stats:
                print_cl_metrics = {k:v for k,v in cl_detailed_metrics.items() if v is not None}
                print_ex_metrics = {k:v for k,v in ex_detailed_metrics.items() if v is not None}
                print(f"  Cloned Metrics (Ep {epoch}): {[(k, round(v,2)) for k,v in list(print_cl_metrics.items())[:2]]}..., UnexpVar: {avg_unexplained_cl:.2e}")
                print(f"  Expand Metrics (Ep {epoch}): {[(k, round(v,2)) for k,v in list(print_ex_metrics.items())[:2]]}..., UnexpVar: {avg_unexplained_ex:.2e}")
                
    print("\nExperiment completed!")
    return {'histories': histories, 'config': config, 'data_info': data_info}


def plot_experiment_results(results):
    histories = results['histories']
    config = results['config']
    dataset_name = config['dataset_name']
    
    epochs_base = range(1, len(histories['base_ref']['val_acc']) + 1)
    epochs_main = range(1, len(histories['cloned']['val_acc']) + 1)
    
    metrics_epochs_cloned = [m['epoch'] for m in histories['cloned']['metrics_trace']]
    metrics_epochs_expanded = [m['epoch'] for m in histories['expanded_scratch']['metrics_trace']]
    metrics_epochs_base = [m['epoch'] for m in histories['base_ref']['metrics_trace']]


    try:
        plt.style.use('seaborn-v0_8-whitegrid') # Or try 'seaborn-v0_8_paper' or just 'seaborn-v0_8-darkgrid'
    except OSError:
        print("Style 'seaborn_v0_8_whitegrid' not found, using default Matplotlib styles.")
        # You can add custom rcParams here as a fallback if needed
        # For example:
        # plt.rcParams.update({'font.size': 10, 'axes.grid': True})
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    colors = {'base_ref': 'blue', 'cloned': 'green', 'expanded_scratch': 'red'}
    linestyles = {'base_ref': '-', 'cloned': '--', 'expanded_scratch': ':'}

    # 1. Val Accuracy
    plt.figure(figsize=(10, 6))
    if histories['base_ref']['val_acc']: plt.plot(epochs_base, histories['base_ref']['val_acc'], label='Base Model (Ref)', color=colors['base_ref'], linestyle=linestyles['base_ref'])
    if histories['cloned']['val_acc']: plt.plot(epochs_main, histories['cloned']['val_acc'], label='Cloned Model', color=colors['cloned'], linestyle=linestyles['cloned'])
    if histories['expanded_scratch']['val_acc']: plt.plot(epochs_main, histories['expanded_scratch']['val_acc'], label='Expanded Model (Scratch)', color=colors['expanded_scratch'], linestyle=linestyles['expanded_scratch'])
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy (%)')
    plt.title(f'Validation Accuracy ({dataset_name.upper()}) - {config["scenario"]} Scenario')
    plt.legend()
    plt.savefig(os.path.join(FIGURES_DIR, f"val_accuracy_{config['scenario']}_{timestamp}.pdf"), bbox_inches='tight')
    plt.close()

    # 2. Val Loss
    plt.figure(figsize=(10, 6))
    if histories['base_ref']['val_loss']: plt.plot(epochs_base, histories['base_ref']['val_loss'], label='Base Model (Ref)', color=colors['base_ref'], linestyle=linestyles['base_ref'])
    if histories['cloned']['val_loss']: plt.plot(epochs_main, histories['cloned']['val_loss'], label='Cloned Model', color=colors['cloned'], linestyle=linestyles['cloned'])
    if histories['expanded_scratch']['val_loss']: plt.plot(epochs_main, histories['expanded_scratch']['val_loss'], label='Expanded Model (Scratch)', color=colors['expanded_scratch'], linestyle=linestyles['expanded_scratch'])
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title(f'Validation Loss ({dataset_name.upper()}) - {config["scenario"]} Scenario')
    plt.legend()
    plt.savefig(os.path.join(FIGURES_DIR, f"val_loss_{config['scenario']}_{timestamp}.pdf"), bbox_inches='tight')
    plt.close()

    # 3. Rank Metrics
    target_rank_layers = config.get('rank_metric_layers', [])
    if not target_rank_layers and len(config['model_params']['hidden_sizes']) > 0:
        target_rank_layers = [f'hidden_activation_0'] # Default if not specified but possible

    for target_rank_layer in target_rank_layers:
        for rank_type in ['hard_rank', 'effective_rank', 'stable_rank']:
            plt.figure(figsize=(10,6))
            metric_key = f"{target_rank_layer}/{rank_type}"
            
            plotted_something = False
            # Define the rank_type_title before using it
            rank_type_title = rank_type.replace("_", " ").title()

            if histories['base_ref']['metrics_trace']:
                base_ranks = [m.get(metric_key) for m in histories['base_ref']['metrics_trace']]
                base_ranks_clean = [(ep, r) for ep, r in zip(metrics_epochs_base, base_ranks) if r is not None]
                if base_ranks_clean:
                     plt.plot([x[0] for x in base_ranks_clean], [x[1] for x in base_ranks_clean], label=f'Base ({rank_type_title})', color=colors['base_ref'], linestyle=linestyles['base_ref'])
                     plotted_something = True
            if histories['cloned']['metrics_trace']:
                cl_ranks = [m.get(metric_key) for m in histories['cloned']['metrics_trace']]
                cl_ranks_clean = [(ep, r) for ep, r in zip(metrics_epochs_cloned, cl_ranks) if r is not None]
                if cl_ranks_clean:
                    plt.plot([x[0] for x in cl_ranks_clean], [x[1] for x in cl_ranks_clean], label=f'Cloned ({rank_type_title})', color=colors['cloned'], linestyle=linestyles['cloned'])
                    plotted_something = True

            if histories['expanded_scratch']['metrics_trace']:
                ex_ranks = [m.get(metric_key) for m in histories['expanded_scratch']['metrics_trace']]
                ex_ranks_clean = [(ep, r) for ep, r in zip(metrics_epochs_expanded, ex_ranks) if r is not None]
                if ex_ranks_clean:
                    plt.plot([x[0] for x in ex_ranks_clean], [x[1] for x in ex_ranks_clean], label=f'Expanded ({rank_type_title})', color=colors['expanded_scratch'], linestyle=linestyles['expanded_scratch'])
                    plotted_something = True

            if plotted_something:
                plt.xlabel('Epoch')
                plt.ylabel(f'{rank_type_title}')
                plt.title(f'{rank_type_title} for {target_rank_layer} ({dataset_name.upper()}) - {config["scenario"]}')
                plt.legend()
                plt.savefig(os.path.join(FIGURES_DIR, f"{rank_type}_{target_rank_layer.replace('/', '_')}_{config['scenario']}_{timestamp}.pdf"), bbox_inches='tight')
            plt.close()

    # 4. Cloning Unexplained Variance
    plt.figure(figsize=(10, 6))
    plotted_unexplained = False
    unexplained_epochs = metrics_epochs_cloned # Assuming same frequency as other metrics for cloned
    
    if histories['cloned']['unexplained_var'] and len(unexplained_epochs) == len(histories['cloned']['unexplained_var']):
         clean_unexplained_cl = [(ep, val) for ep, val in zip(unexplained_epochs, histories['cloned']['unexplained_var']) if not np.isnan(val)]
         if clean_unexplained_cl:
            plt.plot([x[0] for x in clean_unexplained_cl], [x[1] for x in clean_unexplained_cl], label='Cloned Model (vs. Base Ref)', color='darkgreen')
            plotted_unexplained = True

    unexplained_epochs_ex = metrics_epochs_expanded
    if histories['expanded_scratch']['unexplained_var'] and len(unexplained_epochs_ex) == len(histories['expanded_scratch']['unexplained_var']):
         clean_unexplained_ex = [(ep, val) for ep, val in zip(unexplained_epochs_ex, histories['expanded_scratch']['unexplained_var']) if not np.isnan(val)]
         if clean_unexplained_ex:
            plt.plot([x[0] for x in clean_unexplained_ex], [x[1] for x in clean_unexplained_ex], label='Expanded Model (vs. Base Ref - Baseline)', color='darkred', linestyle=':')
            plotted_unexplained = True
    
    if plotted_unexplained:
        plt.xlabel('Epoch (Main Training Phase)')
        plt.ylabel('Mean Unexplained Activation Variance')
        plt.title(f'Cloning Unexplained Variance ({dataset_name.upper()}) - {config["scenario"]}')
        plt.legend()
        plt.yscale('log') 
        plt.savefig(os.path.join(FIGURES_DIR, f"unexplained_variance_{config['scenario']}_{timestamp}.pdf"), bbox_inches='tight')
    plt.close()
    
    print(f"Plots saved to {FIGURES_DIR}")


def main():
    no_escape_config = {
        'scenario': 'no_escape',
        'dataset_name': 'mnist', 
        'data_root': os.path.join(PROJECT_ROOT, 'data'), # Adjusted data_root
        'model_params': {
            'hidden_sizes': [64, 32], 
            'activation': 'relu',
            'normalization': None, 
            'dropout_p': 0.0 
        },
        'expansion_factor': 2,
        'epochs_base_train': 2, 
        'epochs_main_train': 2, 
        'batch_size': 128,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'metrics_interval': 2, 
        'rank_metric_layers': [] # Will be auto-populated below
    }
    
    num_hidden = len(no_escape_config['model_params']['hidden_sizes'])
    no_escape_config['rank_metric_layers'] = [f'hidden_activation_{i}' for i in range(num_hidden)]
    if not no_escape_config['rank_metric_layers'] and num_hidden > 0 : # Default to first if none are added but possible
        no_escape_config['rank_metric_layers'] = ['hidden_activation_0']
    elif num_hidden == 0: # No hidden layers, no hidden activation layers for rank metrics
        no_escape_config['rank_metric_layers'] = []


    results = run_experiment(no_escape_config)
    plot_experiment_results(results)

if __name__ == '__main__':
    main()