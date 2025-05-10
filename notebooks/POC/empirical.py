import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import os

# --- Debug Flag ---
DEBUG_VERBOSE = True # Set to False to reduce print output

# --- Matplotlib Styling for Paper Quality ---
try:
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({
        'font.family': 'serif', 'text.usetex': False, 'figure.dpi': 300,
        'savefig.dpi': 300, 'savefig.bbox': 'tight', 'font.size': 8,
        'axes.labelsize': 8, 'axes.titlesize': 9, 'xtick.labelsize': 7,
        'ytick.labelsize': 7, 'legend.fontsize': 7, 'figure.titlesize': 10,
    })
except OSError:
    print("seaborn-v0_8-paper style not found, using rcParams for basic styling.")
    plt.rcParams.update({
        'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8,
        'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
        'figure.titlesize': 10, 'lines.linewidth': 1.2, 'lines.markersize': 3,
        'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
        'font.family': 'serif', 'text.usetex': False
    })

# --- Global Parameters & Figure Saving ---
FIGURE_DIR = "./figures/"
os.makedirs(FIGURE_DIR, exist_ok=True)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- 1. Helper Functions (Metrics) ---
def compute_effective_rank_empirical(activations_batch_cpu, stage_name_for_debug=""):
    if activations_batch_cpu.ndim == 1: return 1.0 if activations_batch_cpu.shape[0] > 1 else 0.0
    if activations_batch_cpu.shape[1] == 0: return 0.0
    if activations_batch_cpu.shape[1] == 1: return 1.0
    if activations_batch_cpu.shape[0] <= 1: return float(activations_batch_cpu.shape[1] > 0)

    std_devs = torch.std(activations_batch_cpu, dim=0)
    valid_features_mask = std_devs > 1e-5
    if valid_features_mask.sum() < 2: return float(valid_features_mask.sum().item())
    
    activations_batch_filtered = activations_batch_cpu[:, valid_features_mask]
    if activations_batch_filtered.shape[0] <= 1 or activations_batch_filtered.shape[1] < 2:
         return float(activations_batch_filtered.shape[1] > 0)

    try:
        centered_activations = activations_batch_filtered - activations_batch_filtered.mean(dim=0, keepdim=True)
        if centered_activations.shape[0] < centered_activations.shape[1] and centered_activations.shape[0] > 0:
             cov_matrix = torch.cov(centered_activations) 
        elif centered_activations.shape[0] >= centered_activations.shape[1] and centered_activations.shape[1] > 0: # More samples or equal
             cov_matrix = torch.cov(centered_activations.T)
        else: # Not enough data for covariance
            if DEBUG_VERBOSE: print(f"  [ER Debug {stage_name_for_debug}] Not enough data for cov after centering. Shape: {centered_activations.shape}")
            return 0.0


        if torch.isnan(cov_matrix).any(): cov_matrix = torch.nan_to_num(cov_matrix, nan=0.0)
        s_unnormalized = torch.linalg.svdvals(cov_matrix)
    except RuntimeError as e:
        if DEBUG_VERBOSE: print(f"  [ER Debug {stage_name_for_debug}] SVD RuntimeError: {e}. Filtered shape: {activations_batch_filtered.shape}")
        return 1.0 # Fallback, or 0.0? Depends on interpretation.

    sum_s = torch.sum(s_unnormalized)
    if sum_s < 1e-12: return 0.0
    s_norm_for_entropy = s_unnormalized / sum_s
    s_norm_for_entropy = s_norm_for_entropy[s_norm_for_entropy > 1e-15]
    if len(s_norm_for_entropy) == 0: return 0.0
    entropy = -torch.sum(s_norm_for_entropy * torch.log(s_norm_for_entropy))
    er_val = torch.exp(entropy).item()
    if DEBUG_VERBOSE and (er_val > activations_batch_filtered.shape[1] or er_val < 0):
        print(f"  [ER Debug {stage_name_for_debug}] Unusual ER: {er_val:.2f}, Max possible: {activations_batch_filtered.shape[1]}, Svals sum: {sum_s:.2e}")
    return er_val


def compute_frozen_stats(pre_activations_batch_cpu, activation_type_str, frozen_deriv_thresh=1e-3, stage_name_for_debug=""):
    if pre_activations_batch_cpu.numel() == 0 or pre_activations_batch_cpu.shape[1] == 0: return 0.0
    if pre_activations_batch_cpu.shape[0] == 0: return 0.0

    frozen_map = torch.zeros_like(pre_activations_batch_cpu, dtype=torch.bool)
    act_str_lower = activation_type_str.lower()

    if act_str_lower == "relu":
        frozen_map = pre_activations_batch_cpu <= 1e-6 
    elif act_str_lower == "tanh":
        tanh_x_sq = torch.tanh(pre_activations_batch_cpu)**2
        frozen_map = tanh_x_sq > (1.0 - frozen_deriv_thresh)
    elif act_str_lower == "sigmoid":
        sig_x = torch.sigmoid(pre_activations_batch_cpu)
        derivative = sig_x * (1.0 - sig_x)
        frozen_map = derivative < frozen_deriv_thresh 
    else: 
        if DEBUG_VERBOSE: print(f"  [Frozen Debug {stage_name_for_debug}] Frozen stats not applicable for {act_str_lower}, returning 0.")
        return 0.0 
    
    percentage_total_frozen_instances = frozen_map.float().mean().item() * 100.0
    if DEBUG_VERBOSE:
        print(f"  [Frozen Debug {stage_name_for_debug}] Act: {act_str_lower}, Input shape: {pre_activations_batch_cpu.shape}, Frozen elements: {frozen_map.sum().item()}/{frozen_map.numel()}, Perc: {percentage_total_frozen_instances:.2f}%")
    return percentage_total_frozen_instances

def compute_duplicate_neurons_stats(activations_batch_cpu, corr_threshold=0.95, stage_name_for_debug=""):
    if activations_batch_cpu.ndim < 2 or activations_batch_cpu.shape[1] < 2: return 0.0 # Need at least 2 features
    
    std_devs = torch.std(activations_batch_cpu, dim=0)
    valid_features_mask = std_devs > 1e-5 # Features with some variance
    
    if valid_features_mask.sum() < 2: # Need at least 2 *variant* features
        if DEBUG_VERBOSE: print(f"  [Duplicate Debug {stage_name_for_debug}] Not enough variant features ({valid_features_mask.sum().item()}) for duplicates. Original features: {activations_batch_cpu.shape[1]}")
        return 0.0
        
    activations_valid = activations_batch_cpu[:, valid_features_mask]
    num_valid_features = activations_valid.shape[1]

    try:
        # corrcoef expects features as rows, or pass .T for features as columns
        corr_matrix = torch.corrcoef(activations_valid.T) 
        if torch.isnan(corr_matrix).any(): corr_matrix = torch.nan_to_num(corr_matrix, nan=0.0)
    except RuntimeError as e:
        if DEBUG_VERBOSE: print(f"  [Duplicate Debug {stage_name_for_debug}] Corrcoef RuntimeError: {e}. Valid features: {num_valid_features}")
        return 0.0 # Cannot compute

    abs_corr_matrix = torch.abs(corr_matrix)
    # Exclude self-correlation by setting diagonal to 0 before checking for duplicates
    abs_corr_matrix.fill_diagonal_(0) 
    
    # A neuron is duplicate if it's highly correlated with *any* other neuron
    is_duplicate_neuron = torch.any(abs_corr_matrix > corr_threshold, dim=1)
    
    percentage_duplicates = is_duplicate_neuron.float().mean().item() * 100.0
    if DEBUG_VERBOSE:
        print(f"  [Duplicate Debug {stage_name_for_debug}] Input shape: {activations_batch_cpu.shape}, Valid feats: {num_valid_features}, Corr matrix shape: {corr_matrix.shape}, Duplicates found: {is_duplicate_neuron.sum().item()}, Perc: {percentage_duplicates:.2f}%")
    return percentage_duplicates

def get_activation_fn_and_name(activation_name_str):
    name_lower = activation_name_str.lower()
    if name_lower == "relu": return nn.ReLU(), "ReLU"
    elif name_lower == "tanh": return nn.Tanh(), "Tanh"
    elif name_lower == "sigmoid": return nn.Sigmoid(), "Sigmoid"
    else: raise ValueError(f"Unsupported activation: {activation_name_str}")

# --- 2. Model Definition ---
class MLPBlock(nn.Module):
    def __init__(self, in_dim, out_dim, norm_type, activation_fn_instance, activation_name, is_output_layer_block, block_idx):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = None
        self.act_fn_instance = activation_fn_instance
        self.activation_name = activation_name 
        
        layer_prefix = f"H{block_idx + 1}"
        self.names = {
            "linear_out": f"{layer_prefix}_Lin",    
            "norm_out": f"{layer_prefix}_Nrm",      
            "act_in": f"{layer_prefix}_ActIn",     
            "act_out": f"{layer_prefix}_Act"       
        }
        if is_output_layer_block: # Output layer has different naming conventions
            self.names = {"linear_out": "Out_Logits", "act_out": "Out_Logits", "act_in": "Out_Logits_ActIn"} # "act_in" for output usually isn't used for frozen stats unless there's a final act like softmax (not here)

        if not is_output_layer_block and norm_type: # Norm only for hidden layers
            if norm_type.lower() == "batchnorm": self.norm = nn.BatchNorm1d(out_dim)
            elif norm_type.lower() == "layernorm": self.norm = nn.LayerNorm(out_dim)

        self.act = self.act_fn_instance if not is_output_layer_block else nn.Identity() # No activation on logits

    def forward(self, x, record_activations=False):
        recorded_stages = {}
        
        current_features = self.linear(x)
        # Always record linear output if requested
        if record_activations: recorded_stages[self.names["linear_out"]] = current_features.detach().cpu()
        
        pre_act_input = current_features 

        if not (self.names["linear_out"] == self.names["act_out"]): # True for hidden layers
            if self.norm:
                current_features = self.norm(current_features)
                if record_activations: recorded_stages[self.names["norm_out"]] = current_features.detach().cpu()
                pre_act_input = current_features 
            
            if record_activations: recorded_stages[self.names["act_in"]] = pre_act_input.detach().cpu()
            current_features = self.act(current_features)
            if record_activations: recorded_stages[self.names["act_out"]] = current_features.detach().cpu()
        # For output layer, current_features is already logits (from linear_out), no further processing.
        # The names dict already maps act_out to Out_Logits for output layer.
        
        return current_features, recorded_stages

class ConfigurableMLP(nn.Module):
    def __init__(self, layer_dims, activation_name_str="relu", norm_type=None):
        super().__init__()
        self.activation_name_str_for_frozen = activation_name_str # Used by log_metrics
        
        act_fn_instance, act_display_name = get_activation_fn_and_name(activation_name_str)
        self.ordered_stage_names_for_plot = ["Input"]

        self.blocks = nn.ModuleList()
        for i in range(len(layer_dims) - 1):
            is_final_block = (i == len(layer_dims) - 2)
            block = MLPBlock(layer_dims[i], layer_dims[i+1], norm_type, 
                             act_fn_instance, act_display_name, is_final_block, i)
            self.blocks.append(block)
            
            # Collect all unique stage names from blocks in a sensible order for plotting
            if block.names["linear_out"] not in self.ordered_stage_names_for_plot:
                self.ordered_stage_names_for_plot.append(block.names["linear_out"])
            if "norm_out" in block.names and block.names["norm_out"] not in self.ordered_stage_names_for_plot:
                 if not is_final_block and norm_type: # Only add if norm exists for this block
                    self.ordered_stage_names_for_plot.append(block.names["norm_out"])
            if block.names["act_out"] not in self.ordered_stage_names_for_plot:
                 # Only add _Act if it's different from _Lin (i.e., not output layer or it has an activation)
                 if not is_final_block or block.names["act_out"] != block.names["linear_out"]:
                    self.ordered_stage_names_for_plot.append(block.names["act_out"])

    def forward(self, x, record_activations=False):
        x_flattened = x.view(x.size(0), -1)
        all_recorded_stages = {}
        if record_activations: all_recorded_stages["Input"] = x_flattened.detach().cpu()

        current_features = x_flattened
        for block_idx, block in enumerate(self.blocks):
            current_features, block_stages = block(current_features, record_activations)
            if record_activations: 
                # Prefix block_stages keys with B{idx} if needed for very deep NNs to ensure uniqueness, but names are H{idx} based
                all_recorded_stages.update(block_stages)
        
        return current_features, all_recorded_stages

# --- 3. Data Handling ---
def get_cifar10_dataloaders(batch_size=128, num_workers=2, data_root='./data', use_subset=False):
    os.makedirs(data_root, exist_ok=True)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    try: 
        trainset_full = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
        testset_full = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)
    except Exception as e:
        print(f"Failed to download/load CIFAR10: {e}. Trying without download flag.")
        trainset_full = torchvision.datasets.CIFAR10(root=data_root, train=True, download=False, transform=transform)
        testset_full = torchvision.datasets.CIFAR10(root=data_root, train=False, download=False, transform=transform)

    if use_subset: # For faster development/testing
        train_indices = list(range(0, len(trainset_full), 200)) # Approx. 250 samples
        val_indices = list(range(0, len(testset_full), 200))   # Approx. 50 samples
        trainset = Subset(trainset_full, train_indices)
        testset = Subset(testset_full, val_indices)
        print(f"USING SUBSET of CIFAR10: {len(trainset)} train, {len(testset)} val samples.")
    else:
        trainset, testset = trainset_full, testset_full

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    testloader = DataLoader(testset, batch_size=batch_size*2, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    return trainloader, testloader

# --- 4. Training and Metric Logging ---
def log_metrics_for_epoch(model, fixed_val_batch_on_device, epoch_num, model_config_name, metrics_list_accumulator):
    model.eval()
    with torch.no_grad():
        _, recorded_stages_dict = model(fixed_val_batch_on_device, record_activations=True)
        
        if DEBUG_VERBOSE: print(f"\n--- Logging metrics for Epoch {epoch_num}, Config: {model_config_name} ---")
        
        summary_parts = []
        for stage_name, acts_cpu in recorded_stages_dict.items():
            if DEBUG_VERBOSE: print(f" Processing stage: {stage_name}, shape: {acts_cpu.shape}, mean: {acts_cpu.mean():.2f}, std: {acts_cpu.std():.2f}")
            eff_rank, perc_frozen, perc_duplicates = np.nan, np.nan, np.nan
            
            if acts_cpu.numel() > 0 and acts_cpu.shape[0] > 1 and acts_cpu.shape[1] > 0:
                eff_rank = compute_effective_rank_empirical(acts_cpu, stage_name)
                perc_duplicates = compute_duplicate_neurons_stats(acts_cpu, stage_name_for_debug=stage_name)

                if stage_name.endswith("_ActIn"): # This is a pre-activation stage for frozen calculation
                    perc_frozen = compute_frozen_stats(acts_cpu, model.activation_name_str_for_frozen, stage_name_for_debug=stage_name)
                    # Log this under the corresponding _Act stage name for easier merging
                    target_act_out_name = stage_name.replace("_ActIn", "_Act")
                    # It's possible the target_act_out_name isn't in recorded_stages_dict if it's the output layer with Identity activation
                    # So, we log this as a separate potential piece of info for that layer.
                    metrics_list_accumulator.append({
                        'model_config': model_config_name, 'epoch': epoch_num, 'layer_name': target_act_out_name,
                        'eff_rank': np.nan, 'perc_frozen': perc_frozen, 'perc_duplicates': np.nan
                    })
                    if DEBUG_VERBOSE: print(f"  Logged frozen={perc_frozen:.2f} for {target_act_out_name} (from {stage_name})")

                # Log ER and Duplicates for the current stage (e.g. _Lin, _Nrm, _Act outputs)
                metrics_list_accumulator.append({
                    'model_config': model_config_name, 'epoch': epoch_num, 'layer_name': stage_name,
                    'eff_rank': eff_rank, 
                    'perc_frozen': np.nan, # This will be NaN here; filled by agg if an _ActIn entry exists for its _Act counterpart
                    'perc_duplicates': perc_duplicates
                })
                if DEBUG_VERBOSE: print(f"  Logged ER={eff_rank:.2f}, Duplicates={perc_duplicates:.2f} for {stage_name}")


                if "_Act" in stage_name and not stage_name.endswith("_ActIn"): 
                    summary_parts.append(f"{stage_name.split('_')[0]}ER:{eff_rank:.1f}")
            elif DEBUG_VERBOSE:
                print(f"  Skipping metrics for stage {stage_name} due to insufficient data (numel={acts_cpu.numel()}, shape={acts_cpu.shape})")


    print(f"E{epoch_num} [{model_config_name}] Metric Summary (ERs): {' '.join(summary_parts[:4])}...")
    model.train()

def train_model_with_metric_logging(
    model_config_name, model, train_loader, val_loader, 
    optimizer, criterion, num_epochs, device, 
    log_metrics_every_n_epochs, eval_batch_size):
    
    all_metrics_raw = [] 
    
    # Prepare fixed validation batch
    val_iter = iter(val_loader)
    fixed_val_imgs_list = []
    num_batches_to_fetch = max(1, eval_batch_size // (val_loader.batch_size if val_loader.batch_size and val_loader.batch_size > 0 else eval_batch_size))
    try:
        for _ in range(num_batches_to_fetch):
            fixed_val_imgs_list.append(next(val_iter)[0])
    except StopIteration:
        if DEBUG_VERBOSE and not fixed_val_imgs_list: print("Val loader exhausted before fetching any batch for metrics.")
    
    if not fixed_val_imgs_list : 
        print("Warning: val_loader too small or empty. Using one training batch for metric eval.")
        try:
            fixed_val_imgs_list.append(next(iter(train_loader))[0])
        except StopIteration:
             print("ERROR: Training loader also empty. Cannot get eval batch.")
             return pd.DataFrame() # Return empty if no data
        
    fixed_val_batch_on_device = torch.cat(fixed_val_imgs_list, dim=0)[:eval_batch_size].to(device)
    if DEBUG_VERBOSE: print(f"Metric eval batch size: {fixed_val_batch_on_device.shape[0]} on {device}.")

    if fixed_val_batch_on_device.nelement() > 0:
        log_metrics_for_epoch(model, fixed_val_batch_on_device, 0, model_config_name, all_metrics_raw)

    for epoch in range(num_epochs):
        model.train()
        prog_bar = tqdm(train_loader, desc=f"E{epoch+1}/{num_epochs} [{model_config_name}]", leave=False)
        for inputs, labels in prog_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs, _ = model(inputs, record_activations=False)
            loss = criterion(outputs, labels)
            loss.backward(); optimizer.step()
            prog_bar.set_postfix({'loss': f'{loss.item():.3f}'})
        
        if (epoch + 1) % log_metrics_every_n_epochs == 0 and fixed_val_batch_on_device.nelement() > 0:
            log_metrics_for_epoch(model, fixed_val_batch_on_device, epoch + 1, model_config_name, all_metrics_raw)
    
    df_raw = pd.DataFrame(all_metrics_raw)
    if DEBUG_VERBOSE and not df_raw.empty:
        raw_csv_path = os.path.join(FIGURE_DIR, f"debug_raw_metrics_{model_config_name.replace(' ', '_')}_{model.activation_name_str_for_frozen}.csv")
        df_raw.to_csv(raw_csv_path, index=False)
        print(f"Saved RAW unaggregated metrics to {raw_csv_path}")

    # Consolidate metrics: Group by and take the first non-NaN value for each metric.
    # This merges the separately logged 'perc_frozen' (from _ActIn, remapped to _Act) 
    # with other metrics for the same effective layer_name.
    if not df_raw.empty:
        final_df = df_raw.groupby(['model_config', 'epoch', 'layer_name'], as_index=False).agg(
            # For each column in the group, take the first non-NaN value encountered.
            # This relies on the fact that for a given layer (e.g. H1_Act), ER/Duplicates are logged once with NaNs for frozen,
            # and frozen is logged once with NaNs for ER/Duplicates.
            lambda x: x.dropna().unique()[0] if len(x.dropna().unique()) > 0 else np.nan
        )
        return final_df.dropna(subset=['eff_rank', 'perc_frozen', 'perc_duplicates'], how='all')
    else:
        return pd.DataFrame()


# --- 5. Plotting All Empirical Metrics --- (Plotting function remains largely the same as previous version)
def plot_all_empirical_metrics_figure(final_metrics_df, main_activation_name, ordered_layer_names_for_plot, configurations_dict):
    if final_metrics_df.empty: print("No metrics to plot."); return

    unique_epochs = sorted(final_metrics_df['epoch'].dropna().unique())
    if not unique_epochs: print("No epochs in data for plotting."); return
    
    epochs_to_plot = [unique_epochs[0]] 
    if len(unique_epochs) > 2: epochs_to_plot.append(unique_epochs[len(unique_epochs) // 2]) 
    if len(unique_epochs) > 1: epochs_to_plot.append(unique_epochs[-1]) 
    epochs_to_plot = sorted(list(set(epochs_to_plot)))

    num_epoch_subplots = len(epochs_to_plot)
    fig, axs = plt.subplots(3, num_epoch_subplots, 
                            figsize=(2.8 * num_epoch_subplots, 6.5), # Adjusted for 3 rows, compact
                            sharex='col') 
    if num_epoch_subplots == 1: axs = axs.reshape(3, 1) if axs.ndim == 1 else axs # ensure 2D

    config_colors = {"MLP NoNorm": "black", "MLP BatchNorm": "dodgerblue", "MLP LayerNorm": "red"}
    # Ensure all names in ordered_layer_names_for_plot are potential x-ticks
    layer_name_to_idx = {name: i for i, name in enumerate(ordered_layer_names_for_plot)}


    for col_idx, epoch in enumerate(epochs_to_plot):
        epoch_data = final_metrics_df[final_metrics_df['epoch'] == epoch]
        ax_er, ax_fr, ax_dp = axs[0, col_idx], axs[1, col_idx], axs[2, col_idx]
        ax_er.set_title(f"Epoch {epoch}", fontsize=plt.rcParams['axes.titlesize'])

        for config_name_key in configurations_dict.keys():
            color = config_colors.get(config_name_key, "grey") # Fallback color
            config_epoch_data = epoch_data[epoch_data['model_config'] == config_name_key].copy()
            if config_epoch_data.empty: continue

            # Map layer names to indices; only include layers present in ordered_layer_names_for_plot
            config_epoch_data['plot_x_idx'] = config_epoch_data['layer_name'].map(layer_name_to_idx)
            plot_data_for_config = config_epoch_data.dropna(subset=['plot_x_idx']).sort_values(by='plot_x_idx')


            # Effective Rank (plot for all layer_names that have ER data and are in ordered_plot_names)
            er_data = plot_data_for_config.dropna(subset=['eff_rank'])
            if not er_data.empty:
                ax_er.plot(er_data['plot_x_idx'], er_data['eff_rank'], color=color, linestyle='-', marker='o', markersize=2.5, label=config_name_key if col_idx==0 else None)

            # Frozen % (plot for _Act layers that have perc_frozen data)
            # Perc_frozen is logged against _Act names after re-association from _ActIn
            frozen_data = plot_data_for_config[plot_data_for_config['layer_name'].str.contains("_Act", na=False)].dropna(subset=['perc_frozen'])
            if not frozen_data.empty:
                ax_fr.plot(frozen_data['plot_x_idx'], frozen_data['perc_frozen'], color=color, linestyle='--', marker='x', markersize=3, label=config_name_key if col_idx==0 else None)

            # Duplicate % (plot for all layers that have perc_duplicates data)
            dup_data = plot_data_for_config.dropna(subset=['perc_duplicates'])
            if not dup_data.empty:
                 ax_dp.plot(dup_data['plot_x_idx'], dup_data['perc_duplicates'], color=color, linestyle=':', marker='s', markersize=2.5, label=config_name_key if col_idx==0 else None)

        if col_idx == 0:
            ax_er.set_ylabel("Eff. Rank")
            ax_fr.set_ylabel("% Frozen")
            ax_dp.set_ylabel("% Duplicate")
            for ax_row in [ax_er, ax_fr, ax_dp]: ax_row.legend(loc='best') # 'best' or specific
        
        for ax_row in [ax_er, ax_fr, ax_dp]: ax_row.grid(True, linestyle=':', alpha=0.6)
        ax_er.set_ylim(bottom=0); ax_fr.set_ylim(0, 100); ax_dp.set_ylim(0, 100)
        
        ax_dp.set_xticks(list(layer_name_to_idx.values()))
        xticklabels_short = [name.replace("_Lin", "L").replace("_Nrm", "N").replace("_ActIn","Pre").replace("_Act", "A").replace("Out_Logits","Out") for name in ordered_layer_names_for_plot]
        ax_dp.set_xticklabels(xticklabels_short, rotation=45, ha='right')
        if col_idx == num_epoch_subplots // 2 : ax_dp.set_xlabel("Layer Stage")
    
    fig.suptitle(f"Empirical Analysis (Activation: {main_activation_name.upper()})", fontsize=plt.rcParams['figure.titlesize'])
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for suptitle and x-labels
    save_path = os.path.join(FIGURE_DIR, f"empirical_all_metrics_{main_activation_name}.pdf")
    plt.savefig(save_path); print(f"Saved: {save_path}"); plt.show()


# --- 6. Main Experiment Orchestration ---
CONFIGURATIONS = { 
    "MLP NoNorm": {"norm_type": None},
    "MLP BatchNorm": {"norm_type": "batchnorm"},
    "MLP LayerNorm": {"norm_type": "layernorm"}
}

def run_main_experiment(
    num_epochs=15, log_metrics_every_n_epochs=1, 
    mlp_hidden_layers=[128, 128, 64], activation_name="ReLU",
    batch_size=256, eval_batch_size_config=512, use_subset_data=False):

    print(f"Device: {DEVICE}. Activation: {activation_name}. Epochs: {num_epochs}. Subset data: {use_subset_data}")
    train_loader, val_loader = get_cifar10_dataloaders(batch_size=batch_size, use_subset=use_subset_data)
    input_dim = 32 * 32 * 3; num_classes = 10
    layer_dims = [input_dim] + mlp_hidden_layers + [num_classes]
    all_dfs = []
    
    # Instantiate a sample model to get the canonical ordered_stage_names_for_plot
    # This ensures all runs use the same set of x-axis ticks/labels if architectures are identical
    sample_model_for_names = ConfigurableMLP(layer_dims, activation_name, list(CONFIGURATIONS.values())[0]["norm_type"])
    ordered_plot_names = sample_model_for_names.ordered_stage_names_for_plot
    if DEBUG_VERBOSE: print(f"Canonical layer stages for plotting: {ordered_plot_names}")
    del sample_model_for_names


    for config_name, params in CONFIGURATIONS.items():
        if DEBUG_VERBOSE: print(f"\n--- Starting Config: {config_name} ---")
        model = ConfigurableMLP(layer_dims, activation_name, params["norm_type"]).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        metrics_df = train_model_with_metric_logging(
            config_name, model, train_loader, val_loader, optimizer, criterion, 
            num_epochs, DEVICE, log_metrics_every_n_epochs, eval_batch_size_config
        )
        all_dfs.append(metrics_df)
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    if not all_dfs or all(df.empty for df in all_dfs):
        print("No metrics data collected across all configurations."); return
    
    final_df = pd.concat(all_dfs, ignore_index=True)
    if final_df.empty:
        print("Concatenated DataFrame is empty. No data to plot or save.")
        return

    data_path = os.path.join(FIGURE_DIR, f"empirical_metrics_data_AGGREGATED_{activation_name}.csv")
    final_df.to_csv(data_path, index=False); print(f"Saved AGGREGATED data: {data_path}")
    
    plot_all_empirical_metrics_figure(final_df, activation_name, ordered_plot_names, CONFIGURATIONS)

if __name__ == "__main__":
    torch.manual_seed(42); np.random.seed(42)
    
    # Set use_subset_data=True for quick tests, False for full run
    quick_test = False # Change to False for paper figures

    run_main_experiment(
        activation_name="ReLU", 
        num_epochs=3 if quick_test else 15, 
        mlp_hidden_layers=[64, 32] if quick_test else [128,128,64],
        log_metrics_every_n_epochs=1,
        use_subset_data=quick_test
    )
    
    # print("\n" + "="*50 + "\nNOW RUNNING FOR TANH (EXAMPLE)\n" + "="*50)
    # run_main_experiment(
    #    activation_name="Tanh", 
    #    num_epochs=3 if quick_test else 15, 
    #    mlp_hidden_layers=[64,32] if quick_test else [128,128,64],
    #    log_metrics_every_n_epochs=1,
    #    use_subset_data=quick_test
    # )
    print(f"\nAll empirical figures and data saved to {FIGURE_DIR}")