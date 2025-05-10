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

# --- Matplotlib Styling for Paper Quality ---
try:
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({
        'font.family': 'serif',
        'text.usetex': False,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'font.size': 8, # Smaller base for potentially crowded empirical plots
        'axes.labelsize': 8,
        'axes.titlesize': 9,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7,
        'figure.titlesize': 10,
    })
except OSError:
    print("seaborn-v0_8-paper style not found, using rcParams for basic styling.")
    plt.rcParams.update({
        'font.size': 8, 
        'axes.titlesize': 9,
        'axes.labelsize': 8,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7,
        'figure.titlesize': 10,
        'lines.linewidth': 1.2,
        'lines.markersize': 3,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'font.family': 'serif',
        'text.usetex': False
    })

# --- Global Parameters & Figure Saving ---
FIGURE_DIR = "./figures/"
os.makedirs(FIGURE_DIR, exist_ok=True)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- 1. Helper Functions (Metrics) ---
def compute_effective_rank_empirical(activations_batch_cpu):
    if activations_batch_cpu.ndim == 1:
        return 1.0 if activations_batch_cpu.shape[0] > 1 else 0.0
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
        # Ensure there's still enough rank in centered_activations for cov
        if centered_activations.shape[0] < centered_activations.shape[1]: # more features than samples
            # Use feature matrix directly if N < D after centering for covariance computation robustness
             cov_matrix = torch.cov(centered_activations) # Cov over samples
        else:
             cov_matrix = torch.cov(centered_activations.T) # Cov over features

        if torch.isnan(cov_matrix).any(): cov_matrix = torch.nan_to_num(cov_matrix, nan=0.0)
        s_unnormalized = torch.linalg.svdvals(cov_matrix)
    except RuntimeError: return 1.0 

    sum_s = torch.sum(s_unnormalized)
    if sum_s < 1e-12: return 0.0
    s_norm_for_entropy = s_unnormalized / sum_s
    s_norm_for_entropy = s_norm_for_entropy[s_norm_for_entropy > 1e-15]
    if len(s_norm_for_entropy) == 0: return 0.0
    entropy = -torch.sum(s_norm_for_entropy * torch.log(s_norm_for_entropy))
    return torch.exp(entropy).item()

def compute_frozen_stats(pre_activations_batch_cpu, activation_type_str, frozen_deriv_thresh=1e-3):
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
        frozen_map = derivative < frozen_deriv_thresh # Derivative is small
    else: 
        return 0.0 
    
    percentage_total_frozen_instances = frozen_map.float().mean().item() * 100.0
    return percentage_total_frozen_instances

def compute_duplicate_neurons_stats(activations_batch_cpu, corr_threshold=0.95):
    if activations_batch_cpu.ndim < 2 or activations_batch_cpu.shape[1] < 2: return 0.0
    
    std_devs = torch.std(activations_batch_cpu, dim=0)
    valid_features_mask = std_devs > 1e-5
    if valid_features_mask.sum() < 2: return 0.0
        
    activations_valid = activations_batch_cpu[:, valid_features_mask]
    num_valid_features = activations_valid.shape[1]

    try:
        corr_matrix = torch.corrcoef(activations_valid.T)
        if torch.isnan(corr_matrix).any(): corr_matrix = torch.nan_to_num(corr_matrix, nan=0.0)
    except RuntimeError: return 0.0

    abs_corr_matrix = torch.abs(corr_matrix)
    upper_tri_corr = torch.triu(abs_corr_matrix, diagonal=1)
    
    highly_correlated_rows_mask = torch.any(upper_tri_corr > corr_threshold, dim=1)
    highly_correlated_cols_mask = torch.any(upper_tri_corr > corr_threshold, dim=0)
    duplicate_mask_final = highly_correlated_rows_mask | highly_correlated_cols_mask
    
    return duplicate_mask_final.float().mean().item() * 100.0 if num_valid_features > 0 else 0.0

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
            "linear_out": f"{layer_prefix}_Lin",    # Output of linear
            "norm_out": f"{layer_prefix}_Nrm",      # Output of norm (if exists)
            "act_in": f"{layer_prefix}_ActIn",     # Input to activation (for frozen)
            "act_out": f"{layer_prefix}_Act"       # Output of activation
        }
        if is_output_layer_block:
            self.names = {"linear_out": "Out_Logits", "act_out": "Out_Logits"} # No norm/act for logits

        if not is_output_layer_block and norm_type:
            if norm_type.lower() == "batchnorm": self.norm = nn.BatchNorm1d(out_dim)
            elif norm_type.lower() == "layernorm": self.norm = nn.LayerNorm(out_dim)

        self.act = self.act_fn_instance if not is_output_layer_block else nn.Identity()

    def forward(self, x, record_activations=False):
        recorded_stages = {}
        
        current_features = self.linear(x)
        if record_activations: recorded_stages[self.names["linear_out"]] = current_features.detach().cpu()
        
        pre_act_input = current_features # Input to activation if no norm
        if self.names["linear_out"] != self.names["act_out"]: # If not output layer
            if self.norm:
                current_features = self.norm(current_features)
                if record_activations: recorded_stages[self.names["norm_out"]] = current_features.detach().cpu()
                pre_act_input = current_features # Update pre_act_input if norm exists
            
            if record_activations: recorded_stages[self.names["act_in"]] = pre_act_input.detach().cpu()
            current_features = self.act(current_features)
            if record_activations: recorded_stages[self.names["act_out"]] = current_features.detach().cpu()
        
        return current_features, recorded_stages

class ConfigurableMLP(nn.Module):
    def __init__(self, layer_dims, activation_name_str="relu", norm_type=None):
        super().__init__()
        self.activation_name_str_for_frozen = activation_name_str
        
        act_fn_instance, act_display_name = get_activation_fn_and_name(activation_name_str)
        self.ordered_stage_names_for_plot = ["Input"]

        self.blocks = nn.ModuleList()
        for i in range(len(layer_dims) - 1):
            is_final_block = (i == len(layer_dims) - 2)
            block = MLPBlock(layer_dims[i], layer_dims[i+1], norm_type, 
                             act_fn_instance, act_display_name, is_final_block, i)
            self.blocks.append(block)
            # Collect all unique stage names from blocks in order
            for stage_key in ["linear_out", "norm_out", "act_out"]: # Order matters for plot
                if stage_key in block.names and block.names[stage_key] not in self.ordered_stage_names_for_plot:
                    self.ordered_stage_names_for_plot.append(block.names[stage_key])
            
    def forward(self, x, record_activations=False):
        x_flattened = x.view(x.size(0), -1)
        all_recorded_stages = {}
        if record_activations: all_recorded_stages["Input"] = x_flattened.detach().cpu()

        current_features = x_flattened
        for block in self.blocks:
            current_features, block_stages = block(current_features, record_activations)
            if record_activations: all_recorded_stages.update(block_stages)
        
        return current_features, all_recorded_stages

# --- 3. Data Handling ---
def get_cifar10_dataloaders(batch_size=128, num_workers=2, data_root='./data'):
    os.makedirs(data_root, exist_ok=True)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    try: # Attempt to download if not present
        trainset = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)
    except Exception as e:
        print(f"Failed to download/load CIFAR10: {e}. Trying without download flag if data exists.")
        trainset = torchvision.datasets.CIFAR10(root=data_root, train=True, download=False, transform=transform)
        testset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=False, transform=transform)

    # For faster testing, use a subset
    # train_indices = list(range(0, len(trainset), len(trainset) // (batch_size * 10))) # approx 10 batches
    # val_indices = list(range(0, len(testset), len(testset) // (batch_size * 5)))   # approx 5 batches
    # trainset = Subset(trainset, train_indices)
    # testset = Subset(testset, val_indices)
    # print(f"Using subset of CIFAR10: {len(trainset)} train, {len(testset)} val samples.")


    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    testloader = DataLoader(testset, batch_size=batch_size*2, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    return trainloader, testloader

# --- 4. Training and Metric Logging ---
def log_metrics_for_epoch(model, fixed_val_batch_on_device, epoch_num, model_config_name, metrics_list_accumulator):
    model.eval()
    with torch.no_grad():
        _, recorded_stages_dict = model(fixed_val_batch_on_device, record_activations=True)
        
        summary_parts = []
        for stage_name, acts_cpu in recorded_stages_dict.items():
            eff_rank, perc_frozen, perc_duplicates = np.nan, np.nan, np.nan
            
            if acts_cpu.numel() > 0 and acts_cpu.shape[0] > 1: # Min samples
                # ER and Duplicates are computed on any output stage
                eff_rank = compute_effective_rank_empirical(acts_cpu)
                perc_duplicates = compute_duplicate_neurons_stats(acts_cpu)

                # Frozen is computed on ActIn stages
                if stage_name.endswith("_ActIn"):
                    perc_frozen = compute_frozen_stats(acts_cpu, model.activation_name_str_for_frozen)
                    # Log this under the corresponding _Act stage name for easier merging in pandas
                    target_act_out_name = stage_name.replace("_ActIn", "_Act")
                    metrics_list_accumulator.append({
                        'model_config': model_config_name, 'epoch': epoch_num, 'layer_name': target_act_out_name,
                        'eff_rank': np.nan, 'perc_frozen': perc_frozen, 'perc_duplicates': np.nan
                    })
                
                # Log ER and Duplicates for the current stage (could be _Lin, _Nrm, _Act)
                metrics_list_accumulator.append({
                    'model_config': model_config_name, 'epoch': epoch_num, 'layer_name': stage_name,
                    'eff_rank': eff_rank, 'perc_frozen': np.nan, # Frozen is logged separately above
                    'perc_duplicates': perc_duplicates
                })

                if "Act" in stage_name and not stage_name.endswith("_ActIn"): 
                    summary_parts.append(f"{stage_name.split('_')[0]}ER:{eff_rank:.1f}")

    print(f"E{epoch_num} [{model_config_name}] Ranks: {' '.join(summary_parts[:3])}...")
    model.train()

def train_model_with_metric_logging(
    model_config_name, model, train_loader, val_loader, 
    optimizer, criterion, num_epochs, device, 
    log_metrics_every_n_epochs, eval_batch_size):
    
    all_metrics_raw = [] 
    val_iter = iter(val_loader)
    fixed_val_imgs_list = [next(val_iter)[0] for _ in range(max(1, eval_batch_size // val_loader.batch_size if val_loader.batch_size else 1))]
    if not fixed_val_imgs_list : fixed_val_imgs_list.append(next(iter(train_loader))[0]) # Fallback
    fixed_val_batch_on_device = torch.cat(fixed_val_imgs_list, dim=0)[:eval_batch_size].to(device)
    print(f"Metric eval batch size: {fixed_val_batch_on_device.shape[0]} on {device}.")

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
    
    # Consolidate metrics (especially perc_frozen which is logged separately)
    df = pd.DataFrame(all_metrics_raw)
    # Group by and take the first non-NaN value for each metric.
    # This merges the separately logged 'perc_frozen' with other metrics for the same layer_name.
    final_df = df.groupby(['model_config', 'epoch', 'layer_name'], as_index=False).agg(
        lambda x: x.dropna().iloc[0] if x.dropna().any() else np.nan
    )
    return final_df.dropna(subset=['eff_rank', 'perc_frozen', 'perc_duplicates'], how='all')


# --- 5. Plotting All Empirical Metrics ---
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
                            figsize=(2.5 * num_epoch_subplots, 6.0), # Adjusted for 3 rows, compact
                            sharex='col') 
    if num_epoch_subplots == 1: axs = axs.reshape(3, 1)

    config_colors = {"MLP NoNorm": "black", "MLP BatchNorm": "dodgerblue", "MLP LayerNorm": "red"}
    layer_name_to_idx = {name: i for i, name in enumerate(ordered_layer_names_for_plot)}

    for col_idx, epoch in enumerate(epochs_to_plot):
        epoch_data = final_metrics_df[final_metrics_df['epoch'] == epoch]
        ax_er, ax_fr, ax_dp = axs[0, col_idx], axs[1, col_idx], axs[2, col_idx]
        ax_er.set_title(f"Epoch {epoch}", fontsize=plt.rcParams['axes.titlesize'])

        for config_name_key in configurations_dict.keys():
            color = config_colors.get(config_name_key, "grey")
            config_epoch_data = epoch_data[epoch_data['model_config'] == config_name_key].copy()
            if config_epoch_data.empty: continue

            config_epoch_data['plot_x_idx'] = config_epoch_data['layer_name'].map(layer_name_to_idx)
            plot_data = config_epoch_data.dropna(subset=['plot_x_idx']).sort_values(by='plot_x_idx')

            er_data = plot_data.dropna(subset=['eff_rank'])
            if not er_data.empty:
                ax_er.plot(er_data['plot_x_idx'], er_data['eff_rank'], color=color, linestyle='-', marker='o', label=config_name_key if col_idx==0 else None)

            frozen_data = plot_data[plot_data['layer_name'].str.contains("_Act", na=False)].dropna(subset=['perc_frozen']) # Only plot for _Act layers
            if not frozen_data.empty:
                ax_fr.plot(frozen_data['plot_x_idx'], frozen_data['perc_frozen'], color=color, linestyle='--', marker='x', label=config_name_key if col_idx==0 else None)

            dup_data = plot_data.dropna(subset=['perc_duplicates'])
            if not dup_data.empty:
                 ax_dp.plot(dup_data['plot_x_idx'], dup_data['perc_duplicates'], color=color, linestyle=':', marker='s', label=config_name_key if col_idx==0 else None)

        if col_idx == 0:
            ax_er.set_ylabel("Eff. Rank")
            ax_fr.set_ylabel("% Frozen")
            ax_dp.set_ylabel("% Duplicate")
            for ax_row in [ax_er, ax_fr, ax_dp]: ax_row.legend(loc='best')
        
        for ax_row in [ax_er, ax_fr, ax_dp]: ax_row.grid(True, linestyle=':', alpha=0.6)
        ax_er.set_ylim(bottom=0); ax_fr.set_ylim(0, 100); ax_dp.set_ylim(0, 100)
        
        ax_dp.set_xticks(list(layer_name_to_idx.values()))
        xticklabels_short = [name.replace("_Lin", "L").replace("_Nrm", "N").replace("_Act", "A").replace("Out_Logits","Out") for name in ordered_layer_names_for_plot]
        ax_dp.set_xticklabels(xticklabels_short, rotation=45, ha='right')
        if col_idx == num_epoch_subplots // 2 : ax_dp.set_xlabel("Layer Stage")
    
    fig.suptitle(f"Empirical Analysis (Activation: {main_activation_name.upper()})", fontsize=plt.rcParams['figure.titlesize'])
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
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
    mlp_hidden_layers=[128, 128], activation_name="ReLU",
    batch_size=256, eval_batch_size_config=512):

    print(f"Device: {DEVICE}. Activation: {activation_name}")
    train_loader, val_loader = get_cifar10_dataloaders(batch_size=batch_size)
    input_dim = 32 * 32 * 3; num_classes = 10
    layer_dims = [input_dim] + mlp_hidden_layers + [num_classes]
    all_dfs = []
    
    sample_model = ConfigurableMLP(layer_dims, activation_name, list(CONFIGURATIONS.values())[0]["norm_type"])
    ordered_plot_names = sample_model.ordered_stage_names_for_plot
    del sample_model

    for config_name, params in CONFIGURATIONS.items():
        print(f"\n--- Config: {config_name} ---")
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
        print("No metrics data collected."); return
    
    final_df = pd.concat(all_dfs, ignore_index=True)
    data_path = os.path.join(FIGURE_DIR, f"empirical_metrics_data_{activation_name}.csv")
    final_df.to_csv(data_path, index=False); print(f"Saved data: {data_path}")
    plot_all_empirical_metrics_figure(final_df, activation_name, ordered_plot_names, CONFIGURATIONS)

if __name__ == "__main__":
    torch.manual_seed(42); np.random.seed(42)
    run_main_experiment(activation_name="ReLU", num_epochs=15, mlp_hidden_layers=[128,128,64]) # Example
    # print("\n" + "="*50 + "\nNOW RUNNING FOR TANH (EXAMPLE)\n" + "="*50)
    # run_main_experiment(activation_name="Tanh", num_epochs=3, mlp_hidden_layers=[64,32]) # Quick Tanh test
    print(f"\nAll empirical figures and data saved to {FIGURE_DIR}")