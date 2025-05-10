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
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

# --- Settings & Configuration ---
@dataclass
class ExperimentSettings:
    FIGURE_DIR: str = "./figures/"
    DEVICE: torch.device = field(default_factory=lambda: torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    DEBUG_VERBOSE: bool = True 
    QUICK_TEST_MODE: bool = False 
    SEED: int = 42

    # Experimental parameters
    dataset_name: str = "MNIST"
    num_epochs: int = 15
    log_metrics_every_n_epochs: int = 1
    mlp_hidden_layers: List[int] = field(default_factory=lambda: [512,512,512,512])
    default_activation_name: str = "ReLU"

    # Training hyperparameters
    batch_size: int = 256
    learning_rate: float = 1e-3
    eval_batch_size: int = 512

    # Dataset specific
    input_dim: int = 784 
    num_classes: int = 10

    # Metric calculation thresholds
    frozen_relu_threshold: float = 1e-6 # Pre-activation value for ReLU to be considered having near-zero derivative
    frozen_tanh_deriv_threshold: float = 1e-3 # Derivative value below which Tanh is considered frozen
    frozen_sigmoid_deriv_threshold: float = 1e-3 # Derivative value below which Sigmoid is considered frozen
    frozen_feature_batch_threshold: float = 0.95 # << --- NEW: Min fraction of samples in a batch for a feature to be "batch-frozen"
    
    duplicate_corr_threshold: float = 0.95
    duplicate_min_std_dev: float = 1e-5 

    # Plotting settings
    FONT_FAMILY: str = 'serif'
    FONT_SIZE_BASE: int = 8
    TITLE_FONT_SIZE_FACTOR: float = 1.1
    LABEL_FONT_SIZE_FACTOR: float = 1.0
    TICK_FONT_SIZE_FACTOR: float = 0.9
    LEGEND_FONT_SIZE_FACTOR: float = 0.9
    FIG_DPI: int = 300

    def __post_init__(self):
        if self.QUICK_TEST_MODE:
            self.num_epochs = 3
            self.mlp_hidden_layers = [64, 32]
            self.log_metrics_every_n_epochs = 1
            self.batch_size = 128
            self.eval_batch_size = 256
            if self.DEBUG_VERBOSE: print(f"--- QUICK TEST MODE ENABLED ({self.dataset_name}) ---")
        
        if self.dataset_name == "MNIST":
            self.input_dim = 28 * 28 * 1
            self.num_classes = 10
        elif self.dataset_name == "CIFAR10":
            self.input_dim = 32 * 32 * 3
            self.num_classes = 10
        else:
            raise ValueError(f"Unsupported dataset_name: {self.dataset_name}")

@dataclass
class ModelConfig:
    name: str
    norm_type: Optional[str]
    activation_name: Optional[str] = None

# Initialize global settings
SETTINGS = ExperimentSettings()

# --- Matplotlib Styling ---
def setup_matplotlib_style(s: ExperimentSettings):
    try:
        plt.style.use('seaborn-v0_8-paper')
        base_params = {
            'font.family': s.FONT_FAMILY, 'text.usetex': False, 
            'figure.dpi': s.FIG_DPI, 'savefig.dpi': s.FIG_DPI,
            'savefig.bbox': 'tight', 'font.size': s.FONT_SIZE_BASE,
            'axes.labelsize': s.FONT_SIZE_BASE * s.LABEL_FONT_SIZE_FACTOR,
            'axes.titlesize': s.FONT_SIZE_BASE * s.TITLE_FONT_SIZE_FACTOR,
            'xtick.labelsize': s.FONT_SIZE_BASE * s.TICK_FONT_SIZE_FACTOR,
            'ytick.labelsize': s.FONT_SIZE_BASE * s.TICK_FONT_SIZE_FACTOR,
            'legend.fontsize': s.FONT_SIZE_BASE * s.LEGEND_FONT_SIZE_FACTOR,
            'figure.titlesize': s.FONT_SIZE_BASE * s.TITLE_FONT_SIZE_FACTOR * 1.1,
        }
    except OSError:
        print("seaborn-v0_8-paper style not found, using rcParams for basic styling.")
        base_params = {
            'font.size': s.FONT_SIZE_BASE, 
            'axes.titlesize': s.FONT_SIZE_BASE * s.TITLE_FONT_SIZE_FACTOR,
            'axes.labelsize': s.FONT_SIZE_BASE * s.LABEL_FONT_SIZE_FACTOR,
            'xtick.labelsize': s.FONT_SIZE_BASE * s.TICK_FONT_SIZE_FACTOR,
            'ytick.labelsize': s.FONT_SIZE_BASE * s.TICK_FONT_SIZE_FACTOR,
            'legend.fontsize': s.FONT_SIZE_BASE * s.LEGEND_FONT_SIZE_FACTOR,
            'figure.titlesize': s.FONT_SIZE_BASE * s.TITLE_FONT_SIZE_FACTOR * 1.1,
            'lines.linewidth': 1.2, 'lines.markersize': 3,
            'figure.dpi': s.FIG_DPI, 'savefig.dpi': s.FIG_DPI,
            'savefig.bbox': 'tight', 'font.family': s.FONT_FAMILY, 'text.usetex': False
        }
    plt.rcParams.update(base_params)

setup_matplotlib_style(SETTINGS)
os.makedirs(SETTINGS.FIGURE_DIR, exist_ok=True)
torch.manual_seed(SETTINGS.SEED); np.random.seed(SETTINGS.SEED)

# --- Metric Calculation Functions (Updated Signatures) ---
def compute_effective_rank_empirical(activations_batch_cpu, stage_name_for_debug=""):
    if activations_batch_cpu.ndim == 1: return 1.0 if activations_batch_cpu.shape[0] > 1 else 0.0
    if activations_batch_cpu.shape[1] == 0: return 0.0
    if activations_batch_cpu.shape[1] == 1: return 1.0
    if activations_batch_cpu.shape[0] <= 1: return float(activations_batch_cpu.shape[1] > 0)
    std_devs = torch.std(activations_batch_cpu, dim=0); valid_features_mask = std_devs > SETTINGS.duplicate_min_std_dev
    if valid_features_mask.sum() < 2: return float(valid_features_mask.sum().item())
    activations_batch_filtered = activations_batch_cpu[:, valid_features_mask]
    if activations_batch_filtered.shape[0] <= 1 or activations_batch_filtered.shape[1] < 2: return float(activations_batch_filtered.shape[1] > 0)
    try:
        centered_activations = activations_batch_filtered - activations_batch_filtered.mean(dim=0, keepdim=True)
        if centered_activations.shape[0] < centered_activations.shape[1] and centered_activations.shape[0] > 0 : cov_matrix = torch.cov(centered_activations) 
        elif centered_activations.shape[0] >= centered_activations.shape[1] and centered_activations.shape[1] > 0: cov_matrix = torch.cov(centered_activations.T)
        else:
            if SETTINGS.DEBUG_VERBOSE: print(f"  [ER Debug {stage_name_for_debug}] Not enough data for cov. Shape: {centered_activations.shape}")
            return 0.0
        if torch.isnan(cov_matrix).any(): cov_matrix = torch.nan_to_num(cov_matrix, nan=0.0)
        s_unnormalized = torch.linalg.svdvals(cov_matrix)
    except RuntimeError as e:
        if SETTINGS.DEBUG_VERBOSE: print(f"  [ER Debug {stage_name_for_debug}] SVD RuntimeError: {e}. Filtered shape: {activations_batch_filtered.shape}")
        return 0.0 
    sum_s = torch.sum(s_unnormalized)
    if sum_s < 1e-12: return 0.0
    s_norm_for_entropy = s_unnormalized / sum_s; s_norm_for_entropy = s_norm_for_entropy[s_norm_for_entropy > 1e-15]
    if len(s_norm_for_entropy) == 0: return 0.0
    entropy = -torch.sum(s_norm_for_entropy * torch.log(s_norm_for_entropy))
    er_val = torch.exp(entropy).item()
    if SETTINGS.DEBUG_VERBOSE and (er_val > activations_batch_filtered.shape[1] + 1e-3 or er_val < 0):
        print(f"  [ER Debug {stage_name_for_debug}] Unusual ER: {er_val:.2f}, Max: {activations_batch_filtered.shape[1]}, SumS: {sum_s:.2e}")
    return min(er_val, activations_batch_filtered.shape[1])


def compute_frozen_stats(pre_activations_batch_cpu: torch.Tensor, 
                         activation_type_str: str, 
                         s: ExperimentSettings, # Pass full settings object
                         stage_name_for_debug: str = "") -> float:
    """
    Calculates the percentage of features (neurons) that are "frozen" for a batch.
    A feature is considered frozen if its derivative is below a threshold
    for more than `s.frozen_feature_batch_threshold` fraction of the samples in the batch.
    """
    if pre_activations_batch_cpu.numel() == 0 or pre_activations_batch_cpu.shape[1] == 0: return 0.0
    if pre_activations_batch_cpu.shape[0] <= 1: # Need multiple samples to check across batch
        if s.DEBUG_VERBOSE: print(f"  [Frozen Debug {stage_name_for_debug}] Not enough samples ({pre_activations_batch_cpu.shape[0]}) for batch-frozen definition.")
        return 0.0 
        
    frozen_map_per_instance = torch.zeros_like(pre_activations_batch_cpu, dtype=torch.bool) # True if (sample, feature) has low derivative
    act_str_lower = activation_type_str.lower()

    if act_str_lower == "relu": 
        frozen_map_per_instance = pre_activations_batch_cpu <= s.frozen_relu_threshold
    elif act_str_lower == "tanh": 
        tanh_x_sq = torch.tanh(pre_activations_batch_cpu)**2
        frozen_map_per_instance = tanh_x_sq > (1.0 - s.frozen_tanh_deriv_threshold)
    elif act_str_lower == "sigmoid": 
        sig_x = torch.sigmoid(pre_activations_batch_cpu)
        derivative = sig_x * (1.0 - sig_x)
        frozen_map_per_instance = derivative < s.frozen_sigmoid_deriv_threshold
    else: 
        if s.DEBUG_VERBOSE: print(f"  [Frozen Debug {stage_name_for_debug}] Frozen stats NA for {act_str_lower}, ret 0.")
        return 0.0 
    
    # For each feature, calculate the fraction of samples where its derivative is low
    # frozen_map_per_instance has shape (batch_size, num_features)
    fraction_frozen_per_feature = frozen_map_per_instance.float().mean(dim=0) # Shape: (num_features,)
    
    # A feature is "batch-frozen" if this fraction exceeds the threshold
    batch_frozen_features_mask = fraction_frozen_per_feature > s.frozen_feature_batch_threshold # Shape: (num_features,)
    
    # Percentage of features in the layer that are "batch-frozen"
    if batch_frozen_features_mask.numel() == 0: # Should not happen if pre_activations_batch_cpu.shape[1] > 0
        percentage_batch_frozen_features = 0.0
    else:
        percentage_batch_frozen_features = batch_frozen_features_mask.float().mean().item() * 100.0
    
    if s.DEBUG_VERBOSE: 
        print(f"  [Frozen Debug {stage_name_for_debug}] Act: {act_str_lower}, InShape: {pre_activations_batch_cpu.shape}")
        print(f"    Total low-deriv instances: {frozen_map_per_instance.sum().item()}/{frozen_map_per_instance.numel()}")
        print(f"    Num batch-frozen features (>{s.frozen_feature_batch_threshold*100:.0f}% of samples): {batch_frozen_features_mask.sum().item()}/{batch_frozen_features_mask.numel()}")
        print(f"    Percentage of batch-frozen features: {percentage_batch_frozen_features:.2f}%")
    return percentage_batch_frozen_features

def compute_duplicate_neurons_stats(activations_batch_cpu, 
                                    s: ExperimentSettings, 
                                    stage_name_for_debug=""):
    if activations_batch_cpu.ndim < 2 or activations_batch_cpu.shape[1] < 2: return 0.0
    std_devs = torch.std(activations_batch_cpu, dim=0)
    valid_features_mask = std_devs > s.duplicate_min_std_dev
    
    if valid_features_mask.sum() < 2:
        if s.DEBUG_VERBOSE: print(f"  [Duplicate Debug {stage_name_for_debug}] Not enough variant features ({valid_features_mask.sum().item()}). Orig: {activations_batch_cpu.shape[1]}")
        return 0.0
        
    activations_valid = activations_batch_cpu[:, valid_features_mask]
    num_valid_features = activations_valid.shape[1]
    try:
        corr_matrix = torch.corrcoef(activations_valid.T) 
        if torch.isnan(corr_matrix).any(): corr_matrix = torch.nan_to_num(corr_matrix, nan=0.0)
    except RuntimeError as e:
        if s.DEBUG_VERBOSE: print(f"  [Duplicate Debug {stage_name_for_debug}] Corrcoef RuntimeError: {e}. Valid feats: {num_valid_features}")
        return 0.0
        
    abs_corr_matrix = torch.abs(corr_matrix)
    abs_corr_matrix.fill_diagonal_(0) 
    is_duplicate_neuron = torch.any(abs_corr_matrix > s.duplicate_corr_threshold, dim=1)
    perc = is_duplicate_neuron.float().mean().item() * 100.0
    if s.DEBUG_VERBOSE: print(f"  [Duplicate Debug {stage_name_for_debug}] InShape: {activations_batch_cpu.shape}, ValidFeats: {num_valid_features}, DupFound: {is_duplicate_neuron.sum().item()}, Perc: {perc:.2f}%")
    return perc

def get_activation_fn_and_name(activation_name_str: str):
    name_lower = activation_name_str.lower()
    if name_lower == "relu": return nn.ReLU(), "ReLU"
    elif name_lower == "tanh": return nn.Tanh(), "Tanh"
    elif name_lower == "sigmoid": return nn.Sigmoid(), "Sigmoid"
    else: raise ValueError(f"Unsupported activation: {activation_name_str}")

# --- Model Definition ---
class MLPBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, norm_type: Optional[str], 
                 activation_fn_instance: nn.Module, activation_name: str, 
                 is_output_layer_block: bool, block_idx: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm: Optional[nn.Module] = None
        self.act_fn_instance = activation_fn_instance
        self.activation_name = activation_name 
        layer_prefix = f"H{block_idx + 1}"
        self.stage_names: Dict[str, Optional[str]] = {
            "linear_out": f"{layer_prefix}_Lin", "norm_out": None,
            "act_in": f"{layer_prefix}_ActIn", "act_out": f"{layer_prefix}_Act" }
        if is_output_layer_block: self.stage_names = {"linear_out": "Out_Logits", "act_out": "Out_Logits", "act_in": None}
        elif norm_type:
            if norm_type.lower() == "batchnorm": self.norm = nn.BatchNorm1d(out_dim)
            elif norm_type.lower() == "layernorm": self.norm = nn.LayerNorm(out_dim)
            if self.norm: self.stage_names["norm_out"] = f"{layer_prefix}_Nrm"
        self.act_module = self.act_fn_instance if not is_output_layer_block else nn.Identity()
    def forward(self, x: torch.Tensor, record_activations: bool = False) -> (torch.Tensor, Dict[str, torch.Tensor]):
        recorded_stages: Dict[str, torch.Tensor] = {}; current_features = self.linear(x)
        if record_activations and self.stage_names["linear_out"]: recorded_stages[self.stage_names["linear_out"]] = current_features.detach().cpu()
        pre_act_input = current_features 
        if self.stage_names.get("act_out") != self.stage_names.get("linear_out"): 
            if self.norm and self.stage_names["norm_out"]:
                current_features = self.norm(current_features)
                if record_activations: recorded_stages[self.stage_names["norm_out"]] = current_features.detach().cpu()
                pre_act_input = current_features 
            if record_activations and self.stage_names["act_in"]: recorded_stages[self.stage_names["act_in"]] = pre_act_input.detach().cpu()
            current_features = self.act_module(current_features)
            if record_activations and self.stage_names["act_out"]: recorded_stages[self.stage_names["act_out"]] = current_features.detach().cpu()
        return current_features, recorded_stages

class ConfigurableMLP(nn.Module):
    def __init__(self, layer_dims: List[int], model_activation_name: str = "ReLU", norm_type: Optional[str] = None):
        super().__init__()
        self.model_activation_name_for_frozen = model_activation_name
        act_fn_instance, act_display_name = get_activation_fn_and_name(model_activation_name)
        self.ordered_plot_stage_names: List[str] = ["Input"]
        self.blocks = nn.ModuleList()
        for i in range(len(layer_dims) - 1):
            is_final_block = (i == len(layer_dims) - 2)
            block = MLPBlock(layer_dims[i], layer_dims[i+1], norm_type, act_fn_instance, act_display_name, is_final_block, i)
            self.blocks.append(block)
            for stage_key in ["linear_out", "norm_out", "act_out"]:
                stage_name = block.stage_names.get(stage_key)
                if stage_name and stage_name not in self.ordered_plot_stage_names: self.ordered_plot_stage_names.append(stage_name)
    def forward(self, x: torch.Tensor, record_activations: bool = False) -> (torch.Tensor, Dict[str, torch.Tensor]):
        x_flattened = x.view(x.size(0), -1); all_recorded_stages: Dict[str, torch.Tensor] = {}
        if record_activations: all_recorded_stages["Input"] = x_flattened.detach().cpu()
        current_features = x_flattened
        for block in self.blocks:
            current_features, block_stages_dict = block(current_features, record_activations)
            if record_activations: all_recorded_stages.update(block_stages_dict)
        return current_features, all_recorded_stages

# --- Data Utilities (Updated for MNIST) ---
def get_mnist_dataloaders(s: ExperimentSettings, batch_size: int, num_workers: int = 2, data_root: str = './data'):
    os.makedirs(data_root, exist_ok=True)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    try: 
        trainset_full = torchvision.datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
        testset_full = torchvision.datasets.MNIST(root=data_root, train=False, download=True, transform=transform)
    except Exception as e:
        if s.DEBUG_VERBOSE: print(f"Failed to download/load MNIST: {e}. Trying without download flag.")
        trainset_full = torchvision.datasets.MNIST(root=data_root, train=True, download=False, transform=transform)
        testset_full = torchvision.datasets.MNIST(root=data_root, train=False, download=False, transform=transform)
    if s.QUICK_TEST_MODE:
        train_stride = max(1, len(trainset_full) // (500 // (batch_size // 32 or 1) or 1))
        test_stride = max(1, len(testset_full) // (100 // (batch_size // 32 or 1) or 1))
        train_indices = list(range(0, len(trainset_full), train_stride))
        val_indices = list(range(0, len(testset_full), test_stride))
        trainset, testset = Subset(trainset_full, train_indices), Subset(testset_full, val_indices)
        if s.DEBUG_VERBOSE: print(f"USING SUBSET of MNIST: {len(trainset)} train, {len(testset)} val samples.")
    else: trainset, testset = trainset_full, testset_full
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=s.DEVICE.type == 'cuda')
    testloader = DataLoader(testset, batch_size=batch_size*2, shuffle=False, num_workers=num_workers, pin_memory=s.DEVICE.type == 'cuda')
    return trainloader, testloader

def get_fixed_eval_batch(dataloader: DataLoader, batch_size: int, device: torch.device, s: ExperimentSettings) -> torch.Tensor:
    eval_batch_list = []; num_to_fetch = max(1, batch_size // (dataloader.batch_size or batch_size))
    data_iter = iter(dataloader)
    try:
        for _ in range(num_to_fetch): eval_batch_list.append(next(data_iter)[0])
    except StopIteration:
        if s.DEBUG_VERBOSE and not eval_batch_list: print("Dataloader exhausted for eval batch.")
    if not eval_batch_list: 
        try: eval_batch_list.append(next(iter(dataloader))[0])
        except StopIteration: raise ValueError("Cannot get any batch for evaluation.")
    eval_batch = torch.cat(eval_batch_list, dim=0)[:batch_size].to(device)
    if s.DEBUG_VERBOSE: print(f"Metric eval batch size: {eval_batch.shape[0]} on {device}.")
    return eval_batch

# --- Metrics Logging (Updated to pass settings to compute_* functions) ---
class MetricsLogger:
    def __init__(self, model: ConfigurableMLP, fixed_eval_batch: torch.Tensor, 
                 model_config_name: str, s: ExperimentSettings):
        self.model = model; self.fixed_eval_batch = fixed_eval_batch
        self.model_config_name = model_config_name; self.s = s
    def log_epoch_metrics(self, epoch_num: int) -> List[Dict[str, Any]]:
        self.model.eval(); raw_metrics_for_epoch: List[Dict[str, Any]] = []
        with torch.no_grad(): _, recorded_stages_dict = self.model(self.fixed_eval_batch, record_activations=True)
        if self.s.DEBUG_VERBOSE: print(f"\n--- Logging metrics for Epoch {epoch_num}, Config: {self.model_config_name} ---")
        temp_frozen_stats: Dict[str, float] = {}
        for stage_name, acts_cpu in recorded_stages_dict.items():
            if stage_name is None: continue
            if stage_name.endswith("_ActIn"):
                if self.s.DEBUG_VERBOSE: print(f" Processing for frozen: {stage_name}, shape: {acts_cpu.shape}")
                perc_frozen = compute_frozen_stats(acts_cpu, self.model.model_activation_name_for_frozen, s=self.s, stage_name_for_debug=stage_name)
                target_act_out_name = stage_name.replace("_ActIn", "_Act")
                temp_frozen_stats[target_act_out_name] = perc_frozen
                if self.s.DEBUG_VERBOSE: print(f"  Stored frozen={perc_frozen:.2f} for {target_act_out_name} (from {stage_name})")
        summary_parts = []
        for stage_name, acts_cpu in recorded_stages_dict.items():
            if stage_name is None or stage_name.endswith("_ActIn"): continue
            if self.s.DEBUG_VERBOSE: print(f" Processing stage for ER/Dup: {stage_name}, shape: {acts_cpu.shape}")
            eff_rank, perc_duplicates = np.nan, np.nan
            if acts_cpu.numel() > 0 and acts_cpu.shape[0] > 1 and acts_cpu.shape[1] > 0:
                eff_rank = compute_effective_rank_empirical(acts_cpu, stage_name)
                perc_duplicates = compute_duplicate_neurons_stats(activations_batch_cpu=acts_cpu, s=self.s, stage_name_for_debug=stage_name)
            elif self.s.DEBUG_VERBOSE: print(f"  Skipping ER/Dup for stage {stage_name} due to insufficient data.")
            current_perc_frozen = temp_frozen_stats.get(stage_name, np.nan)
            raw_metrics_for_epoch.append({'model_config': self.model_config_name, 'epoch': epoch_num, 'layer_name': stage_name,
                'eff_rank': eff_rank, 'perc_frozen': current_perc_frozen, 'perc_duplicates': perc_duplicates})
            if self.s.DEBUG_VERBOSE: print(f"  Logged for {stage_name}: ER={eff_rank:.2f}, Frozen={current_perc_frozen:.2f}, Dup={perc_duplicates:.2f}")
            if "_Act" in stage_name and not np.isnan(eff_rank): summary_parts.append(f"{stage_name.split('_')[0]}ER:{eff_rank:.1f}")
        print(f"E{epoch_num} [{self.model_config_name}] Metric Summary (ERs): {' '.join(summary_parts[:4])}...")
        self.model.train(); return raw_metrics_for_epoch

# --- Training Functions ---
def train_model_epoch(model: nn.Module, train_loader: DataLoader, optimizer: optim.Optimizer, criterion: nn.Module, device: torch.device, epoch_desc: str) -> float:
    model.train(); total_loss = 0.0
    prog_bar = tqdm(train_loader, desc=epoch_desc, leave=False)
    for inputs, labels in prog_bar:
        inputs, labels = inputs.to(device), labels.to(device); optimizer.zero_grad()
        outputs, _ = model(inputs, record_activations=False)
        loss = criterion(outputs, labels); loss.backward(); optimizer.step()
        total_loss += loss.item(); prog_bar.set_postfix({'loss': f'{loss.item():.3f}'})
    return total_loss / len(train_loader)

def run_experiment_configuration(model_cfg: ModelConfig, s: ExperimentSettings) -> pd.DataFrame:
    activation_to_run = model_cfg.activation_name if model_cfg.activation_name else s.default_activation_name
    print(f"\n===== Running Exp: {model_cfg.name} (Act: {activation_to_run}, Data: {s.dataset_name}) =====")
    layer_dims = [s.input_dim] + s.mlp_hidden_layers + [s.num_classes]
    train_loader, val_loader = get_mnist_dataloaders(s, batch_size=s.batch_size)
    model = ConfigurableMLP(layer_dims, activation_to_run, model_cfg.norm_type).to(s.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=s.learning_rate)
    criterion = nn.CrossEntropyLoss()
    fixed_eval_batch = get_fixed_eval_batch(val_loader, s.eval_batch_size, s.DEVICE, s)
    metrics_logger = MetricsLogger(model, fixed_eval_batch, model_cfg.name, s)
    all_metrics_data_raw: List[Dict[str, Any]] = []
    if fixed_eval_batch.nelement() > 0: all_metrics_data_raw.extend(metrics_logger.log_epoch_metrics(0))
    for epoch in range(s.num_epochs):
        epoch_description = f"E{epoch+1}/{s.num_epochs} [{model_cfg.name}]"
        train_model_epoch(model, train_loader, optimizer, criterion, s.DEVICE, epoch_description)
        if (epoch + 1) % s.log_metrics_every_n_epochs == 0 and fixed_eval_batch.nelement() > 0:
            all_metrics_data_raw.extend(metrics_logger.log_epoch_metrics(epoch + 1))
    df_raw = pd.DataFrame(all_metrics_data_raw)
    if s.DEBUG_VERBOSE and not df_raw.empty:
        raw_csv_path = os.path.join(s.FIGURE_DIR, f"debug_raw_metrics_{model_cfg.name.replace(' ', '_')}_{activation_to_run}_{s.dataset_name}.csv")
        df_raw.to_csv(raw_csv_path, index=False)
        print(f"Saved RAW unaggregated metrics to {raw_csv_path}")
    if not df_raw.empty: 
        final_df = df_raw.groupby(['model_config', 'epoch', 'layer_name'], as_index=False).agg(
             lambda x: x.dropna().unique()[0] if len(x.dropna().unique()) > 0 else np.nan
        ).dropna(subset=['eff_rank', 'perc_frozen', 'perc_duplicates'], how='all')
        return final_df
    else: return pd.DataFrame()

# --- Plotting Function ---
def plot_all_empirical_metrics_figure(final_metrics_df: pd.DataFrame, 
                                      current_activation_name: str, 
                                      ordered_layer_names_for_plot: List[str], 
                                      model_configurations_list: List[ModelConfig], 
                                      s: ExperimentSettings): 
    if final_metrics_df.empty: print("No metrics to plot."); return
    unique_epochs = sorted(final_metrics_df['epoch'].dropna().unique())
    if not unique_epochs: print("No epochs in data for plotting."); return
    epochs_to_plot = [unique_epochs[0]] 
    if len(unique_epochs) > 2: epochs_to_plot.append(unique_epochs[len(unique_epochs) // 2]) 
    if len(unique_epochs) > 1: epochs_to_plot.append(unique_epochs[-1]) 
    epochs_to_plot = sorted(list(set(epochs_to_plot)))
    num_epoch_subplots = len(epochs_to_plot)
    fig, axs = plt.subplots(3, num_epoch_subplots, figsize=(2.8 * num_epoch_subplots, 6.5), sharex='col') 
    if num_epoch_subplots == 1: axs = axs.reshape(3, 1) if axs.ndim == 1 else axs
    config_colors = {"MLP NoNorm": "black", "MLP BatchNorm": "dodgerblue", "MLP LayerNorm": "red"}
    layer_name_to_idx = {name: i for i, name in enumerate(ordered_layer_names_for_plot)}
    for col_idx, epoch in enumerate(epochs_to_plot):
        epoch_data = final_metrics_df[final_metrics_df['epoch'] == epoch]
        ax_er, ax_fr, ax_dp = axs[0, col_idx], axs[1, col_idx], axs[2, col_idx]
        ax_er.set_title(f"Epoch {epoch}", fontsize=plt.rcParams['axes.titlesize'])
        for model_cfg_item in model_configurations_list:
            config_name_key = model_cfg_item.name; color = config_colors.get(config_name_key, "grey")
            config_epoch_data = epoch_data[epoch_data['model_config'] == config_name_key].copy()
            if config_epoch_data.empty: continue
            config_epoch_data['plot_x_idx'] = config_epoch_data['layer_name'].map(layer_name_to_idx)
            plot_data_for_config = config_epoch_data.dropna(subset=['plot_x_idx']).sort_values(by='plot_x_idx')
            er_data = plot_data_for_config.dropna(subset=['eff_rank'])
            if not er_data.empty: ax_er.plot(er_data['plot_x_idx'], er_data['eff_rank'], color=color, linestyle='-', marker='o', markersize=2.5, label=config_name_key if col_idx==0 else None)
            frozen_data = plot_data_for_config[plot_data_for_config['layer_name'].str.contains("_Act", na=False)].dropna(subset=['perc_frozen'])
            if not frozen_data.empty: ax_fr.plot(frozen_data['plot_x_idx'], frozen_data['perc_frozen'], color=color, linestyle='--', marker='x', markersize=3, label=config_name_key if col_idx==0 else None)
            dup_data = plot_data_for_config.dropna(subset=['perc_duplicates'])
            if not dup_data.empty: ax_dp.plot(dup_data['plot_x_idx'], dup_data['perc_duplicates'], color=color, linestyle=':', marker='s', markersize=2.5, label=config_name_key if col_idx==0 else None)
        if col_idx == 0:
            ax_er.set_ylabel("Eff. Rank"); ax_fr.set_ylabel("% Frozen"); ax_dp.set_ylabel("% Duplicate")
            for ax_row in [ax_er, ax_fr, ax_dp]: ax_row.legend(loc='best')
        for ax_row in [ax_er, ax_fr, ax_dp]: ax_row.grid(True, linestyle=':', alpha=0.6)
        ax_er.set_ylim(bottom=0); ax_fr.set_ylim(0, 100); ax_dp.set_ylim(0, 100)
        ax_dp.set_xticks(list(layer_name_to_idx.values()))
        xticklabels_short = [name.replace("_Lin", "L").replace("_Nrm", "N").replace("_ActIn","Pre").replace("_Act", "A").replace("Out_Logits","Out") for name in ordered_layer_names_for_plot]
        ax_dp.set_xticklabels(xticklabels_short, rotation=45, ha='right')
        if col_idx == num_epoch_subplots // 2 : ax_dp.set_xlabel("Layer Stage")
    fig.suptitle(f"Empirical Analysis ({s.dataset_name}, Activation: {current_activation_name.upper()})", fontsize=plt.rcParams['figure.titlesize'])
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(s.FIGURE_DIR, f"empirical_all_metrics_{current_activation_name}_{s.dataset_name}.pdf")
    plt.savefig(save_path); print(f"Saved: {save_path}"); plt.show()

# --- Main Orchestration ---
def run_all_experiments(s: ExperimentSettings):
    print(f"Device: {s.DEVICE}. Activation: {s.default_activation_name}. Epochs: {s.num_epochs}. Dataset: {s.dataset_name}")
    print(f"Quick Test Mode: {s.QUICK_TEST_MODE}")
    print(f"Metric Thresholds: ReLU Frozen={s.frozen_relu_threshold}, Tanh Deriv Frozen={s.frozen_tanh_deriv_threshold}, Sigmoid Deriv Frozen={s.frozen_sigmoid_deriv_threshold}, Batch Frozen Feature Thresh={s.frozen_feature_batch_threshold}, Duplicate Corr={s.duplicate_corr_threshold}, Duplicate Min Std={s.duplicate_min_std_dev}")
    
    MODEL_CONFIGURATIONS_TO_RUN = [
        ModelConfig(name="MLP NoNorm", norm_type=None),
        ModelConfig(name="MLP BatchNorm", norm_type="batchnorm"),
        ModelConfig(name="MLP LayerNorm", norm_type="layernorm"),
    ]
    
    all_experiments_dfs: List[pd.DataFrame] = []
    temp_layer_dims = [s.input_dim] + s.mlp_hidden_layers + [s.num_classes]
    current_run_activation = s.default_activation_name 
    sample_model = ConfigurableMLP(temp_layer_dims, current_run_activation, MODEL_CONFIGURATIONS_TO_RUN[0].norm_type)
    canonical_ordered_plot_names = sample_model.ordered_plot_stage_names
    if s.DEBUG_VERBOSE: print(f"Canonical layer stages for plotting: {canonical_ordered_plot_names}")
    del sample_model

    for model_config_obj in MODEL_CONFIGURATIONS_TO_RUN:
        model_config_obj.activation_name = current_run_activation 
        df_for_config = run_experiment_configuration(model_cfg=model_config_obj, s=s)
        if not df_for_config.empty:
            all_experiments_dfs.append(df_for_config)

    if not all_experiments_dfs: print("No data collected. Exiting."); return
    
    final_aggregated_df = pd.concat(all_experiments_dfs, ignore_index=True)
    if final_aggregated_df.empty: print("Aggregated DataFrame empty. No plot/save."); return

    data_save_path = os.path.join(s.FIGURE_DIR, f"empirical_metrics_data_AGGREGATED_{current_run_activation}_{s.dataset_name}.csv")
    final_aggregated_df.to_csv(data_save_path, index=False)
    print(f"Saved final AGGREGATED data to {data_save_path}")
    
    plot_all_empirical_metrics_figure(
        final_aggregated_df, current_run_activation, 
        canonical_ordered_plot_names, MODEL_CONFIGURATIONS_TO_RUN, s
    )

if __name__ == "__main__":
    # --- CONFIGURE YOUR EXPERIMENT ---
    SETTINGS.QUICK_TEST_MODE = False  # <--- SET TO False FOR FULL RUN FOR PAPER
    SETTINGS.DEBUG_VERBOSE = False    # <--- SET TO False FOR CLEANER OUTPUT IN FULL RUN
    SETTINGS.dataset_name = "MNIST" 
    SETTINGS.default_activation_name = "ReLU" # Or "Tanh", "Sigmoid"

    # Example of changing a metric threshold for a specific run:
    # SETTINGS.frozen_feature_batch_threshold = 0.90 

    SETTINGS.__post_init__() # Re-run post_init if any core settings were changed
    
    run_all_experiments(SETTINGS)
    
    print(f"\nAll empirical figures and data processing complete for {SETTINGS.dataset_name} using {SETTINGS.default_activation_name}. Check {SETTINGS.FIGURE_DIR}")

