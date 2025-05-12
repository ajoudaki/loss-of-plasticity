import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set # NEW: Added Tuple, Set

# --- Settings & Configuration ---
@dataclass
class ExperimentSettings:
    FIGURE_DIR: str = "./figures/"
    DEVICE: torch.device = field(default_factory=lambda: torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"))
    DEBUG_VERBOSE: bool = False # Set to True for detailed print statements
    QUICK_TEST_MODE: bool = False # True for small epochs, subset of data, smaller model
    SEED: int = 42

    # Experimental parameters
    experiment_type: str = "Permuted" # "Standard" or "Permuted"
    dataset_name: str = "MNIST"
    
    # For StandardMNIST
    num_epochs_standard: int = 15 
    
    # For PermutedMNIST
    num_initial_tasks: int = 1 # Task 0: Original MNIST
    num_permutation_tasks: int = 2 # Number of *additional* permuted tasks
    epochs_per_task: int = 15 # Epochs to train on each task (original and each permuted)

    log_metrics_every_n_epochs: int = 1 # Used within each task for standard, or as a placeholder
    mlp_hidden_layers: List[int] = field(default_factory=lambda: [128,64]) # Hidden layer sizes
    default_activation_name: str = "ReLU"

    # Training hyperparameters
    batch_size: int = 256
    learning_rate: float = 1e-3
    eval_batch_size: int = 512

    # Dataset specific
    input_dim: int = 784 
    num_classes: int = 10

    # Metric calculation thresholds
    frozen_relu_threshold: float = 1e-6
    frozen_tanh_deriv_threshold: float = 1e-3
    frozen_sigmoid_deriv_threshold: float = 1e-3 
    frozen_feature_batch_threshold: float = 0.95 # Min fraction of samples for a feature to be "batch-frozen"
    
    duplicate_corr_threshold: float = 0.95
    duplicate_min_std_dev: float = 1e-5 

    # Plotting settings
    FONT_FAMILY: str = 'serif'
    FONT_SIZE_BASE: int = 8
    TITLE_FONT_SIZE_FACTOR: float = 1.1; LABEL_FONT_SIZE_FACTOR: float = 1.0
    TICK_FONT_SIZE_FACTOR: float = 0.9; LEGEND_FONT_SIZE_FACTOR: float = 0.9; FIG_DPI: int = 300


    def __post_init__(self):
        if self.QUICK_TEST_MODE:
            self.num_epochs_standard = 3
            self.num_permutation_tasks = 1 # Original + 1 permuted task
            self.epochs_per_task = 2
            self.mlp_hidden_layers = [64, 32]
            self.batch_size = 128
            self.eval_batch_size = 256
            if self.DEBUG_VERBOSE: print(f"--- QUICK TEST MODE ENABLED ({self.dataset_name}, ExpType: {self.experiment_type}) ---")

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
    activation_name: Optional[str] = None # If None, uses default_activation_name

# Initialize global settings
SETTINGS = ExperimentSettings()

# --- Matplotlib Styling & Setup ---
def setup_matplotlib_style(s: ExperimentSettings): # Unchanged from previous
    try:
        plt.style.use('seaborn-v0_8-paper')
        base_params = {'font.family': s.FONT_FAMILY, 'text.usetex': False, 'figure.dpi': s.FIG_DPI,
                       'savefig.dpi': s.FIG_DPI, 'savefig.bbox': 'tight', 'font.size': s.FONT_SIZE_BASE,
                       'axes.labelsize': s.FONT_SIZE_BASE * s.LABEL_FONT_SIZE_FACTOR,
                       'axes.titlesize': s.FONT_SIZE_BASE * s.TITLE_FONT_SIZE_FACTOR,
                       'xtick.labelsize': s.FONT_SIZE_BASE * s.TICK_FONT_SIZE_FACTOR,
                       'ytick.labelsize': s.FONT_SIZE_BASE * s.TICK_FONT_SIZE_FACTOR,
                       'legend.fontsize': s.FONT_SIZE_BASE * s.LEGEND_FONT_SIZE_FACTOR,
                       'figure.titlesize': s.FONT_SIZE_BASE * s.TITLE_FONT_SIZE_FACTOR * 1.1,}
    except OSError:
        print("seaborn-v0_8-paper style not found, using rcParams for basic styling.")
        base_params = {'font.size': s.FONT_SIZE_BASE, 'axes.titlesize': s.FONT_SIZE_BASE * s.TITLE_FONT_SIZE_FACTOR,
                       'axes.labelsize': s.FONT_SIZE_BASE * s.LABEL_FONT_SIZE_FACTOR,
                       'xtick.labelsize': s.FONT_SIZE_BASE * s.TICK_FONT_SIZE_FACTOR,
                       'ytick.labelsize': s.FONT_SIZE_BASE * s.TICK_FONT_SIZE_FACTOR,
                       'legend.fontsize': s.FONT_SIZE_BASE * s.LEGEND_FONT_SIZE_FACTOR,
                       'figure.titlesize': s.FONT_SIZE_BASE * s.TITLE_FONT_SIZE_FACTOR * 1.1,
                       'lines.linewidth': 1.2, 'lines.markersize': 3, 'figure.dpi': s.FIG_DPI,
                       'savefig.dpi': s.FIG_DPI, 'savefig.bbox': 'tight', 'font.family': s.FONT_FAMILY,
                       'text.usetex': False}
    plt.rcParams.update(base_params)

setup_matplotlib_style(SETTINGS)
os.makedirs(SETTINGS.FIGURE_DIR, exist_ok=True)
torch.manual_seed(SETTINGS.SEED); np.random.seed(SETTINGS.SEED)

# --- Metric Calculation Functions ---
def compute_effective_rank_empirical(activations_batch_cpu, stage_name_for_debug=""):
    if activations_batch_cpu.ndim == 1: return 1.0 if activations_batch_cpu.shape[0] > 1 else 0.0
    if activations_batch_cpu.shape[1] == 0: return 0.0;  # No features
    if activations_batch_cpu.shape[1] == 1: return 1.0;  # Single feature
    if activations_batch_cpu.shape[0] <= 1: return float(activations_batch_cpu.shape[1] > 0) # Not enough samples for covariance
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

# MODIFIED: To return indices of frozen features
def compute_frozen_stats(pre_activations_batch_cpu: torch.Tensor, activation_type_str: str, s: ExperimentSettings, stage_name_for_debug: str = "") -> Tuple[float, Set[int]]:
    if pre_activations_batch_cpu.numel() == 0 or pre_activations_batch_cpu.shape[1] == 0: return 0.0, set()
    if pre_activations_batch_cpu.shape[0] <= 1: 
        if s.DEBUG_VERBOSE: print(f"  [Frozen Debug {stage_name_for_debug}] Not enough samples ({pre_activations_batch_cpu.shape[0]}) for batch-frozen def.")
        return 0.0, set()
    frozen_map_per_instance = torch.zeros_like(pre_activations_batch_cpu, dtype=torch.bool)
    act_str_lower = activation_type_str.lower()
    if act_str_lower == "relu": frozen_map_per_instance = pre_activations_batch_cpu <= s.frozen_relu_threshold
    elif act_str_lower == "tanh": frozen_map_per_instance = torch.tanh(pre_activations_batch_cpu)**2 > (1.0 - s.frozen_tanh_deriv_threshold)
    elif act_str_lower == "sigmoid": sig_x = torch.sigmoid(pre_activations_batch_cpu); derivative = sig_x * (1.0 - sig_x); frozen_map_per_instance = derivative < s.frozen_sigmoid_deriv_threshold
    else: 
        if s.DEBUG_VERBOSE: print(f"  [Frozen Debug {stage_name_for_debug}] Frozen stats NA for {act_str_lower}, ret 0.")
        return 0.0, set()
    fraction_frozen_per_feature = frozen_map_per_instance.float().mean(dim=0)
    batch_frozen_features_mask = fraction_frozen_per_feature >= s.frozen_feature_batch_threshold 
    perc_batch_frozen = batch_frozen_features_mask.float().mean().item() * 100.0 if batch_frozen_features_mask.numel() > 0 else 0.0
    
    batch_frozen_indices = set(torch.where(batch_frozen_features_mask)[0].tolist()) # NEW: Get indices

    if s.DEBUG_VERBOSE: 
        print(f"  [Frozen Debug {stage_name_for_debug}] Act: {act_str_lower}, InShape: {pre_activations_batch_cpu.shape}, LowDerivInstances: {frozen_map_per_instance.sum().item()}/{frozen_map_per_instance.numel()}, BatchFrozenFeats (>{s.frozen_feature_batch_threshold*100:.0f}%): {batch_frozen_features_mask.sum().item()}/{batch_frozen_features_mask.numel()}, PercBatchFrozen: {perc_batch_frozen:.2f}%")
    return perc_batch_frozen, batch_frozen_indices

# MODIFIED: To return indices of duplicate pairs
def compute_duplicate_neurons_stats(activations_batch_cpu: torch.Tensor, s: ExperimentSettings, stage_name_for_debug: str = "") -> Tuple[float, Set[Tuple[int, int]]]:
    if activations_batch_cpu.ndim < 2 or activations_batch_cpu.shape[1] < 2: return 0.0, set()
    
    original_indices = torch.arange(activations_batch_cpu.shape[1])
    std_devs = torch.std(activations_batch_cpu, dim=0)
    valid_features_mask = std_devs > s.duplicate_min_std_dev
    
    if valid_features_mask.sum() < 2:
        if s.DEBUG_VERBOSE: print(f"  [Duplicate Debug {stage_name_for_debug}] Not enough variant features ({valid_features_mask.sum().item()}). Orig: {activations_batch_cpu.shape[1]}")
        return 0.0, set()
        
    activations_valid = activations_batch_cpu[:, valid_features_mask]
    valid_original_indices_map = original_indices[valid_features_mask] # Map local valid indices to original
    num_valid_features = activations_valid.shape[1]

    try:
        corr_matrix = torch.corrcoef(activations_valid.T) 
        if torch.isnan(corr_matrix).any(): corr_matrix = torch.nan_to_num(corr_matrix, nan=0.0)
    except RuntimeError as e:
        if s.DEBUG_VERBOSE: print(f"  [Duplicate Debug {stage_name_for_debug}] Corrcoef RuntimeError: {e}. Valid feats: {num_valid_features}")
        return 0.0, set()
        
    abs_corr_matrix = torch.abs(corr_matrix)
    # Find pairs above threshold, excluding diagonal and redundant pairs
    upper_tri_highly_corr = torch.triu(abs_corr_matrix > s.duplicate_corr_threshold, diagonal=1)
    
    duplicate_local_indices_rows, duplicate_local_indices_cols = torch.where(upper_tri_highly_corr)
    
    duplicate_original_pairs: Set[Tuple[int, int]] = set()
    for r_local, c_local in zip(duplicate_local_indices_rows.tolist(), duplicate_local_indices_cols.tolist()):
        orig_idx1 = valid_original_indices_map[r_local].item()
        orig_idx2 = valid_original_indices_map[c_local].item()
        duplicate_original_pairs.add(tuple(sorted((orig_idx1, orig_idx2))))

    # To calculate percentage based on unique neurons involved in duplication:
    involved_in_duplication_mask = torch.zeros(num_valid_features, dtype=torch.bool)
    if duplicate_local_indices_rows.numel() > 0:
      involved_in_duplication_mask[duplicate_local_indices_rows] = True
      involved_in_duplication_mask[duplicate_local_indices_cols] = True
    
    perc = involved_in_duplication_mask.float().mean().item() * 100.0 if num_valid_features > 0 else 0.0
    
    if s.DEBUG_VERBOSE: print(f"  [Duplicate Debug {stage_name_for_debug}] InShape: {activations_batch_cpu.shape}, ValidFeats: {num_valid_features}, DupPairsFound: {len(duplicate_original_pairs)}, PercInvolved: {perc:.2f}%")
    return perc, duplicate_original_pairs


def get_activation_fn_and_name(activation_name_str: str): # Unchanged
    name_lower = activation_name_str.lower()
    if name_lower == "relu": return nn.ReLU(), "ReLU"
    elif name_lower == "tanh": return nn.Tanh(), "Tanh"
    elif name_lower == "sigmoid": return nn.Sigmoid(), "Sigmoid"
    else: raise ValueError(f"Unsupported activation: {activation_name_str}")

# --- Model Definition ---
class MLPBlock(nn.Module): # From previous version
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

class ConfigurableMLP(nn.Module): # From previous version
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

# --- Data Utilities & Permutation Logic ---
def get_pixel_permutation(s: ExperimentSettings, task_id: int, num_pixels: int) -> Optional[torch.Tensor]:
    if task_id == 0: return None 
    rng = torch.Generator(); rng.manual_seed(s.SEED + task_id) 
    permutation_indices = torch.randperm(num_pixels, generator=rng)
    return permutation_indices

class PermutedDataset(Dataset):
    def __init__(self, dataset: torch.utils.data.Dataset,
                 permutation_indices: Optional[torch.Tensor] = None,
                 image_shape: Optional[tuple] = None):
        self.dataset = dataset
        self.permutation_indices = permutation_indices
        if image_shape is None:
            sample_img, _ = dataset[0]
            self.image_shape = sample_img.shape 
        else: self.image_shape = image_shape
        self.num_pixels = self.image_shape[0] * self.image_shape[1] * self.image_shape[2]
        if self.permutation_indices is not None and len(self.permutation_indices) != self.num_pixels:
            raise ValueError(f"Permutation length {len(self.permutation_indices)} does not match num_pixels {self.num_pixels}")
    def __len__(self): return len(self.dataset)
    def __getitem__(self, idx):
        img, label = self.dataset[idx] 
        if self.permutation_indices is not None:
            img_flat = img.view(-1) 
            img_permuted_flat = img_flat[self.permutation_indices]
            img = img_permuted_flat.view(self.image_shape) 
        return img, label

def get_dataset_dataloaders_for_task(s: ExperimentSettings, task_id: int, batch_size: int,
                                      num_workers: int = 2, data_root: str = './data'):
    os.makedirs(data_root, exist_ok=True)
    if s.dataset_name == "MNIST":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        try:
            trainset_full_orig = torchvision.datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
            testset_full_orig = torchvision.datasets.MNIST(root=data_root, train=False, download=True, transform=transform)
        except Exception as e:
            if s.DEBUG_VERBOSE: print(f"Failed to download/load MNIST: {e}. Trying without download flag.")
            trainset_full_orig = torchvision.datasets.MNIST(root=data_root, train=True, download=False, transform=transform)
            testset_full_orig = torchvision.datasets.MNIST(root=data_root, train=False, download=False, transform=transform)
    elif s.dataset_name == "CIFAR10":
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
        transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
        try:
            trainset_full_orig = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform_train)
            testset_full_orig = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform_test)
        except Exception as e:
            if s.DEBUG_VERBOSE: print(f"Failed to download/load CIFAR10: {e}. Trying without download flag.")
            trainset_full_orig = torchvision.datasets.CIFAR10(root=data_root, train=True, download=False, transform=transform_train)
            testset_full_orig = torchvision.datasets.CIFAR10(root=data_root, train=False, download=False, transform=transform_test)
    else: raise ValueError(f"Unsupported dataset: {s.dataset_name}")

    current_permutation = get_pixel_permutation(s, task_id, s.input_dim)
    trainset_task = PermutedDataset(trainset_full_orig, current_permutation)
    testset_task = PermutedDataset(testset_full_orig, current_permutation)

    if s.QUICK_TEST_MODE:
        train_stride = max(1, len(trainset_task) // (500 // (batch_size // 32 or 1) or 1))
        test_stride = max(1, len(testset_task) // (100 // (batch_size // 32 or 1) or 1))
        train_indices = list(range(0, len(trainset_task), train_stride))
        val_indices = list(range(0, len(testset_task), test_stride))
        trainset_final = Subset(trainset_task, train_indices)
        testset_final = Subset(testset_task, val_indices)
        if s.DEBUG_VERBOSE: print(f"  Task {task_id}: USING SUBSET of {s.dataset_name}: {len(trainset_final)} train, {len(testset_final)} val samples.")
    else: trainset_final, testset_final = trainset_task, testset_task

    trainloader = DataLoader(trainset_final, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=s.DEVICE.type in ['cuda', 'mps'])
    testloader = DataLoader(testset_final, batch_size=batch_size*2, shuffle=False, num_workers=num_workers, pin_memory=s.DEVICE.type in ['cuda', 'mps'])
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

# --- Metrics Logging & Persistence Tracking ---
# NEW: PersistentFeatureTracker class
@dataclass
class PersistentFeatureInfo:
    features: Set[Any] = field(default_factory=set) # Stores indices (int) or pairs (tuple)

class PersistentFeatureTracker:
    def __init__(self, s: ExperimentSettings):
        self.settings = s
        # Key: (model_config_name, layer_name, task_id) -> PersistentFeatureInfo
        self.frozen_history: Dict[Tuple[str, str, int], PersistentFeatureInfo] = {}
        self.duplicate_history: Dict[Tuple[str, str, int], PersistentFeatureInfo] = {}

    def record_task_features(self, model_config_name: str, task_id: int,
                             all_frozen_indices_by_layer: Dict[str, Set[int]],
                             all_duplicate_pairs_by_layer: Dict[str, Set[Tuple[int, int]]]):
        for layer_name, indices in all_frozen_indices_by_layer.items():
            self.frozen_history[(model_config_name, layer_name, task_id)] = PersistentFeatureInfo(features=indices)
        for layer_name, pairs in all_duplicate_pairs_by_layer.items():
            # Ensure pairs are sorted for consistent hashing if not already
            sorted_pairs = {tuple(sorted(p)) for p in pairs}
            self.duplicate_history[(model_config_name, layer_name, task_id)] = PersistentFeatureInfo(features=sorted_pairs)

    def calculate_persistence_data(self, total_tasks: int, model_configurations_list: List[ModelConfig], ordered_layer_names: List[str]) -> pd.DataFrame:
        persistence_records = []
        # Focus on hidden activation layers for persistence analysis
        relevant_layer_names = [ln for ln in ordered_layer_names if "_Act" in ln and not ln.startswith("Input") and not ln.startswith("Out_")]
        if not relevant_layer_names and self.settings.DEBUG_VERBOSE:
            print("[PersistenceTracker] No relevant hidden activation layers found for persistence analysis based on ordered_layer_names.")


        for model_cfg in model_configurations_list:
            model_name = model_cfg.name
            for layer_name in relevant_layer_names:
                for prev_task_id in range(total_tasks - 1): # Iterate up to second to last task
                    current_task_id = prev_task_id + 1
                    task_transition_label = f"T{prev_task_id}→T{current_task_id}"

                    # Frozen persistence
                    prev_frozen_info = self.frozen_history.get((model_name, layer_name, prev_task_id))
                    current_frozen_info = self.frozen_history.get((model_name, layer_name, current_task_id))
                    frozen_persistence_rate = np.nan
                    num_init_frozen = 0
                    num_persisted_frozen = 0

                    if prev_frozen_info and current_frozen_info and prev_frozen_info.features:
                        num_init_frozen = len(prev_frozen_info.features)
                        persisted_frozen = prev_frozen_info.features.intersection(current_frozen_info.features)
                        num_persisted_frozen = len(persisted_frozen)
                        frozen_persistence_rate = num_persisted_frozen / num_init_frozen if num_init_frozen > 0 else 0.0 # Avoid NaN if num_init is 0, give 0% or 100% if no features? Let's use 0.0. If no features, persistence is vacuously high/low.

                    # Duplicate persistence
                    prev_duplicate_info = self.duplicate_history.get((model_name, layer_name, prev_task_id))
                    current_duplicate_info = self.duplicate_history.get((model_name, layer_name, current_task_id))
                    duplicate_persistence_rate = np.nan
                    num_init_duplicate = 0
                    num_persisted_duplicate = 0

                    if prev_duplicate_info and current_duplicate_info and prev_duplicate_info.features:
                        num_init_duplicate = len(prev_duplicate_info.features)
                        persisted_duplicate = prev_duplicate_info.features.intersection(current_duplicate_info.features)
                        num_persisted_duplicate = len(persisted_duplicate)
                        duplicate_persistence_rate = num_persisted_duplicate / num_init_duplicate if num_init_duplicate > 0 else 0.0
                    
                    # Only add record if there's something to report or initial features existed
                    if num_init_frozen > 0 or num_init_duplicate > 0 or not (np.isnan(frozen_persistence_rate) and np.isnan(duplicate_persistence_rate)):
                        persistence_records.append({
                            'model_config': model_name,
                            'layer_name': layer_name,
                            'task_transition': task_transition_label,
                            'task_id_start': prev_task_id, 
                            'frozen_persistence_rate': frozen_persistence_rate * 100 if not np.isnan(frozen_persistence_rate) else np.nan,
                            'duplicate_persistence_rate': duplicate_persistence_rate * 100 if not np.isnan(duplicate_persistence_rate) else np.nan,
                            'num_initial_frozen': num_init_frozen,
                            'num_persisted_frozen': num_persisted_frozen,
                            'num_initial_duplicate': num_init_duplicate,
                            'num_persisted_duplicate': num_persisted_duplicate,
                        })
        df = pd.DataFrame(persistence_records)
        if not df.empty:
            df = df.sort_values(by=['model_config', 'layer_name', 'task_id_start'])
        return df

class MetricsLogger:
    def __init__(self, model: ConfigurableMLP, fixed_eval_batch: torch.Tensor, 
                 model_config_name: str, s: ExperimentSettings):
        self.model = model; self.fixed_eval_batch = fixed_eval_batch
        self.model_config_name = model_config_name; self.s = s

    # MODIFIED: To return collected feature indices for persistence tracking
    def log_metrics_after_task(self, task_id: int) -> Tuple[List[Dict[str, Any]], Dict[str, Set[int]], Dict[str, Set[Tuple[int, int]]]]:
        self.model.eval(); raw_metrics_for_task: List[Dict[str, Any]] = []
        
        # NEW: Dictionaries to store indices for persistence
        all_frozen_indices_by_layer: Dict[str, Set[int]] = {}
        all_duplicate_pairs_by_layer: Dict[str, Set[Tuple[int, int]]] = {}

        with torch.no_grad(): _, recorded_stages_dict = self.model(self.fixed_eval_batch, record_activations=True)
        
        if self.s.DEBUG_VERBOSE: print(f"\n--- Logging metrics after Task {task_id}, Config: {self.model_config_name} ---")
        temp_frozen_stats_val: Dict[str, float] = {}
        temp_frozen_stats_idx: Dict[str, Set[int]] = {} # Store indices from _ActIn to map to _Act

        for stage_name, acts_cpu in recorded_stages_dict.items():
            if stage_name is None: continue
            if stage_name.endswith("_ActIn"): # Pre-activations used for frozen stats
                if self.s.DEBUG_VERBOSE: print(f" Processing for frozen: {stage_name}, shape: {acts_cpu.shape}")
                perc_frozen, frozen_indices = compute_frozen_stats(acts_cpu, self.model.model_activation_name_for_frozen, s=self.s, stage_name_for_debug=stage_name)
                target_act_out_name = stage_name.replace("_ActIn", "_Act") # Metrics associated with the output of activation
                temp_frozen_stats_val[target_act_out_name] = perc_frozen
                temp_frozen_stats_idx[target_act_out_name] = frozen_indices
        
        summary_parts = []
        for stage_name, acts_cpu in recorded_stages_dict.items():
            if stage_name is None or stage_name.endswith("_ActIn"): continue # Skip pre-activations here, already processed

            if self.s.DEBUG_VERBOSE: print(f" Processing stage for ER/Dup: {stage_name}, shape: {acts_cpu.shape}")
            eff_rank, perc_duplicates, duplicate_pairs = np.nan, np.nan, set()
            
            if acts_cpu.numel() > 0 and acts_cpu.shape[0] > 1 and acts_cpu.shape[1] > 0:
                eff_rank = compute_effective_rank_empirical(acts_cpu, stage_name)
                perc_duplicates, duplicate_pairs = compute_duplicate_neurons_stats(activations_batch_cpu=acts_cpu, s=self.s, stage_name_for_debug=stage_name)
            
            current_perc_frozen = temp_frozen_stats_val.get(stage_name, np.nan)
            current_frozen_indices = temp_frozen_stats_idx.get(stage_name, set())

            # Store indices for persistence tracking for relevant layers
            if "_Act" in stage_name and not stage_name.startswith("Input") and not stage_name.startswith("Out_"): # Focus on hidden activation layers
                all_frozen_indices_by_layer[stage_name] = current_frozen_indices
                all_duplicate_pairs_by_layer[stage_name] = duplicate_pairs

            raw_metrics_for_task.append({'model_config': self.model_config_name, 
                                         'task_id': task_id,
                                         'layer_name': stage_name,
                                         'eff_rank': eff_rank, 'perc_frozen': current_perc_frozen, 
                                         'perc_duplicates': perc_duplicates})
            if "_Act" in stage_name and not np.isnan(eff_rank): summary_parts.append(f"{stage_name.split('_')[0]}ER:{eff_rank:.1f}")
        
        print(f"Task {task_id} [{self.model_config_name}] Metric Summary (ERs): {' '.join(summary_parts[:4])}...")
        self.model.train(); 
        return raw_metrics_for_task, all_frozen_indices_by_layer, all_duplicate_pairs_by_layer


# --- Training Functions ---
def train_model_on_task_epoch(model: nn.Module, train_loader: DataLoader, optimizer: optim.Optimizer, 
                              criterion: nn.Module, device: torch.device, epoch_desc: str) -> float:
    model.train(); total_loss = 0.0
    prog_bar = tqdm(train_loader, desc=epoch_desc, leave=False)
    for inputs, labels in prog_bar:
        inputs, labels = inputs.to(device), labels.to(device); optimizer.zero_grad()
        outputs, _ = model(inputs, record_activations=False)
        loss = criterion(outputs, labels); loss.backward(); optimizer.step()
        total_loss += loss.item(); prog_bar.set_postfix({'loss': f'{loss.item():.3f}'})
    return total_loss / len(train_loader)

# MODIFIED: To include persistence tracking
def run_permuted_experiment_configuration(model_cfg: ModelConfig, s: ExperimentSettings, persistence_tracker: PersistentFeatureTracker) -> pd.DataFrame: # NEW: pass tracker
    activation_to_run = model_cfg.activation_name if model_cfg.activation_name else s.default_activation_name
    print(f"\n===== Running Permuted Exp: {model_cfg.name} (Act: {activation_to_run}, Data: {s.dataset_name}) =====")

    layer_dims = [s.input_dim] + s.mlp_hidden_layers + [s.num_classes]
    model = ConfigurableMLP(layer_dims, activation_to_run, model_cfg.norm_type).to(s.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=s.learning_rate)
    criterion = nn.CrossEntropyLoss()

    all_metrics_data_raw: List[Dict[str, Any]] = []
    total_tasks = s.num_initial_tasks + s.num_permutation_tasks

    for task_id in range(total_tasks):
        print(f"\n--- Starting Task {task_id} ({'Original' if task_id == 0 else f'Permutation {task_id}'}) for {model_cfg.name} ---")
        train_loader_task, val_loader_task = get_dataset_dataloaders_for_task(s, task_id, batch_size=s.batch_size)
        fixed_eval_batch_task = get_fixed_eval_batch(val_loader_task, s.eval_batch_size, s.DEVICE, s)
        metrics_logger = MetricsLogger(model, fixed_eval_batch_task, model_cfg.name, s)

        for epoch_in_task in range(s.epochs_per_task):
            epoch_desc = f"Task {task_id} - E{epoch_in_task+1}/{s.epochs_per_task} [{model_cfg.name}]"
            train_model_on_task_epoch(model, train_loader_task, optimizer, criterion, s.DEVICE, epoch_desc)

        # Log metrics AFTER training on this task & collect indices for persistence
        # MODIFIED: to get indices from logger
        raw_task_metrics, frozen_indices_task, duplicate_pairs_task = metrics_logger.log_metrics_after_task(task_id)
        all_metrics_data_raw.extend(raw_task_metrics)
        
        # NEW: Record features for persistence
        persistence_tracker.record_task_features(model_cfg.name, task_id, frozen_indices_task, duplicate_pairs_task)

    df_raw = pd.DataFrame(all_metrics_data_raw)
    if s.DEBUG_VERBOSE and not df_raw.empty:
        raw_csv_path = os.path.join(s.FIGURE_DIR, f"debug_raw_metrics_permuted_{model_cfg.name.replace(' ', '_')}_{activation_to_run}_{s.dataset_name}.csv")
        df_raw.to_csv(raw_csv_path, index=False)
        print(f"Saved Permuted RAW unaggregated metrics to {raw_csv_path}")

    if not df_raw.empty:
        final_df = df_raw.groupby(['model_config', 'task_id', 'layer_name'], as_index=False).agg(
             lambda x: x.dropna().unique()[0] if len(x.dropna().unique()) > 0 and pd.api.types.is_scalar(x.dropna().unique()[0]) else (x.dropna().unique().tolist() if len(x.dropna().unique()) > 0 else np.nan)
        ).dropna(subset=['eff_rank', 'perc_frozen', 'perc_duplicates'], how='all')
        return final_df
    else: return pd.DataFrame()

# --- Plotting Functions ---
def plot_permuted_task_metrics_figure(final_metrics_df: pd.DataFrame, 
                                      current_activation_name: str, 
                                      ordered_layer_names_for_plot: List[str], 
                                      model_configurations_list: List[ModelConfig], 
                                      s: ExperimentSettings): 
    if final_metrics_df.empty: print("No permuted metrics to plot."); return
    unique_task_ids = sorted(final_metrics_df['task_id'].dropna().unique())
    if not unique_task_ids: print("No task_ids in data for plotting."); return
    
    tasks_to_plot = unique_task_ids 
    num_task_subplots = len(tasks_to_plot)

    fig, axs = plt.subplots(3, num_task_subplots, 
                            figsize=(2.8 * num_task_subplots, 6.5), 
                            sharex='col', sharey='row') 
    if num_task_subplots == 1: axs = axs.reshape(3, 1) if axs.ndim == 1 else axs
    elif num_task_subplots == 0: print("No tasks to plot."); return

    config_colors = {"MLP NoNorm": "black", "MLP BatchNorm": "dodgerblue", "MLP LayerNorm": "red"}
    layer_name_to_idx = {name: i for i, name in enumerate(ordered_layer_names_for_plot)}

    for col_idx, task_id_val in enumerate(tasks_to_plot):
        task_data = final_metrics_df[final_metrics_df['task_id'] == task_id_val]
        ax_er, ax_fr, ax_dp = axs[0, col_idx], axs[1, col_idx], axs[2, col_idx]
        
        task_label = f"Task {int(task_id_val)}"
        if task_id_val == 0 and s.num_initial_tasks > 0: task_label += " (Orig.)"
        elif task_id_val > 0 : task_label += f" (Perm. {int(task_id_val - s.num_initial_tasks + 1)})" # Adjust perm label

        ax_er.set_title(task_label, fontsize=plt.rcParams['axes.titlesize'])

        for model_cfg_item in model_configurations_list:
            config_name_key = model_cfg_item.name; color = config_colors.get(config_name_key, "grey")
            config_task_data = task_data[task_data['model_config'] == config_name_key].copy()
            if config_task_data.empty: continue
            config_task_data['plot_x_idx'] = config_task_data['layer_name'].map(layer_name_to_idx)
            plot_data_for_config = config_task_data.dropna(subset=['plot_x_idx']).sort_values(by='plot_x_idx')
            
            er_data = plot_data_for_config.dropna(subset=['eff_rank'])
            if not er_data.empty: ax_er.plot(er_data['plot_x_idx'], er_data['eff_rank'], color=color, linestyle='-', marker='o', markersize=2.5, label=config_name_key if col_idx==0 else None)
            frozen_data = plot_data_for_config[plot_data_for_config['layer_name'].str.contains("_Act", na=False)].dropna(subset=['perc_frozen']) # Only plot frozen for _Act layers
            if not frozen_data.empty: ax_fr.plot(frozen_data['plot_x_idx'], frozen_data['perc_frozen'], color=color, linestyle='--', marker='x', markersize=3, label=config_name_key if col_idx==0 else None)
            dup_data = plot_data_for_config.dropna(subset=['perc_duplicates'])
            if not dup_data.empty: ax_dp.plot(dup_data['plot_x_idx'], dup_data['perc_duplicates'], color=color, linestyle=':', marker='s', markersize=2.5, label=config_name_key if col_idx==0 else None)

        if col_idx == 0:
            ax_er.set_ylabel("Eff. Rank"); ax_fr.set_ylabel("% Frozen"); ax_dp.set_ylabel("% Duplicate")
            for ax_row in [ax_er, ax_fr, ax_dp]: ax_row.legend(loc='best')
        
        # Dynamic Y-axis scaling for each metric, applied per column if not sharing Y globally
        for ax_row_idx_local, metric_col_local in enumerate(['eff_rank', 'perc_frozen', 'perc_duplicates']):
            current_ax_local = axs[ax_row_idx_local, col_idx]
            relevant_vals_local = task_data[metric_col_local].dropna() # Values for this task only
            if not relevant_vals_local.empty:
                min_val_l, max_val_l = relevant_vals_local.min(), relevant_vals_local.max()
                padding_l = (max_val_l - min_val_l) * 0.05 if (max_val_l - min_val_l) > 0 else 1
                if metric_col_local == 'eff_rank': current_ax_local.set_ylim(bottom=0, top=max_val_l + padding_l if max_val_l > 0 else 10)
                else: current_ax_local.set_ylim(0, 100 + padding_l if metric_col_local != 'eff_rank' and max_val_l > 90 else max(10, max_val_l + padding_l))
            elif metric_col_local == 'eff_rank': current_ax_local.set_ylim(bottom=0, top=10) # Default if no data
            else: current_ax_local.set_ylim(0,100)


        for ax_row in [ax_er, ax_fr, ax_dp]: ax_row.grid(True, linestyle=':', alpha=0.6)
        
        ax_dp.set_xticks(list(layer_name_to_idx.values()))
        xticklabels_short = [name.replace("_Lin", "L").replace("_Nrm", "N").replace("_ActIn","Pre").replace("_Act", "A").replace("Out_Logits","Out") for name in ordered_layer_names_for_plot]
        ax_dp.set_xticklabels(xticklabels_short, rotation=45, ha='right')
        if col_idx == num_task_subplots // 2 : ax_dp.set_xlabel("Layer Stage")
    
    fig.suptitle(f"Permuted MNIST Analysis ({s.dataset_name}, Activation: {current_activation_name.upper()})", fontsize=plt.rcParams['figure.titlesize'])
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(s.FIGURE_DIR, f"permuted_metrics_summary_{current_activation_name}_{s.dataset_name}.pdf")
    plt.savefig(save_path); print(f"Saved: {save_path}"); plt.show()

# NEW: Plotting function for feature persistence
def plot_feature_persistence_figure(persistence_df: pd.DataFrame, s: ExperimentSettings, current_activation_name: str):
    if persistence_df.empty:
        print("Persistence DataFrame is empty. No persistence plot will be generated.")
        return

    # Aggregate by averaging over relevant hidden layers for each model_config and task_transition
    # Relevant layers are already implicitly handled if calculate_persistence_data only includes them.
    # If not, filter here: e.g. persistence_df = persistence_df[persistence_df['layer_name'].str.contains("_Act")]
    
    # We want one line per model_config, so group by model_config and task_transition, then average rates.
    # Ensure task_transition is categorical for correct plot ordering.
    df_agg = persistence_df.groupby(['model_config', 'task_transition', 'task_id_start'], as_index=False).agg(
        avg_frozen_persistence_rate=('frozen_persistence_rate', 'mean'),
        avg_duplicate_persistence_rate=('duplicate_persistence_rate', 'mean'),
        # For error bars or confidence, one might also calculate 'std' or count
    ).sort_values(by=['model_config', 'task_id_start']) # Sort by task_id_start to ensure transitions are ordered

    if df_agg.empty:
        print("Aggregated persistence DataFrame is empty. No plot.")
        return

    unique_transitions = df_agg['task_transition'].unique() # Already sorted due to task_id_start
    num_transitions = len(unique_transitions)

    fig, axs = plt.subplots(2, 1, figsize=(max(6, 2.5 * num_transitions), 7), sharex=True) # Adjusted figsize
    
    ax_frozen = axs[0]
    ax_duplicate = axs[1]
    
    config_colors = {"MLP NoNorm": "black", "MLP BatchNorm": "dodgerblue", "MLP LayerNorm": "red"}
    model_configs = df_agg['model_config'].unique()

    for config_name in model_configs:
        config_data = df_agg[df_agg['model_config'] == config_name]
        color = config_colors.get(config_name, "grey")
        
        # Frozen Persistence Plot
        ax_frozen.plot(config_data['task_transition'], config_data['avg_frozen_persistence_rate'], 
                       marker='o', linestyle='-', color=color, label=config_name, markersize=5)
        
        # Duplicate Persistence Plot
        ax_duplicate.plot(config_data['task_transition'], config_data['avg_duplicate_persistence_rate'], 
                          marker='s', linestyle='--', color=color, label=config_name, markersize=5)

    ax_frozen.set_title('Persistence of Frozen Features Across Tasks', fontsize=plt.rcParams['axes.titlesize'] * 0.9)
    ax_frozen.set_ylabel('Avg. Persistence Rate (%)')
    ax_frozen.grid(True, linestyle=':', alpha=0.7)
    ax_frozen.legend(loc='best')
    ax_frozen.set_ylim(0, 105)

    ax_duplicate.set_title('Persistence of Duplicate Features Across Tasks', fontsize=plt.rcParams['axes.titlesize'] * 0.9)
    ax_duplicate.set_ylabel('Avg. Persistence Rate (%)')
    ax_duplicate.set_xlabel('Task Transition')
    ax_duplicate.grid(True, linestyle=':', alpha=0.7)
    ax_duplicate.legend(loc='best')
    ax_duplicate.set_ylim(0, 105)

    if num_transitions > 3: # Rotate labels if many transitions
         plt.setp(ax_duplicate.get_xticklabels(), rotation=30, ha="right")


    fig.suptitle(f"Feature Persistence ({s.dataset_name}, Act: {current_activation_name.upper()}, Avg over Hidden Layers)", 
                 fontsize=plt.rcParams['figure.titlesize'])
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    save_path = os.path.join(s.FIGURE_DIR, f"feature_persistence_across_tasks_{current_activation_name}_{s.dataset_name}.pdf")
    plt.savefig(save_path)
    print(f"Saved persistence plot: {save_path}")
    plt.show()


# --- Main Orchestration ---
def run_all_experiments(s: ExperimentSettings):
    print(f"Device: {s.DEVICE}. Activation: {s.default_activation_name}. Dataset: {s.dataset_name}")
    print(f"Experiment Type: {s.experiment_type}")
    print(f"Quick Test Mode: {s.QUICK_TEST_MODE}")
    print(f"Metric Thresholds: ReLU Frozen={s.frozen_relu_threshold}, Tanh Deriv Frozen={s.frozen_tanh_deriv_threshold}, Sigmoid Deriv Frozen={s.frozen_sigmoid_deriv_threshold}, Batch Frozen Feat Thresh={s.frozen_feature_batch_threshold}, Dup Corr={s.duplicate_corr_threshold}, Dup Min Std={s.duplicate_min_std_dev}")
    
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

    experiment_runner_fn = None
    if s.experiment_type == "Standard":
        print("StandardMNIST experiment type selected. Ensure run_experiment_configuration is adapted if needed.")
        return 
    elif s.experiment_type == "Permuted":
        experiment_runner_fn = run_permuted_experiment_configuration
    else:
        raise ValueError(f"Unknown experiment_type: {s.experiment_type}")

    # NEW: Initialize persistence tracker here, once for all configs if sharing, or per config
    # For this setup, it's easier to pass it into the runner function which uses it per config.
    # The tracker itself stores by model_config_name.
    global_persistence_tracker = PersistentFeatureTracker(s)


    for model_config_obj in MODEL_CONFIGURATIONS_TO_RUN:
        model_config_obj.activation_name = current_run_activation 
        # MODIFIED: Pass persistence_tracker to the runner
        df_for_config = experiment_runner_fn(model_cfg=model_config_obj, s=s, persistence_tracker=global_persistence_tracker) # type: ignore
        if not df_for_config.empty:
            all_experiments_dfs.append(df_for_config)

    if not all_experiments_dfs: print("No data collected for standard metrics. Exiting."); return
    
    final_aggregated_df = pd.concat(all_experiments_dfs, ignore_index=True)
    if final_aggregated_df.empty: print("Aggregated DataFrame empty. No standard plot/save."); return

    data_save_path = os.path.join(s.FIGURE_DIR, f"{s.experiment_type.lower()}_metrics_AGGREGATED_{current_run_activation}_{s.dataset_name}.csv")
    final_aggregated_df.to_csv(data_save_path, index=False)
    print(f"Saved final AGGREGATED data for standard metrics to {data_save_path}")
    
    if s.experiment_type == "Permuted":
        plot_permuted_task_metrics_figure(
            final_aggregated_df, current_run_activation, 
            canonical_ordered_plot_names, MODEL_CONFIGURATIONS_TO_RUN, s
        )
        
        # NEW: Calculate and plot persistence data
        total_tasks_run = s.num_initial_tasks + s.num_permutation_tasks
        persistence_df = global_persistence_tracker.calculate_persistence_data(total_tasks_run, MODEL_CONFIGURATIONS_TO_RUN, canonical_ordered_plot_names)
        if not persistence_df.empty:
            persistence_data_save_path = os.path.join(s.FIGURE_DIR, f"feature_persistence_AGGREGATED_{current_run_activation}_{s.dataset_name}.csv")
            persistence_df.to_csv(persistence_data_save_path, index=False)
            print(f"Saved feature PERSISTENCE data to {persistence_data_save_path}")
            plot_feature_persistence_figure(persistence_df, s, current_run_activation)
        else:
            print("No persistence data generated to plot/save.")


if __name__ == "__main__":
    # --- CONFIGURE YOUR EXPERIMENT ---
    SETTINGS.QUICK_TEST_MODE = False  # <--- SET TO False FOR FULL RUN FOR PAPER
    SETTINGS.DEBUG_VERBOSE = False    # <--- SET TO False FOR CLEANER OUTPUT IN FULL RUN
    
    SETTINGS.experiment_type = "Permuted" # "Standard" or "Permuted"
    SETTINGS.dataset_name = "MNIST" 
    SETTINGS.default_activation_name = "ReLU" 

    # For PermutedMNIST specific settings (if not overridden by QUICK_TEST_MODE)
    if not SETTINGS.QUICK_TEST_MODE and SETTINGS.experiment_type == "Permuted":
        SETTINGS.num_initial_tasks = 1 # Original MNIST
        SETTINGS.mlp_hidden_layers = [512, 512, 512] # Hidden layer sizes
        # MODIFIED: Increased num_permutation_tasks for more transitions to see persistence
        SETTINGS.num_permutation_tasks = 2  # e.g., 3 additional permuted tasks (total 4 tasks, 3 transitions) 
        SETTINGS.epochs_per_task = 15      # e.g., 5 epochs on each task
    elif not SETTINGS.QUICK_TEST_MODE and SETTINGS.experiment_type == "Standard":
        SETTINGS.num_epochs_standard = 5 # As before

    SETTINGS.__post_init__() # Re-run post_init to apply QUICK_TEST_MODE or other changes
    
    run_all_experiments(SETTINGS)
    
    print(f"\nAll empirical figures and data processing complete for {SETTINGS.experiment_type} on {SETTINGS.dataset_name} using {SETTINGS.default_activation_name}. Check {SETTINGS.FIGURE_DIR}")