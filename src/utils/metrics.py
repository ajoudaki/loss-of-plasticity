import torch
import numpy as np
from scipy import stats
import re
from omegaconf import DictConfig
from typing import Optional, List
from collections import defaultdict


def _subsample_rows(X: torch.Tensor, max_rows: int = 8192, seed: int = None):
    """Uniformly subsample rows to cap O(N^2) ops; keeps device and dtype."""
    N = X.shape[0]
    if N <= max_rows:
        return X
    g = torch.Generator(device=X.device)
    if seed is not None:
        g.manual_seed(int(seed))
    idx = torch.randperm(N, generator=g, device=X.device)[:max_rows]
    return X.index_select(0, idx)


def _standardize_columns(X: torch.Tensor, means: torch.Tensor, stds: torch.Tensor, eps: float = 1e-12):
    return (X - means) / (stds + eps)


def _flatten_activations(layer_act: torch.Tensor):
    """Reshape layer activations to 2D matrix (samples × features)."""
    shape = layer_act.shape
    if len(shape) == 4:  # Convolutional layer
        return layer_act.permute(0, 2, 3, 1).contiguous().view(-1, shape[1])
    elif len(shape) == 3:  # Transformer layer
        return layer_act.contiguous().view(-1, shape[2])
    else:  # Linear layer
        return layer_act.view(-1, shape[1])


def compute_activation_statistics(flattened_act: torch.Tensor):
    """Compute mean and standard deviation of activations for each unit."""
    means = flattened_act.mean(dim=0)
    stds = flattened_act.std(dim=0)
    normalized_means = means / (stds + 1e-12)
    return means, stds, normalized_means


@torch.no_grad()
def measure_dead_neurons(flattened_act: torch.Tensor, dead_threshold: float = 0.95):
    """Measure fraction of neurons that are inactive (dead)."""
    is_zero = (flattened_act.abs() < 1e-7)
    frac_zero_per_neuron = is_zero.float().mean(dim=0)
    dead_mask = (frac_zero_per_neuron > dead_threshold)
    dead_fraction = dead_mask.float().mean().item()
    return {"dead_fraction": dead_fraction}, {}


@torch.no_grad()
def measure_duplicate_neurons(flattened_act: torch.Tensor, corr_threshold: float) -> float:
    """Measure fraction of neurons that are duplicates of others."""
    flattened_act = flattened_act.t()  
    flattened_act = torch.nn.functional.normalize(flattened_act, p=2, dim=1)
    similarity_matrix = torch.matmul(flattened_act, flattened_act.t())
    upper_tri_mask = torch.triu(torch.ones_like(similarity_matrix), diagonal=1).bool()
    dup_pairs = (similarity_matrix > corr_threshold) & upper_tri_mask
    neuron_is_dup = dup_pairs.any(dim=1)
    fraction_dup = neuron_is_dup.float().mean().item()
    return {"dup_fraction": fraction_dup}, {}


@torch.no_grad()
def measure_effective_rank(flattened_act: torch.Tensor, svd_sample_size: int = 1024, seed: Optional[int] = None):
    """
    Compute effective rank (entropy of normalized singular values).
    
    Args:
        layer_act: Layer activations
        svd_sample_size: Maximum number of samples to use for SVD
        seed: Optional random seed for sampling
    """
    N = flattened_act.shape[0]
    if N > svd_sample_size:
        # Use seed if provided, otherwise use the current random state
        if seed is not None:
            generator = torch.Generator()
            generator.manual_seed(seed)
            idx = torch.randperm(N, generator=generator)[:svd_sample_size]
        else:
            idx = torch.randperm(N)[:svd_sample_size]
        flattened_act = flattened_act[idx]
    S = torch.linalg.svdvals(flattened_act)
    S_sum = S.sum()
    if S_sum < 1e-9:
        return {"eff_rank": 0.0}, {}
    p = S / S_sum
    p_log_p = p * torch.log(p + 1e-12)
    eff_rank = torch.exp(-p_log_p.sum()).item()
    return {"eff_rank": eff_rank}, {}


@torch.no_grad()
def measure_stable_rank(flattened_act, means=None, sample_size=8192, use_gram=True, seed=None):
    """
    Compute stable rank (squared Frobenius norm / spectral norm squared).
    
    Args:
        layer_act: Layer activations
        means: Optional means for normalization
        sample_size: Maximum number of samples to use
        use_gram: Whether to use the Gram matrix approach
        seed: Optional random seed for sampling
    """
    N, D = flattened_act.shape
    if N > sample_size:
        # Use seed if provided, otherwise use the current random state
        if seed is not None:
            generator = torch.Generator()
            generator.manual_seed(seed)
            idx = torch.randperm(N, generator=generator)[:sample_size]
        else:
            idx = torch.randperm(N)[:sample_size]
        flattened_act = flattened_act[idx]
        N = sample_size
        if means is not None:
            means = means[idx]

    if means is None:
        means = flattened_act.mean(dim=0, keepdim=True)
    flattened_act = flattened_act - means

    if use_gram or D < N:
        frob_norm_sq = torch.sum(flattened_act**2).item()
        gram = torch.matmul(flattened_act.t(), flattened_act)
        trace_gram_squared = torch.sum(gram**2).item()
        if trace_gram_squared < 1e-9:
            return {"stable_rank": 0.0}, {}
        stable_rank = (frob_norm_sq**2) / trace_gram_squared
    else:
        cov = torch.matmul(flattened_act, flattened_act.t())
        trace_cov = torch.trace(cov).item()
        trace_cov_squared = torch.sum(cov**2).item()
        if trace_cov_squared < 1e-9:
            return {"stable_rank": 0.0}, {}
        stable_rank = (trace_cov**2) / trace_cov_squared
    return {"stable_rank": stable_rank}, {}


@torch.no_grad()
def measure_saturated_neurons(flattened_act, layer_grad, saturation_threshold=1e-4, saturation_percentage=0.99):
    """
    Measures the fraction of saturated neurons in a layer.
    
    Saturated neurons are identified as those where the ratio of gradient magnitude
    to mean activation magnitude is very small, indicating the neuron is in a flat
    region of the loss landscape.
    """
    flattened_grad = _flatten_activations(layer_grad)
    
    # Calculate the mean activation magnitude for each neuron
    mean_act_magnitude = flattened_act.abs().mean(dim=0, keepdim=True)
    
    # Avoid division by zero
    mean_act_magnitude = torch.clamp(mean_act_magnitude, min=1e-12)
    
    # Calculate the ratio of gradient magnitude to mean activation magnitude
    saturation_ratio = flattened_grad.abs() / mean_act_magnitude
    
    # Mark neurons as saturated if the ratio is below the threshold
    is_saturated = (saturation_ratio < saturation_threshold).float()
    
    # Calculate fraction of samples where each neuron appears saturated
    saturation_per_neuron = is_saturated.mean(dim=0)
    
    # Consider a neuron truly saturated if it's saturated in most samples
    saturated_mask = (saturation_per_neuron > saturation_percentage)
    
    # Calculate the overall fraction of saturated neurons
    saturated_fraction = saturated_mask.float().mean().item()
    
    return {"saturated_fraction": saturated_fraction}, {"neuron_saturation": saturation_per_neuron.detach().cpu().numpy()}


# --------------------------------------------------------
# Gaussianity measures
# --------------------------------------------------------


@torch.no_grad()
def measure_non_gaussianity(flattened_act, sample_size=1024, seed=None, method="shapiro"):
    """
    Measure the distance to Gaussianity for each neuron's activations.
    
    This function quantifies how much the distribution of activations for each neuron
    deviates from a Gaussian (normal) distribution. In many neural network theories,
    activations that follow Gaussian distributions are considered optimal for information
    transfer and learning. Significant deviations may indicate issues with network training
    or specialized feature extraction.
    
    The function supports multiple statistical tests to measure non-Gaussianity:
    
    Args:
        layer_act: Layer activations tensor of shape [batch_size, n_units]
        sample_size: Maximum number of samples to use for the test (for efficiency)
        seed: Optional random seed for sampling
        method: Method to use for Gaussianity testing:
                - "shapiro": Shapiro-Wilk test (more accurate for smaller samples)
                  Returns 1-W where W is in [0,1], higher values mean less Gaussian
                - "ks": Kolmogorov-Smirnov test against normal distribution
                  Returns D statistic, higher values mean less Gaussian
                - "anderson": Anderson-Darling test (more sensitive to tails)
                  Returns A² statistic normalized by critical value, higher values mean less Gaussian
                - "kurtosis": Use excess kurtosis as a measure of non-Gaussianity
                  Returns absolute value of excess kurtosis, 0 = perfectly Gaussian
    
    Returns:
        A measure of non-Gaussianity (averaged across all neurons in the layer).
        Higher values indicate greater deviation from Gaussian distribution.
        The range depends on the method used:
        - shapiro: [0, 1] where 0 = perfectly Gaussian
        - ks: [0, 1] where 0 = perfectly Gaussian
        - anderson: [0, 10] (capped) where 0 = perfectly Gaussian
        - kurtosis: [0, 10] (capped) where 0 = perfectly Gaussian
    """
    N, D = flattened_act.shape
    
    # If we have more than sample_size samples, subsample to save computation
    if N > sample_size:
        # Use seed if provided, otherwise use the current random state
        if seed is not None:
            generator = torch.Generator()
            generator.manual_seed(seed)
            idx = torch.randperm(N, generator=generator)[:sample_size]
        else:
            idx = torch.randperm(N)[:sample_size]
        flattened_act = flattened_act[idx]
        N = sample_size
    
    # Convert to numpy for statistical tests
    act_np = flattened_act.detach().cpu().numpy()
    
    if method == "shapiro":
        # Shapiro-Wilk test - returns W statistic and p-value
        # Lower W values indicate deviation from normality
        # We convert to a non-Gaussianity score (1-W) so higher means less Gaussian
        non_gaussianity = []
        for j in range(D):
            # Calculate standard deviation
            std_val = np.std(act_np[:, j])
            
            # If std is very small (effectively constant values), skip the statistical test
            if std_val < 1e-6:
                # For constant values, consider them maximally non-Gaussian
                non_gaussianity.append(1.0)
                continue
                
            # Normalize to zero mean and unit variance
            x = (act_np[:, j] - np.mean(act_np[:, j])) / (std_val + 1e-8)
            try:
                # Maximum sample size is 5000 for Shapiro-Wilk
                if len(x) > 5000:
                    x = x[:5000]
                w, _ = stats.shapiro(x)
                # Convert W to a non-Gaussianity score (1-W ranges from 0 to 1)
                non_gaussianity.append(1.0 - w)
            except Exception:
                # Return a high value if test fails (maximum non-Gaussianity)
                non_gaussianity.append(1.0)
    
    elif method == "ks":
        # Kolmogorov-Smirnov test against a normal distribution
        non_gaussianity = []
        for j in range(D):
            # Calculate standard deviation
            std_val = np.std(act_np[:, j])
            
            # If std is very small (effectively constant values), skip the statistical test
            if std_val < 1e-6:
                # For constant values, consider them maximally non-Gaussian
                non_gaussianity.append(1.0)
                continue
                
            # Normalize to zero mean and unit variance
            x = (act_np[:, j] - np.mean(act_np[:, j])) / (std_val + 1e-8)
            try:
                # Test against normal distribution - returns KS statistic and p-value
                # Higher KS indicates greater deviation from normality
                ks, _ = stats.kstest(x, 'norm')
                non_gaussianity.append(ks)
            except Exception:
                non_gaussianity.append(1.0)
    
    elif method == "anderson":
        # Anderson-Darling test - more sensitive to tails
        non_gaussianity = []
        for j in range(D):
            # Calculate standard deviation
            std_val = np.std(act_np[:, j])
            
            # If std is very small (effectively constant values), skip the statistical test
            if std_val < 1e-6:
                # For constant values, consider them maximally non-Gaussian
                non_gaussianity.append(10.0)  # Use max value consistent with this method
                continue
                
            # Normalize to zero mean and unit variance
            x = (act_np[:, j] - np.mean(act_np[:, j])) / (std_val + 1e-8)
            try:
                result = stats.anderson(x, 'norm')
                # Higher statistic means greater deviation from normality
                # Normalize by critical value for significance level 5%
                stat = result.statistic / result.critical_values[2]
                non_gaussianity.append(min(stat, 10.0))  # Cap at 10 to avoid extreme values
            except Exception:
                non_gaussianity.append(10.0)
    
    elif method == "kurtosis":
        # Use excess kurtosis as a measure of non-Gaussianity
        # Gaussian distribution has excess kurtosis of 0
        # We take absolute value so both super- and sub-Gaussian show as deviation
        non_gaussianity = []
        for j in range(D):
            # Calculate standard deviation
            std_val = np.std(act_np[:, j])
            
            # If std is very small (effectively constant values), skip the statistical test
            if std_val < 1e-6:
                # For constant values, consider them maximally non-Gaussian
                non_gaussianity.append(10.0)  # Use max value consistent with this method
                continue
                
            # Normalize to zero mean and unit variance
            x = (act_np[:, j] - np.mean(act_np[:, j])) / (std_val + 1e-8)
            try:
                kurtosis = stats.kurtosis(x)
                non_gaussianity.append(min(abs(kurtosis), 10.0))  # Cap at 10
            except Exception:
                non_gaussianity.append(10.0)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    mean_non_gaussianity = float(np.mean(non_gaussianity))
    
    return {"mean_non_gaussianity": mean_non_gaussianity}, {"non_gaussianity": non_gaussianity}


@torch.no_grad()
def measure_univariate_diagnostics(
    flattened_act: torch.Tensor,
    means: torch.Tensor = None,
    stds: torch.Tensor = None,
    sample_rows: int = 8192,
    seed: int = None,
):
    """
    Per-neuron (feature) diagnostics across samples in this layer:
      - skewness, excess kurtosis
      - IQR-to-sigma ratio (Gaussian ≈ 1)
    Returns aggregates and per-neuron arrays (for optional hist logging).
    """
    X = flattened_act
    if means is None or stds is None:
        means = X.mean(dim=0)
        stds  = X.std(dim=0)

    Xs = _subsample_rows(X, sample_rows, seed=seed)
    Z  = _standardize_columns(Xs, means, stds)

    skewness =  Z.pow(3).mean(dim=0)
    excess_kurtosis =  Z.pow(4).mean(dim=0) - 3.0

    q75 = torch.quantile(Z, 0.75, dim=0)
    q25 = torch.quantile(Z, 0.25, dim=0)
    IQR = q75 - q25
    iqr_over_sigma = IQR / 1.349  # IQR / (1.349 * sigma) ~ 1 for Gaussian

    metrics = {
        "uv_med_abs_skewness":           skewness.abs().median().item(),
        # "uv_p10_abs_skewness":           skewness.abs().quantile(0.10).item(),
        # "uv_p90_abs_skewness":           skewness.abs().quantile(0.90).item(),
        "uv_med_excess_kurtosis":        excess_kurtosis.median().item(),
        # "uv_p10_excess_kurtosis":        excess_kurtosis.quantile(0.10).item(),
        # "uv_p90_excess_kurtosis":        excess_kurtosis.quantile(0.90).item(),
        "uv_med_iqr_over_sigma":         iqr_over_sigma.median().item(),
        # "uv_frac_abs_skewness_gt_0p5":   (skewness.abs() > 0.5).float().mean().item(),
        # "uv_frac_excess_kurtosis_gt_1":  (excess_kurtosis.abs() > 1.0).float().mean().item(),
    }
    hists = {
        "skewness": skewness.detach().cpu().numpy(),
        "excess_kurtosis": excess_kurtosis.detach().cpu().numpy(),
        "iqr_over_sigma": iqr_over_sigma.detach().cpu().numpy(),
    }
    return metrics, hists


# --------------------------------------------------------
# Covariance and correlation matrix metrics
# --------------------------------------------------------


def compute_cov_matrix(flattened_act: torch.Tensor, means=None):
    """Compute covariance matrix of neuron activations."""
    if means is None:
        means = flattened_act.mean(dim=0, keepdim=True)
    flattened_act = flattened_act - means
    B, _ = flattened_act.shape
    cov_matrix = torch.matmul(flattened_act.t(), flattened_act) / B
    return (cov_matrix + cov_matrix.t()) / 2.0  # Ensure symmetry


def compute_eigenvalues(hermitian_matrix: torch.Tensor):
    """Compute eigenvalues of a hermitian matrix."""
    eigenvalues = torch.linalg.eigvalsh(hermitian_matrix)
    return eigenvalues


def compute_corr_matrix(cov_matrix: torch.Tensor, eps: float = 1e-12):
    """Compute correlation matrix from covariance matrix."""
    diag = torch.sqrt(torch.diag(cov_matrix) + eps)
    corr_matrix = cov_matrix / diag[:, None] / diag[None, :]
    return corr_matrix


@torch.no_grad()
def measure_cov_corr_metrics(flattened_act: torch.Tensor, means: torch.Tensor):
    """Compute metrics based on covariance and correlation matrices."""

    cov_matrix = compute_cov_matrix(flattened_act, means)
    corr_matrix = compute_corr_matrix(cov_matrix)

    metrics, hists = {}, {}
    hists["covariance_eigenvals"] = compute_eigenvalues(cov_matrix).detach().cpu().numpy()
    hists["correlation_eigenvals"] = compute_eigenvalues(corr_matrix).detach().cpu().numpy()

    D = corr_matrix.shape[0]
    off_diag_mask = ~(torch.eye(D, device=corr_matrix.device).bool())
    hists["off_diagonal_corr_coeffs"] = corr_matrix[off_diag_mask].detach().cpu().numpy()
    metrics["mean_abs_off_diag_correlation"] = np.abs(hists["off_diagonal_corr_coeffs"]).mean().item()

    return metrics, hists


def analyze_fixed_batch(model, monitor, fixed_batch, fixed_targets, criterion, 
                      dead_threshold, 
                      corr_threshold, 
                      saturation_threshold, 
                      saturation_percentage,
                      gaussianity_method="shapiro",
                      use_wandb=False,
                      log_histograms=False,
                      prefix="",
                      metrics_log=None,
                      device='cpu',
                      selected_metrics: Optional[List[str]] = None,
                      seed=None):
    """
    Analyze model behavior on a fixed batch to compute comprehensive metrics.
    
    This function performs a forward and backward pass with the provided batch,
    then computes a variety of metrics to analyze the model's internal behavior.
    The metrics include measures of dead neurons, duplicate neurons, effective rank,
    stable rank, neuron saturation, and non-Gaussianity of activations.
    
    Additionally, it computes statistics of neuron activations (means and standard deviations)
    and can format these for visualization with Weights & Biases.
    
    Args:
        model: Neural network model to analyze
        monitor: NetworkMonitor instance for collecting activations and gradients
        fixed_batch: Input data batch for analysis
        fixed_targets: Target labels for the batch
        criterion: Loss function to compute gradients
        dead_threshold: Threshold for dead neuron detection (fraction of zero activations)
        corr_threshold: Threshold for duplicate neuron detection (correlation cutoff)
        saturation_threshold: Threshold for saturated neuron detection (gradient magnitude ratio)
        saturation_percentage: Percentage of samples required for a neuron to be considered saturated
        gaussianity_method: Method to use for Gaussianity measurement ("shapiro", "ks", "anderson", "kurtosis")
        use_wandb: Whether Weights & Biases is being used for logging
        log_histograms: Whether to prepare histograms of activation statistics for logging
        prefix: Prefix for metrics (e.g., "train/" or "val/") for organizing in dashboards
        metrics_log: Dictionary to add metrics to (if None, a new one is created)
        device: Device to run computations on ('cpu', 'cuda', 'mps')
        selected_metrics: If None → compute all allowed metrics. Else compute only
            the metrics listed here.
        seed: Optional random seed for sampling operations (for reproducibility)
        
    Returns:
        Tuple containing:
        - Dictionary of metrics for each layer (metric_name -> value)
        - Dictionary of activation statistics for each layer (means, stds)
        - Dictionary of metrics formatted for wandb logging (if use_wandb is True)
    """
    if fixed_batch.device != device:
        fixed_batch = fixed_batch.to(device)
        fixed_targets = fixed_targets.to(device)
    
    hooks_were_active = monitor.hooks_active
    monitor.register_hooks()
    
    with torch.set_grad_enabled(criterion is not None):
        outputs = model(fixed_batch)
        loss = criterion(outputs, fixed_targets)
        loss.backward()
    
    metrics = defaultdict(dict)
    hists = defaultdict(dict)  # Change hists to defaultdict(dict)
    activation_stats = {}
    latest_acts = monitor.get_latest_activations()
    latest_grads = monitor.get_latest_gradients()
    
    if metrics_log is None:
        metrics_log = {}

    for layer_name, act in latest_acts.items():
        if layer_name not in latest_grads:
            continue
            
        grad = latest_grads[layer_name]

        flattened_act = _flatten_activations(act)
        flattened_grad = _flatten_activations(grad)
        
        means, stds, normalized_means = compute_activation_statistics(flattened_act)
        activation_stats[layer_name] = {
            'means': means.detach().cpu(),
            'stds': stds.detach().cpu(),
            'normalized_means': normalized_means.detach().cpu()
        }

        # Build the per-layer metrics mapping (lazily evaluated)
        all_metric_fns = {
            "dead_fraction": lambda: measure_dead_neurons(flattened_act, dead_threshold),
            "dup_fraction": lambda: measure_duplicate_neurons(flattened_act, corr_threshold),
            "eff_rank": lambda: measure_effective_rank(flattened_act, seed=seed),
            "stable_rank": lambda: measure_stable_rank(flattened_act, seed=seed),
            "saturated_frac": lambda: measure_saturated_neurons(flattened_act, flattened_grad, saturation_threshold, saturation_percentage),
            "non_gaussianity": lambda: measure_non_gaussianity(flattened_act, seed=seed, method=gaussianity_method),
            "cov_corr_metrics": lambda: measure_cov_corr_metrics(flattened_act, means),
            "univariate_diagnostics": lambda: measure_univariate_diagnostics(flattened_act, means, stds, seed=seed),
        }

        # Decide which metrics to run
        wanted = all_metric_fns.keys() if selected_metrics is None else selected_metrics
        # Validate keys
        unknown = sorted(set(wanted) - set(all_metric_fns.keys()))
        if unknown:
            raise ValueError(
            f"Unknown metrics requested: {unknown}. Allowed: {list(all_metric_fns.keys())}"
            )
        
        for metric_fn_name in wanted:
            metric_dict, hist_dict = all_metric_fns[metric_fn_name]()
            metrics[layer_name] |= metric_dict
            hists[layer_name] |= hist_dict

        if use_wandb:
            for metric_name, metric_value in metrics[layer_name].items():
                metrics_log[f"{prefix}{layer_name}/{metric_name}"] = metric_value
            
            if log_histograms:
                # Convert to numpy for histogram creation
                means_np = means.numpy()
                stds_np = stds.numpy()
                normalized_means_np = normalized_means.numpy()
                
                try:
                    import wandb
                    
                    metrics_log[f"{prefix}{layer_name}/act_means_hist"] = wandb.Histogram(means_np)
                    metrics_log[f"{prefix}{layer_name}/act_stds_hist"] = wandb.Histogram(stds_np)
                    metrics_log[f"{prefix}{layer_name}/act_normalized_means_hist"] = wandb.Histogram(normalized_means_np)

                    # Also log summary statistics about the means and stds
                    metrics_log[f"{prefix}{layer_name}/mean_of_means"] = means_np.mean()
                    metrics_log[f"{prefix}{layer_name}/std_of_means"] = means_np.std()
                    metrics_log[f"{prefix}{layer_name}/mean_of_stds"] = stds_np.mean()
                    metrics_log[f"{prefix}{layer_name}/std_of_stds"] = stds_np.std()
                    metrics_log[f"{prefix}{layer_name}/mean_of_normalized_means"] = normalized_means_np.mean()
                    metrics_log[f"{prefix}{layer_name}/std_of_normalized_means"] = normalized_means_np.std()

                    # Log any additional histograms from the metrics
                    for hist_name, hist_value in hists[layer_name].items():
                        metrics_log[f"{prefix}{layer_name}/{hist_name}"] = wandb.Histogram(hist_value)

                except (ImportError, Exception) as e:
                    print(f"Warning: Could not create wandb histograms: {e}")
    
    if not hooks_were_active:
        monitor.remove_hooks()
    
    return metrics, activation_stats, metrics_log

def create_module_filter(filters, model_name, cfg: DictConfig=None):
    """
    Create a filter function for selectively monitoring model layers.
    
    Args:
        filters: List of filter strings to match layer names against
        model_name: Name of the model being monitored
        cfg: Configuration object containing model-specific settings
    
    Returns:
        A function that takes a layer name and returns True if it should be monitored
    """
    
    if 'default' in filters:
        if model_name.lower() == 'resnet':
            # For ResNet: monitor main layers and direct block outputs, but not their internals
            def resnet_filter(name):
                # Match direct block layers (layer1_block0) but not internals with layers.
                if re.search(r'layer\d+_block\d+$', name):
                    return True
                # Also include other main model components
                if name in ['conv1', 'bn1', 'activation', 'avgpool', 'flatten', 'dropout', 'out']:
                    return True
                return False
            return resnet_filter
        
        elif model_name.lower() == 'vit':
            # For ViT: monitor main layers and direct block outputs, but not their internals
            def vit_filter(name):
                # Match direct block references (block_0) but not internals
                if re.search(r'block_\d+$', name):
                    return True
                # Also include other main model components
                if name in ['patch_embed', 'pos_drop', 'norm', 'out']:
                    return True
                return False
            return vit_filter
        
        elif model_name.lower() == 'mlp' or model_name.lower() == 'gated_mlp':
            # For MLP and Gated MLP: monitor all layers
            def mlp_filter(name):
                return True
            return mlp_filter
        elif model_name.lower() == 'cnn':
            # For CNN: monitor all layers
            def cnn_filter(name):
                return True
            return cnn_filter
    
    # Default case: match any of the provided filters
    return lambda name: any(f in name for f in filters)
