import torch
import numpy as np
from scipy import stats
import re
from omegaconf import DictConfig

def flatten_activations(layer_act):
    """Reshape layer activations to 2D matrix (samples × features)."""
    shape = layer_act.shape
    if len(shape) == 4:  # Convolutional layer
        return layer_act.permute(0, 2, 3, 1).contiguous().view(-1, shape[1])
    elif len(shape) == 3:  # Transformer layer
        return layer_act.contiguous().view(-1, shape[2])
    else:  # Linear layer
        return layer_act.view(-1, shape[1])

def compute_activation_statistics(layer_act):
    """
    Compute mean and standard deviation of activations for each unit.
    
    This function calculates the mean and standard deviation of activations for each
    neuron in a layer. These statistics provide insights into the activation distribution
    and can be used to detect neurons with unusual behavior.
    
    Args:
        layer_act: Layer activations of shape [batch_size, n_units]
        
    Returns:
        means: Mean activation of each unit
        stds: Standard deviation of each unit's activation
    
    Example:
        >>> means, stds = compute_activation_statistics(layer_activations)
        >>> print(f"Mean range: {means.min().item():.4f} to {means.max().item():.4f}")
        >>> print(f"Std range: {stds.min().item():.4f} to {stds.max().item():.4f}")
    """
    flattened_act = flatten_activations(layer_act)
    means = flattened_act.mean(dim=0)
    stds = flattened_act.std(dim=0)
    return means, stds

def measure_dead_neurons(layer_act, dead_threshold=0.95):
    """Measure fraction of neurons that are inactive (dead)."""
    flattened_act = flatten_activations(layer_act)
    is_zero = (flattened_act.abs() < 1e-7)
    frac_zero_per_neuron = is_zero.float().mean(dim=0)
    dead_mask = (frac_zero_per_neuron > dead_threshold)
    dead_fraction = dead_mask.float().mean().item()
    return dead_fraction

def measure_duplicate_neurons(layer_act, corr_threshold):
    """Measure fraction of neurons that are duplicates of others."""
    flattened_act = flatten_activations(layer_act)
    flattened_act = flattened_act.t()  
    flattened_act = torch.nn.functional.normalize(flattened_act, p=2, dim=1)
    similarity_matrix = torch.matmul(flattened_act, flattened_act.t())
    upper_tri_mask = torch.triu(torch.ones_like(similarity_matrix), diagonal=1).bool()
    dup_pairs = (similarity_matrix > corr_threshold) & upper_tri_mask
    neuron_is_dup = dup_pairs.any(dim=1)
    fraction_dup = neuron_is_dup.float().mean().item()
    return fraction_dup

def measure_effective_rank(layer_act, svd_sample_size=1024, seed=None):
    """
    Compute effective rank (entropy of normalized singular values).
    
    Args:
        layer_act: Layer activations
        svd_sample_size: Maximum number of samples to use for SVD
        seed: Optional random seed for sampling
    """
    flattened_act = flatten_activations(layer_act)
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
        return 0.0
    p = S / S_sum
    p_log_p = p * torch.log(p + 1e-12)
    eff_rank = torch.exp(-p_log_p.sum()).item()
    return eff_rank

def measure_stable_rank(layer_act, sample_size=1024, use_gram=True, seed=None):
    """
    Compute stable rank (squared Frobenius norm / spectral norm squared).
    
    Args:
        layer_act: Layer activations
        sample_size: Maximum number of samples to use
        use_gram: Whether to use the Gram matrix approach
        seed: Optional random seed for sampling
    """
    flattened_act = flatten_activations(layer_act)
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
    flattened_act = flattened_act - flattened_act.mean(dim=0, keepdim=True)
    if use_gram or D < N:
        frob_norm_sq = torch.sum(flattened_act**2).item()
        gram = torch.matmul(flattened_act.t(), flattened_act)
        trace_gram_squared = torch.sum(gram**2).item()
        if trace_gram_squared < 1e-9:
            return 0.0
        stable_rank = (frob_norm_sq**2) / trace_gram_squared
    else:
        cov = torch.matmul(flattened_act, flattened_act.t())
        trace_cov = torch.trace(cov).item()
        trace_cov_squared = torch.sum(cov**2).item()
        if trace_cov_squared < 1e-9:
            return 0.0
        stable_rank = (trace_cov**2) / trace_cov_squared
    return stable_rank

def measure_saturated_neurons(layer_act, layer_grad, saturation_threshold=1e-4, saturation_percentage=0.99):
    """
    Measures the fraction of saturated neurons in a layer.
    
    Saturated neurons are identified as those where the ratio of gradient magnitude
    to mean activation magnitude is very small, indicating the neuron is in a flat
    region of the loss landscape.
    """
    flattened_act = flatten_activations(layer_act)
    flattened_grad = flatten_activations(layer_grad)
    
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
    
    return saturated_fraction

def measure_gaussianity(layer_act, sample_size=1000, seed=None, method="shapiro"):
    """
    Measure the distance to Gaussianity for each neuron's activations using PyTorch operations.
    
    This version keeps all computations on the device (GPU/MPS) without transferring to CPU.
    
    Args:
        layer_act: Layer activations tensor of shape [batch_size, n_units]
        sample_size: Maximum number of samples to use for the test (for efficiency)
        seed: Optional random seed for sampling
        method: Method to use for Gaussianity testing:
                - "shapiro": Approximation of Shapiro-Wilk test using PyTorch
                - "ks": Approximation of Kolmogorov-Smirnov test
                - "anderson": Not available in device version, falls back to kurtosis
                - "kurtosis": Uses excess kurtosis as a measure of non-Gaussianity
    
    Returns:
        A measure of non-Gaussianity (averaged across all neurons in the layer).
    """
    flattened_act = flatten_activations(layer_act)
    N, D = flattened_act.shape
    
    # If we have more than sample_size samples, subsample to save computation
    if N > sample_size:
        # Use seed if provided, otherwise use the current random state
        if seed is not None:
            generator = torch.Generator(device=flattened_act.device)
            generator.manual_seed(seed)
            idx = torch.randperm(N, generator=generator, device=flattened_act.device)[:sample_size]
        else:
            idx = torch.randperm(N, device=flattened_act.device)[:sample_size]
        flattened_act = flattened_act[idx]
        N = sample_size
    
    if method == "kurtosis" or method == "anderson":  # anderson falls back to kurtosis
        # Use excess kurtosis as a measure of non-Gaussianity
        non_gaussianity = []
        for j in range(D):
            # Calculate standard deviation
            std_val = torch.std(flattened_act[:, j])
            
            # If std is very small (effectively constant values), skip the statistical test
            if std_val < 1e-6:
                # For constant values, consider them maximally non-Gaussian
                non_gaussianity.append(10.0)
                continue
                
            # Normalize to zero mean and unit variance
            x = (flattened_act[:, j] - torch.mean(flattened_act[:, j])) / (std_val + 1e-8)
            
            # Compute kurtosis: mean((x - mean(x))^4) / std(x)^4 - 3
            # The -3 makes the kurtosis of a normal distribution = 0
            mean_x = torch.mean(x)
            x_centered = x - mean_x
            kurtosis = torch.mean(x_centered**4) / (torch.mean(x_centered**2)**2) - 3
            non_gaussianity.append(min(abs(kurtosis.item()), 10.0))  # Cap at 10
    
    elif method == "shapiro":
        # Approximation of Shapiro-Wilk using sorted data and moments
        non_gaussianity = []
        for j in range(D):
            # Calculate standard deviation
            std_val = torch.std(flattened_act[:, j])
            
            # If std is very small (effectively constant values), skip the statistical test
            if std_val < 1e-6:
                # For constant values, consider them maximally non-Gaussian
                non_gaussianity.append(1.0)
                continue
                
            # Normalize to zero mean and unit variance
            x = (flattened_act[:, j] - torch.mean(flattened_act[:, j])) / (std_val + 1e-8)
            
            # Sort the values (this is a key part of Shapiro-Wilk)
            x_sorted, _ = torch.sort(x)
            
            # Simple approximation based on comparing ordered data to expected normal values
            # This isn't a real Shapiro-Wilk test but gives a rough measure of non-Gaussianity
            n = x.size(0)
            
            # Create expected values for a normal distribution
            # We use a linear approximation of the normal quantile function for simplicity
            positions = torch.linspace(1/(n*2), 1-1/(n*2), n, device=x.device)
            
            # Fix for the error: Create a tensor for sqrt(2)
            sqrt_2 = torch.tensor(2.0, device=x.device).sqrt()
            
            # Simple approximation of normal quantiles using an error function approximation
            normal_approx = sqrt_2 * torch.erfinv(2 * positions - 1)
            
            # Calculate correlation between sorted data and expected normal values
            # Higher correlation means more Gaussian
            mean_x_sorted = torch.mean(x_sorted)
            mean_normal = torch.mean(normal_approx)
            numerator = torch.sum((x_sorted - mean_x_sorted) * (normal_approx - mean_normal))
            denominator = torch.sqrt(torch.sum((x_sorted - mean_x_sorted)**2) * torch.sum((normal_approx - mean_normal)**2))
            correlation = numerator / denominator
            
            # Convert to non-Gaussianity score (1 - correlation)
            # This ranges from 0 (perfectly Gaussian) to 1 (maximally non-Gaussian)
            non_gaussianity.append(1.0 - correlation.item())
    
    elif method == "ks":
        # Approximation of KS test using empirical CDF
        non_gaussianity = []
        for j in range(D):
            # Calculate standard deviation
            std_val = torch.std(flattened_act[:, j])
            
            # If std is very small (effectively constant values), skip the statistical test
            if std_val < 1e-6:
                # For constant values, consider them maximally non-Gaussian
                non_gaussianity.append(1.0)
                continue
                
            # Normalize to zero mean and unit variance
            x = (flattened_act[:, j] - torch.mean(flattened_act[:, j])) / (std_val + 1e-8)
            
            # Sort the values to compute empirical CDF
            x_sorted, _ = torch.sort(x)
            n = x_sorted.size(0)
            
            # Empirical CDF (from 1/n to 1.0)
            ecdf = torch.linspace(1/n, 1.0, n, device=x.device)
            
            # Create tensor for sqrt(2)
            sqrt_2 = torch.tensor(2.0, device=x.device).sqrt()
            
            # Theoretical CDF for normal distribution: 0.5 * (1 + erf(x / sqrt(2)))
            tcdf = 0.5 * (1 + torch.erf(x_sorted / sqrt_2))
            
            # KS statistic is the maximum absolute difference between ECDFs
            ks_stat = torch.max(torch.abs(tcdf - ecdf)).item()
            non_gaussianity.append(ks_stat)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Average non-Gaussianity across all neurons
    mean_non_gaussianity = sum(non_gaussianity) / len(non_gaussianity)
    
    return mean_non_gaussianity

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
        seed: Optional random seed for sampling operations (for reproducibility)
        
    Returns:
        Tuple containing:
        - Dictionary of metrics for each layer (metric_name -> value)
        - Dictionary of activation statistics for each layer (means, stds)
        - Dictionary of metrics formatted for wandb logging (if use_wandb is True)
    
    Example:
        >>> metrics, act_stats, metrics_log = analyze_fixed_batch(
        >>>     model, monitor, batch, targets, loss_fn,
        >>>     dead_threshold=0.95, corr_threshold=0.95,
        >>>     use_wandb=True, log_histograms=True, prefix="train/"
        >>> )
        >>> # metrics contains numerical values for each computed metric
        >>> # act_stats contains activation means and standard deviations
        >>> # metrics_log is ready for wandb.log()
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
    
    metrics = {}
    activation_stats = {}
    latest_acts = monitor.get_latest_activations()
    latest_grads = monitor.get_latest_gradients()
    
    # Create or use provided metrics log dict
    if metrics_log is None:
        metrics_log = {}

    for layer_name, act in latest_acts.items():
        # Skip layers without gradients when computing metrics
        if layer_name not in latest_grads:
            continue
            
        grad = latest_grads[layer_name]
        
        # Compute neuron activation statistics (mean and std)
        means, stds = compute_activation_statistics(act)
        activation_stats[layer_name] = {
            'means': means.detach().cpu(),
            'stds': stds.detach().cpu()
        }
        
        # Compute all metrics for this layer
        metrics[layer_name] = {
            'dead_fraction': measure_dead_neurons(act, dead_threshold),
            'dup_fraction': measure_duplicate_neurons(act, corr_threshold),
            'eff_rank': measure_effective_rank(act, seed=seed),
            'stable_rank': measure_stable_rank(act, seed=seed),
            'saturated_frac': measure_saturated_neurons(act, grad, saturation_threshold, saturation_percentage),
            'non_gaussianity': measure_gaussianity(act, seed=seed, method=gaussianity_method),
        }
        
        # Add metrics to the metrics_log for wandb if enabled
        if use_wandb:
            for metric_name, value in metrics[layer_name].items():
                metrics_log[f"{prefix}{layer_name}/{metric_name}"] = value
            
            # Add histograms and statistics if requested
            if log_histograms:
                # Convert to numpy for histogram creation
                means_np = means.numpy()
                stds_np = stds.numpy()
                
                try:
                    # Only import wandb if we need it (makes the function still usable without wandb)
                    import wandb
                    
                    # Add histograms
                    metrics_log[f"{prefix}{layer_name}/act_means_hist"] = wandb.Histogram(means_np)
                    metrics_log[f"{prefix}{layer_name}/act_stds_hist"] = wandb.Histogram(stds_np)
                    
                    # Also log summary statistics about the means and stds
                    metrics_log[f"{prefix}{layer_name}/mean_of_means"] = means_np.mean()
                    metrics_log[f"{prefix}{layer_name}/std_of_means"] = means_np.std()
                    metrics_log[f"{prefix}{layer_name}/mean_of_stds"] = stds_np.mean()
                    metrics_log[f"{prefix}{layer_name}/std_of_stds"] = stds_np.std()
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
        
        elif model_name.lower() == 'mlp':
            # For MLP: monitor all layers
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