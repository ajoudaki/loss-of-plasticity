import torch

def flatten_activations(layer_act):
    """Reshape layer activations to 2D matrix (samples × features)."""
    shape = layer_act.shape
    if len(shape) == 4:  # Convolutional layer
        return layer_act.permute(0, 2, 3, 1).contiguous().view(-1, shape[1])
    elif len(shape) == 3:  # Transformer layer
        return layer_act.contiguous().view(-1, shape[2])
    else:  # Linear layer
        return layer_act.view(-1, shape[1])

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

def measure_effective_rank(layer_act, svd_sample_size=1024):
    """Compute effective rank (entropy of normalized singular values)."""
    flattened_act = flatten_activations(layer_act)
    N = flattened_act.shape[0]
    if N > svd_sample_size:
        idx = torch.randperm(N)[:svd_sample_size]
        flattened_act = flattened_act[idx]
    U, S, Vt = torch.linalg.svd(flattened_act, full_matrices=False)
    S_sum = S.sum()
    if S_sum < 1e-9:
        return 0.0
    p = S / S_sum
    p_log_p = p * torch.log(p + 1e-12)
    eff_rank = torch.exp(-p_log_p.sum()).item()
    return eff_rank

def measure_stable_rank(layer_act, sample_size=1024, use_gram=True):
    """Compute stable rank (squared Frobenius norm / spectral norm squared)."""
    flattened_act = flatten_activations(layer_act)
    N, D = flattened_act.shape
    if N > sample_size:
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


def analyze_fixed_batch(model, monitor, fixed_batch, fixed_targets, criterion, 
                      dead_threshold, 
                      corr_threshold, 
                      saturation_threshold, 
                      saturation_percentage,
                      device='cpu'):
    """
    Analyze model behavior on a fixed batch to compute metrics.
    
    Args:
        model: Neural network model
        monitor: NetworkMonitor instance
        fixed_batch: Input data batch
        fixed_targets: Target labels
        criterion: Loss function
        dead_threshold: Threshold for dead neuron detection
        corr_threshold: Threshold for duplicate neuron detection
        saturation_threshold: Threshold for saturated neuron detection
        saturation_percentage: Percentage of samples required for a neuron to be considered saturated
        device: Device to run computations on
        
    Returns:
        Dictionary of metrics for each layer
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
    latest_acts = monitor.get_latest_activations()
    latest_grads = monitor.get_latest_gradients()

    for layer_name, act in latest_acts.items():
        # Skip layers without gradients when computing metrics
        if layer_name not in latest_grads:
            continue
            
        grad = latest_grads[layer_name]
        
        # Compute all metrics for this layer
        metrics[layer_name] = {
            'dead_fraction': measure_dead_neurons(act, dead_threshold),
            'dup_fraction': measure_duplicate_neurons(act, corr_threshold),
            'eff_rank': measure_effective_rank(act),
            'stable_rank': measure_stable_rank(act),
            'saturated_frac': measure_saturated_neurons(act, grad, saturation_threshold, saturation_percentage),
        }
    
    if not hooks_were_active:
        monitor.remove_hooks()
    
    return metrics