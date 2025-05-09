import torch
import torch.nn as nn
import time
import wandb
from collections import defaultdict
from omegaconf import DictConfig
from typing import Dict, Any, Optional, Tuple, List
import copy

from ..utils.monitor import NetworkMonitor
from .eval import evaluate_model
from ..utils.metrics import analyze_fixed_batch, create_module_filter
from ..config.utils import create_optimizer
from ..utils.cloning import create_cloned_model, test_activation_cloning


def train_cloning_experiment(original_model, 
                           dataloaders, 
                           cfg: DictConfig, 
                           device='cpu'):
    """
    Train a model using the cloning approach to study the effects of neuron duplication.
    
    This experiment first trains a base model, then creates expanded versions by cloning
    neurons/channels, and studies their behavior compared to the original model.
    
    Args:
        original_model: The original neural network model (will be trained first)
        dataloaders: Dictionary with train/val/test dataloaders (using first task only)
        cfg: Hydra configuration object
        device: Device to train on
    
    Returns:
        Dictionary with training history
    """
    # Extract the first task's dataloaders (we don't use tasks in cloning experiments)
    task_id = list(dataloaders.keys())[0]
    train_loader = dataloaders[task_id]['train']
    val_loader = dataloaders[task_id]['val']
    fixed_train_loader = dataloaders[task_id]['fixed_train']
    fixed_val_loader = dataloaders[task_id]['fixed_val']
    
    # Print experiment configuration
    print(f"\n{'='*60}")
    print(f"Starting Cloning Experiment with the following settings:")
    print(f"- Initial epochs: {cfg.training.initial_epochs}")
    print(f"- Epochs per expansion: {cfg.training.epochs_per_expansion}")
    print(f"- Expansion factor: {cfg.training.expansion_factor}")
    print(f"- Number of expansions: {cfg.training.num_expansions}")
    print(f"{'='*60}\n")
    
    # Setup for tracking all models (base and expanded)
    models = []
    optimizers = []
    histories = []
    monitors = []
    
    # Start with the base model
    base_model = original_model
    base_optimizer = create_optimizer(base_model, cfg)
    base_criterion = nn.CrossEntropyLoss()
    
    # Create a filter function for metrics monitoring
    module_filter = create_module_filter(cfg.metrics.monitor_filters, cfg.model.name, cfg)
    
    # Create a monitor for the base model
    base_monitor = NetworkMonitor(base_model, module_filter)
    
    # Store initial models, optimizers, and monitors
    models.append(base_model)
    optimizers.append(base_optimizer)
    monitors.append(base_monitor)
    
    # Create a history structure for the experiment
    experiment_history = {
        'models': [],
        'global_metrics': {
            'epochs': [],
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    }
    
    # Create a base history for the original model
    base_history = {
        'model_type': 'base',
        'expansion_factor': 1,
        'epochs': [],
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'metrics_history': {}
    }
    
    # Add the base history to the experiment
    experiment_history['models'].append(base_history)
    
    # Keep track of global epoch counter
    global_epoch = 0
    
    # Get fixed batches for metrics computation
    try:
        fixed_train_batch, fixed_train_targets = next(iter(fixed_train_loader))
        fixed_val_batch, fixed_val_targets = next(iter(fixed_val_loader))
        
        fixed_train_batch = fixed_train_batch.to(device)
        fixed_train_targets = fixed_train_targets.to(device)
        fixed_val_batch = fixed_val_batch.to(device)
        fixed_val_targets = fixed_val_targets.to(device)
    except StopIteration:
        print("Warning: Not enough samples for fixed batch metrics")
        fixed_train_batch = fixed_train_targets = fixed_val_batch = fixed_val_targets = None
    
    # Helper function for computing and storing metrics
    def compute_and_store_metrics(model, monitor, model_history, curr_epoch, prefix=""):
        """Compute and store metrics for a model at a given epoch."""
        metrics_log = {
            "epoch": curr_epoch,
            "global_epoch": global_epoch,
            "model_type": model_history['model_type'],
            "expansion_factor": model_history['expansion_factor']
        }
        
        # Train metrics
        if fixed_train_batch is not None:
            train_metrics, train_act_stats, metrics_log = analyze_callback(
                model, monitor, fixed_train_batch, fixed_train_targets, 
                f"{prefix}train/", metrics_log
            )
            
            # Store metrics
            for layer_name, metrics in train_metrics.items():
                for metric_name, value in metrics.items():
                    metric_key = f"train/{layer_name}/{metric_name}"
                    if metric_key not in model_history['metrics_history']:
                        model_history['metrics_history'][metric_key] = []
                    model_history['metrics_history'][metric_key].append(value)
        
        # Validation metrics
        if fixed_val_batch is not None:
            val_metrics, val_act_stats, metrics_log = analyze_callback(
                model, monitor, fixed_val_batch, fixed_val_targets, 
                f"{prefix}val/", metrics_log
            )
            
            # Store metrics
            for layer_name, metrics in val_metrics.items():
                for metric_name, value in metrics.items():
                    metric_key = f"val/{layer_name}/{metric_name}"
                    if metric_key not in model_history['metrics_history']:
                        model_history['metrics_history'][metric_key] = []
                    model_history['metrics_history'][metric_key].append(value)
        
        # Log to wandb if enabled
        if cfg.use_wandb:
            wandb.log(metrics_log)
        
        return metrics_log
    
    # Helper function for analyze_fixed_batch with consistent parameters
    def analyze_callback(model, monitor, fixed_batch, fixed_targets, prefix="", metrics_log=None):
        """Run analyze_fixed_batch with consistent parameters."""
        use_wandb = cfg.use_wandb
        return analyze_fixed_batch(
            model, monitor, fixed_batch, fixed_targets, 
            nn.CrossEntropyLoss(), device=device, 
            dead_threshold=cfg.metrics.dead_threshold, 
            corr_threshold=cfg.metrics.corr_threshold, 
            saturation_threshold=cfg.metrics.saturation_threshold, 
            saturation_percentage=cfg.metrics.saturation_percentage,
            gaussianity_method=cfg.metrics.gaussianity_method,
            use_wandb=use_wandb,
            log_histograms=cfg.metrics.log_activation_histograms,
            prefix=prefix,
            metrics_log=metrics_log,
            seed=cfg.training.seed
        )
    
    # === First stage: Train the base model ===
    print(f"\n{'='*60}")
    print(f"Stage 1: Training base model for {cfg.training.initial_epochs} epochs")
    print(f"{'='*60}\n")
    
    # Evaluate the untrained model first
    base_val_loss, base_val_acc = evaluate_model(base_model, val_loader, base_criterion, device)
    print(f"Initial performance - Val Loss: {base_val_loss:.4f}, Val Acc: {base_val_acc:.2f}%")
    
    # Record initial metrics
    base_history['epochs'].append(0)
    base_history['train_loss'].append(0)  # Placeholder
    base_history['train_acc'].append(0)   # Placeholder
    base_history['val_loss'].append(base_val_loss)
    base_history['val_acc'].append(base_val_acc)
    
    # Compute initial metrics
    if fixed_train_batch is not None:
        compute_and_store_metrics(base_model, base_monitor, base_history, 0, "base_")
    
    # Train the base model
    start_time = time.time()
    for epoch in range(1, cfg.training.initial_epochs + 1):
        # Train for one epoch
        train_loss, train_acc = train_epoch(base_model, train_loader, base_criterion, base_optimizer, device)
        
        # Evaluate
        val_loss, val_acc = evaluate_model(base_model, val_loader, base_criterion, device)
        
        # Record metrics
        base_history['epochs'].append(epoch)
        base_history['train_loss'].append(train_loss)
        base_history['train_acc'].append(train_acc)
        base_history['val_loss'].append(val_loss)
        base_history['val_acc'].append(val_acc)
        
        # Global metrics
        global_epoch += 1
        experiment_history['global_metrics']['epochs'].append(global_epoch)
        experiment_history['global_metrics']['train_loss'].append(train_loss)
        experiment_history['global_metrics']['train_acc'].append(train_acc)
        experiment_history['global_metrics']['val_loss'].append(val_loss)
        experiment_history['global_metrics']['val_acc'].append(val_acc)
        
        # Compute metrics at set frequency
        if epoch % cfg.metrics.metrics_frequency == 0 or epoch == cfg.training.initial_epochs:
            if fixed_train_batch is not None:
                base_monitor.clear_data()
                compute_and_store_metrics(base_model, base_monitor, base_history, epoch, "base_")
        
        # Log to wandb
        if cfg.use_wandb:
            wandb.log({
                "epoch": epoch,
                "global_epoch": global_epoch,
                "model_type": "base",
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc
            })
        
        # Print progress
        elapsed = time.time() - start_time
        print(f'Base model - Epoch {epoch}/{cfg.training.initial_epochs}, '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, '
              f'Time: {elapsed:.2f}s')
    
    # === Second stage: Create and train expanded models ===
    current_model = base_model
    current_expansion = 1
    
    for expansion_idx in range(cfg.training.num_expansions):
        current_expansion *= cfg.training.expansion_factor
        
        print(f"\n{'='*60}")
        print(f"Stage {expansion_idx + 2}: Training {current_expansion}x expanded model "
              f"for {cfg.training.epochs_per_expansion} epochs")
        print(f"{'='*60}\n")
        
        # Create expanded model by cloning the current model
        expanded_model = create_cloned_model(
            current_model, 
            cfg, 
            cfg.training.expansion_factor
        ).to(device)
        
        # Test if cloning is successful 
        train_success, train_unexplained_var = test_activation_cloning(current_model, expanded_model, fixed_train_batch, fixed_train_targets, tolerance=1e-3, model_name=cfg.model.name)
        
        val_success, val_unexplained_var = test_activation_cloning(current_model, expanded_model, fixed_val_batch, fixed_val_targets, tolerance=1e-3, model_name=cfg.model.name)
        print(f"✓ Epoch {epoch}: Cloning validation successful - activations match between models")


        # Add cloning metrics to wandb log
        if cfg.use_wandb:
            similarity_log = {
                "epoch": epoch,
                "global_epoch": global_epoch,
                "model_type": cfg.model.name,
                f"expanded_{current_expansion}x_train/success": float(train_success),
                f"expanded_{current_expansion}x_train/unexplained_var": sum(train_unexplained_var.values())/len(train_unexplained_var),
                f"expanded_{current_expansion}x_val/success": float(val_success),
                f"expanded_{current_expansion}x_val/unexplained_var": sum(val_unexplained_var.values())/len(val_unexplained_var),
            }
            for k,v in train_unexplained_var.items():
                similarity_log[f"expanded_{current_expansion}x_train/unexplained_var/{k}"] = v
                similarity_log[f"expanded_{current_expansion}x_val/unexplained_var/{k}"] = val_unexplained_var[k]
            wandb.log(similarity_log)
        
        # Create optimizer for expanded model
        expanded_optimizer = create_optimizer(expanded_model, cfg)
        expanded_criterion = nn.CrossEntropyLoss()
        
        # Create monitor for expanded model
        expanded_monitor = NetworkMonitor(expanded_model, module_filter)
        
        # Create history for expanded model
        expanded_history = {
            'model_type': f'expanded_{current_expansion}x',
            'expansion_factor': current_expansion,
            'epochs': [],
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'metrics_history': {}
        }
        
        # Add to lists
        models.append(expanded_model)
        optimizers.append(expanded_optimizer)
        monitors.append(expanded_monitor)
        experiment_history['models'].append(expanded_history)
        
        # Evaluate initial performance
        exp_val_loss, exp_val_acc = evaluate_model(expanded_model, val_loader, expanded_criterion, device)
        print(f"Initial performance - Val Loss: {exp_val_loss:.4f}, Val Acc: {exp_val_acc:.2f}%")
        
        # Record initial metrics
        expanded_history['epochs'].append(0)
        expanded_history['train_loss'].append(0)  # Placeholder
        expanded_history['train_acc'].append(0)   # Placeholder
        expanded_history['val_loss'].append(exp_val_loss)
        expanded_history['val_acc'].append(exp_val_acc)
        
        # Compute initial metrics
        if fixed_train_batch is not None:
            compute_and_store_metrics(
                expanded_model, 
                expanded_monitor, 
                expanded_history, 
                0, 
                f"expanded_{current_expansion}x_"
            )
        
        # Train the expanded model
        start_time = time.time()
        for epoch in range(1, cfg.training.epochs_per_expansion + 1):
            # Train original model if tracking is enabled
            if cfg.training.track_original:
                orig_train_loss, orig_train_acc = train_epoch(
                    current_model, train_loader, base_criterion, base_optimizer, device
                )
                orig_val_loss, orig_val_acc = evaluate_model(
                    current_model, val_loader, base_criterion, device
                )
            
            # Train expanded model
            exp_train_loss, exp_train_acc = train_epoch(
                expanded_model, train_loader, expanded_criterion, expanded_optimizer, device
            )
            
            # Evaluate expanded model
            exp_val_loss, exp_val_acc = evaluate_model(
                expanded_model, val_loader, expanded_criterion, device
            )
            
            # Record metrics for expanded model
            expanded_history['epochs'].append(epoch)
            expanded_history['train_loss'].append(exp_train_loss)
            expanded_history['train_acc'].append(exp_train_acc)
            expanded_history['val_loss'].append(exp_val_loss)
            expanded_history['val_acc'].append(exp_val_acc)
            
            # Global metrics (use expanded model)
            global_epoch += 1
            experiment_history['global_metrics']['epochs'].append(global_epoch)
            experiment_history['global_metrics']['train_loss'].append(exp_train_loss)
            experiment_history['global_metrics']['train_acc'].append(exp_train_acc)
            experiment_history['global_metrics']['val_loss'].append(exp_val_loss)
            experiment_history['global_metrics']['val_acc'].append(exp_val_acc)
            
            # Compute metrics at set frequency
            if epoch % cfg.metrics.metrics_frequency == 0 or epoch == cfg.training.epochs_per_expansion:
                if fixed_train_batch is not None:
                    # Reset monitors
                    expanded_monitor.clear_data()
                    if cfg.training.track_original:
                        base_monitor.clear_data()
                    
                    # Compute metrics for expanded model
                    metrics_log = compute_and_store_metrics(
                        expanded_model, 
                        expanded_monitor, 
                        expanded_history, 
                        epoch, 
                        f"expanded_{current_expansion}x_"
                    )
                    
                    # Compute metrics for original model if tracking
                    if cfg.training.track_original:
                        compute_and_store_metrics(
                            current_model, 
                            base_monitor, 
                            base_history,  # Update the base history
                            epoch, 
                            "base_"
                        )
                    
                    # Test if cloning is still effective at the end of each epoch
                    train_success, train_unexplained_var = test_activation_cloning(current_model, expanded_model, fixed_train_batch, fixed_train_targets, model_name=cfg.model.name, tolerance=1e-3)
                    
                    val_success, val_unexplained_var = test_activation_cloning(current_model, expanded_model, fixed_val_batch, fixed_val_targets, model_name=cfg.model.name, tolerance=1e-3)
                    print(f"✓ Epoch {epoch}: Cloning training set: {train_success} validation success: {val_success} ")


                    # Add cloning metrics to wandb log
                    if cfg.use_wandb:
                        similarity_log = {
                            "epoch": epoch,
                            "global_epoch": global_epoch,
                            "model_type": expanded_history['model_type'],
                            f"expanded_{current_expansion}x_train/success": float(train_success),
                            f"expanded_{current_expansion}x_train/unexplained_var": sum(train_unexplained_var.values())/len(train_unexplained_var),
                            f"expanded_{current_expansion}x_val/success": float(val_success),
                            f"expanded_{current_expansion}x_val/unexplained_var": sum(val_unexplained_var.values())/len(val_unexplained_var),
                        }
                        for k,v in train_unexplained_var.items():
                            similarity_log[f"expanded_{current_expansion}x_train/unexplained_var/{k}"] = v
                            similarity_log[f"expanded_{current_expansion}x_val/unexplained_var/{k}"] = val_unexplained_var[k]
                        wandb.log(similarity_log)

                        # Store in history
                        metric_key = "cloning_test_success"
                        if metric_key not in expanded_history['metrics_history']:
                            expanded_history['metrics_history'][metric_key] = []
                        expanded_history['metrics_history'][metric_key].append(1.0)

            
            # Log to wandb
            if cfg.use_wandb:
                log_data = {
                    "epoch": epoch,
                    "global_epoch": global_epoch,
                    "model_type": expanded_history['model_type'],
                    "expansion_factor": current_expansion,
                    "train_loss": exp_train_loss,
                    "train_acc": exp_train_acc,
                    "val_loss": exp_val_loss,
                    "val_acc": exp_val_acc
                }
                
                if cfg.training.track_original:
                    log_data.update({
                        "original_train_loss": orig_train_loss,
                        "original_train_acc": orig_train_acc,
                        "original_val_loss": orig_val_loss,
                        "original_val_acc": orig_val_acc
                    })
                
                wandb.log(log_data)
            
            # Print progress
            elapsed = time.time() - start_time
            print(f'Expanded {current_expansion}x model - Epoch {epoch}/{cfg.training.epochs_per_expansion}, '
                  f'Train Loss: {exp_train_loss:.4f}, Train Acc: {exp_train_acc:.2f}%, '
                  f'Val Loss: {exp_val_loss:.4f}, Val Acc: {exp_val_acc:.2f}%, '
                  f'Time: {elapsed:.2f}s')
            
            if cfg.training.track_original:
                print(f'Original model - '
                      f'Train Loss: {orig_train_loss:.4f}, Train Acc: {orig_train_acc:.2f}%, '
                      f'Val Loss: {orig_val_loss:.4f}, Val Acc: {orig_val_acc:.2f}%')
        
        # Use this expanded model as the base for the next expansion
        current_model = expanded_model
    
    return experiment_history


def train_epoch(model, dataloader, criterion, optimizer, device='cpu') -> Tuple[float, float]:
    """Train model for one epoch and return average loss and accuracy."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    batch_count = 0
    
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        batch_count += 1
    
    epoch_loss = running_loss / batch_count
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


