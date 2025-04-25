import torch.nn as nn
import time
import wandb
from collections import defaultdict
from omegaconf import DictConfig
from typing import Dict, Any, Optional
import re

from ..utils.monitor import NetworkMonitor
from .eval import evaluate_model
from ..utils.metrics import analyze_fixed_batch
from ..config.utils import reinitialize_output_weights, create_optimizer

def train_continual_learning(model, 
                             task_dataloaders, 
                             cfg: DictConfig, 
                             device='cpu'):
    """
    Train a model using continual learning on a sequence of tasks.
    
    Args:
        model: The neural network model
        task_dataloaders: Dictionary mapping task_id -> task data loaders
        cfg: Hydra configuration object
        device: Device to train on
    
    Returns:
        Dictionary with training history
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(model, cfg)
    
    # Extract metrics parameters from config
    dead_threshold = cfg.metrics.dead_threshold 
    corr_threshold = cfg.metrics.corr_threshold 
    saturation_threshold = cfg.metrics.saturation_threshold 
    saturation_percentage = cfg.metrics.saturation_percentage
    gaussianity_method = cfg.metrics.gaussianity_method
    log_activation_histograms = cfg.metrics.log_activation_histograms

    # Create module filter function
    def create_module_filter(filters, model_name):
        # Check if we have model-specific filters (from model.metrics.monitor_filters)
        model_filters = None
        if hasattr(cfg.model, 'metrics') and hasattr(cfg.model.metrics, 'monitor_filters'):
            model_filters = cfg.model.metrics.monitor_filters
            print(f"Found model-specific filters: {model_filters}")
        
        # Use model-specific filters if available, otherwise use global metrics filters
        filters_to_use = model_filters if model_filters else filters
        print(f"Using filters: {filters_to_use} for model: {model_name}")
        
        if not filters_to_use:
            # If no filters, monitor all layers
            return lambda name: True
        
        if 'block_outputs' in filters_to_use:
            if model_name.lower() == 'resnet':
                # For ResNet: monitor main layers and direct block outputs, but not their internals
                def resnet_filter(name):
                    # Match direct block layers (layer1_block0) but not internals with layers.
                    if re.search(r'layer\d+_block\d+$', name):
                        return True
                    # Also include other main model components
                    if name in ['conv1', 'bn1', 'activation', 'avgpool', 'flatten', 'dropout', 'fc']:
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
                    if name in ['patch_embed', 'pos_drop', 'norm', 'head']:
                        return True
                    return False
                return vit_filter
        
        # Default case: match any of the provided filters
        return lambda name: any(f in name for f in filters_to_use)
    
    module_filter = create_module_filter(cfg.metrics.monitor_filters, cfg.model.name)
    
    # For monitoring metrics
    train_monitor = NetworkMonitor(model, module_filter)
    val_monitor = NetworkMonitor(model, module_filter)
    
    # History tracking
    history = {
        'tasks': {},
        'global_metrics': {
            'epochs': [],
            'steps': [],
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    }
    
    print(f"Starting continual learning with {len(task_dataloaders)} tasks...")

    # Track global counters
    global_epoch = 0
    global_step = 0
    
    def analyze_callback(monitor, fixed_batch, fixed_targets, prefix="", metrics_log=None):
        use_wandb = cfg.logging.use_wandb
        return analyze_fixed_batch(model, monitor, fixed_batch, fixed_targets, criterion, device=device, 
                                  dead_threshold=dead_threshold, corr_threshold=corr_threshold, 
                                  saturation_threshold=saturation_threshold, saturation_percentage=saturation_percentage,
                                  gaussianity_method=gaussianity_method,
                                  use_wandb=use_wandb,
                                  log_histograms=log_activation_histograms,
                                  prefix=prefix,
                                  metrics_log=metrics_log,
                                  seed=cfg.training.seed)
                                
    def analyze_train_callback(metrics_log=None):
        return analyze_callback(train_monitor, fixed_train_batch, fixed_train_targets, "train/", metrics_log)

    def analyze_val_callback(metrics_log=None):
        return analyze_callback(val_monitor, fixed_val_batch, fixed_val_targets, "val/", metrics_log)
    
    for task_id, task_data in task_dataloaders.items():
        print(f"\n{'='*50}")
        print(f"Starting Task {task_id}: Classes {task_data['classes']}")
        print(f"{'='*50}")
        
        train_loader = task_data['train']
        val_loader = task_data['val']
        fixed_train_loader = task_data['fixed_train']
        fixed_val_loader = task_data['fixed_val']
        
        # Reset entire model if configured
        if cfg.training.reset and task_id > 0:
            # Create a new model with the same configuration
            from ..models.model_factory import create_model
            new_model = create_model(cfg).to(device)
            model.load_state_dict(new_model.state_dict())
            del new_model
            print("Reinitialized all model weights for new task")
            # When we reset the model, we should also reset the optimizer
            optimizer = create_optimizer(model, cfg)
        # Reinitialize only output weights if configured
        elif cfg.training.reinit_output:
            reinitialize_output_weights(
                model, 
                task_data['classes'], 
                cfg.model.name.lower()
            )

        # Reinitialize optimizer state if configured (and we haven't already reset it)
        if cfg.optimizer.reinit_optimizer and task_id > 0 and not cfg.training.reset:
            optimizer = create_optimizer(model, cfg)
            print("Reinitialized optimizer state for new task")
        
        task_history = {
            'epochs': [],
            'steps': [],
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'training_metrics_history': defaultdict(lambda: defaultdict(list)),
            'validation_metrics_history': defaultdict(lambda: defaultdict(list))
        }
        
        # Track local step counter
        local_step = 0
        
        # Get a fixed batch for metrics
        try:
            fixed_train_batch, fixed_train_targets = next(iter(fixed_train_loader))
            fixed_val_batch, fixed_val_targets = next(iter(fixed_val_loader))
            
            fixed_train_batch = fixed_train_batch.to(device)
            fixed_train_targets = fixed_train_targets.to(device)
            fixed_val_batch = fixed_val_batch.to(device)
            fixed_val_targets = fixed_val_targets.to(device)
            
            # Initial metrics
            print("Measuring initial metrics...")
            
            # Create initial metrics log dictionary for wandb
            initial_metrics_log = {
                "task_id": task_id,
                "local_epoch": 0,
                "global_epoch": global_epoch,
                "local_step": 0,
                "global_step": global_step
            }
            
            # Get metrics - the function will populate metrics_log with all metrics
            train_metrics, train_act_stats, initial_metrics_log = analyze_train_callback(initial_metrics_log)
            val_metrics, val_act_stats, initial_metrics_log = analyze_val_callback(initial_metrics_log)
            
            # Store metrics in history
            for layer_name, metrics in train_metrics.items():
                for metric_name, value in metrics.items():
                    task_history['training_metrics_history'][layer_name][metric_name].append(value)
            
            for layer_name, metrics in val_metrics.items():
                for metric_name, value in metrics.items():
                    task_history['validation_metrics_history'][layer_name][metric_name].append(value)
                    
            # Log to wandb if enabled
            if cfg.logging.use_wandb:
                wandb.log(initial_metrics_log)
        except StopIteration:
            print("Warning: Not enough samples for fixed batch metrics")
        
        # Evaluate on task before training
        train_loss, train_acc = evaluate_model(model, train_loader, criterion, device)
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
        
        print(f"Initial performance:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Record initial metrics
        task_history['epochs'].append(0)
        task_history['steps'].append(local_step)
        task_history['train_loss'].append(train_loss)
        task_history['train_acc'].append(train_acc)
        task_history['val_loss'].append(val_loss)
        task_history['val_acc'].append(val_acc)
        
        # Record in global metrics
        history['global_metrics']['epochs'].append(global_epoch)
        history['global_metrics']['steps'].append(global_step)
        history['global_metrics']['train_loss'].append(train_loss)
        history['global_metrics']['train_acc'].append(train_acc)
        history['global_metrics']['val_loss'].append(val_loss)
        history['global_metrics']['val_acc'].append(val_acc)
        
        # Training loop for this task
        start_time = time.time()
        for local_epoch in range(1, cfg.training.epochs_per_task + 1):
            global_epoch += 1
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            batch_count = 0
            
            for inputs, targets in train_loader:
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
                local_step += 1
                global_step += 1
            
            epoch_train_loss = running_loss / batch_count
            epoch_train_acc = 100. * correct / total
            
            # Evaluate on task
            val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
            
            # Record task metrics
            task_history['epochs'].append(local_epoch)
            task_history['steps'].append(local_step)
            task_history['train_loss'].append(epoch_train_loss)
            task_history['train_acc'].append(epoch_train_acc)
            task_history['val_loss'].append(val_loss)
            task_history['val_acc'].append(val_acc)
            
            # Record in global metrics
            history['global_metrics']['epochs'].append(global_epoch)
            history['global_metrics']['steps'].append(global_step)
            history['global_metrics']['train_loss'].append(epoch_train_loss)
            history['global_metrics']['train_acc'].append(epoch_train_acc)
            history['global_metrics']['val_loss'].append(val_loss)
            history['global_metrics']['val_acc'].append(val_acc)
            
            # Periodically collect network metrics
            if local_epoch % cfg.metrics.metrics_frequency == 0 or local_epoch == cfg.training.epochs_per_task:
                try:
                    train_monitor.clear_data()
                    val_monitor.clear_data()
                    
                    # Create metrics log dictionary for wandb
                    fixed_metrics_log = {
                        "task_id": task_id, 
                        "local_epoch": local_epoch, 
                        "global_epoch": global_epoch,
                        "local_step": local_step,
                        "global_step": global_step
                    }
                    
                    # Get metrics - analyze_fixed_batch will populate the metrics_log
                    train_metrics, train_act_stats, fixed_metrics_log = analyze_train_callback(fixed_metrics_log)
                    val_metrics, val_act_stats, fixed_metrics_log = analyze_val_callback(fixed_metrics_log)
                    
                    # Log all metrics to wandb if enabled
                    if cfg.logging.use_wandb:
                        wandb.log(fixed_metrics_log)
                    
                    # Store metrics in history for later analysis
                    for layer_name, metrics in train_metrics.items():
                        for metric_name, value in metrics.items():
                            task_history['training_metrics_history'][layer_name][metric_name].append(value)
                    
                    for layer_name, metrics in val_metrics.items():
                        for metric_name, value in metrics.items():
                            task_history['validation_metrics_history'][layer_name][metric_name].append(value)
                except Exception as e:
                    print(f"Error collecting metrics: {e}")
            
            # Log to wandb if enabled
            log_data = {
                "task_id": task_id,
                "local_epoch": local_epoch,
                "global_epoch": global_epoch,
                "local_step": local_step,
                "global_step": global_step,
                "train_loss": epoch_train_loss,
                "train_acc": epoch_train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc
            }
            
            if cfg.logging.use_wandb:
                wandb.log(log_data)
            
            # Print progress
            elapsed = time.time() - start_time
            print(f'Task {task_id}, Epoch {local_epoch}/{cfg.training.epochs_per_task} (Global: {global_epoch}):')
            print(f'  Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print(f'  Steps: {local_step} (Global: {global_step}), Time: {elapsed:.2f}s')
        
        # Store task history
        history['tasks'][task_id] = {
            'classes': task_data['classes'],
            'history': task_history
        }
    
    return history