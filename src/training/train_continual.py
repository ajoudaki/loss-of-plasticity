import torch.nn as nn
import time
import wandb
from collections import defaultdict
from omegaconf import DictConfig
from typing import Dict, Any, Optional

from ..utils.monitor import NetworkMonitor
from .eval import evaluate_model
from ..utils.metrics import analyze_fixed_batch
from .training_utils import reinitialize_output_weights, create_optimizer

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

    # Create module filter function
    def module_filter(name):
        return 'linear' in name or '.mlp' in name or 'fc' in name or name.endswith('.proj')
    
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
    
    def analyze_callback(monitor, fixed_batch, fixed_targets):
        return analyze_fixed_batch(model, monitor, fixed_batch, fixed_targets, criterion, device=device, 
                                  dead_threshold=dead_threshold, corr_threshold=corr_threshold, 
                                  saturation_threshold=saturation_threshold, saturation_percentage=saturation_percentage)
                                
    def analyze_train_callback():
        return analyze_callback(train_monitor, fixed_train_batch, fixed_train_targets)

    def analyze_val_callback():
        return analyze_callback(val_monitor, fixed_val_batch, fixed_val_targets)
    
    for task_id, task_data in task_dataloaders.items():
        print(f"\n{'='*50}")
        print(f"Starting Task {task_id}: Classes {task_data['classes']}")
        print(f"{'='*50}")
        
        train_loader = task_data['train']
        val_loader = task_data['val']
        fixed_train_loader = task_data['fixed_train']
        fixed_val_loader = task_data['fixed_val']
        
        # Reinitialize output weights if configured
        if cfg.training.reinit_output:
            reinitialize_output_weights(
                model, 
                task_data['classes'], 
                cfg.model.name.lower()
            )

        # Reinitialize optimizer state if configured
        if cfg.optimizer.reinit_adam and task_id > 0:
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
            
            train_metrics = analyze_train_callback()
            val_metrics = analyze_val_callback()
            
            for layer_name, metrics in train_metrics.items():
                for metric_name, value in metrics.items():
                    task_history['training_metrics_history'][layer_name][metric_name].append(value)
            
            for layer_name, metrics in val_metrics.items():
                for metric_name, value in metrics.items():
                    task_history['validation_metrics_history'][layer_name][metric_name].append(value)
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
                    
                    train_metrics = analyze_callback(train_monitor, fixed_train_batch, fixed_train_targets)
                    val_metrics = analyze_callback(val_monitor, fixed_val_batch, fixed_val_targets)
                    
                    # Log metrics to wandb
                    fixed_metrics_log = {
                        "task_id": task_id, 
                        "local_epoch": local_epoch, 
                        "global_epoch": global_epoch,
                        "local_step": local_step,
                        "global_step": global_step
                    }
                    
                    for layer_name, metrics in train_metrics.items():
                        for metric_name, value in metrics.items():
                            fixed_metrics_log[f"train/{layer_name}/{metric_name}"] = value
                    
                    for layer_name, metrics in val_metrics.items():
                        for metric_name, value in metrics.items():
                            fixed_metrics_log[f"val/{layer_name}/{metric_name}"] = value
                    
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