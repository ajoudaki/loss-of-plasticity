import torch
import torch.nn as nn
import torch.optim as optim
import time
import wandb
from collections import defaultdict

from ..utils.monitor import NetworkMonitor
from ..training.metrics import analyze_fixed_batch
from ..training.train import evaluate_model

def train_continual_learning(model, 
                           task_dataloaders, 
                           config, 
                           device='cpu'):
    """
    Train a model using continual learning on a sequence of tasks.
    
    Args:
        model: The neural network model
        task_dataloaders: Dictionary mapping task_id -> task data loaders
        config: Configuration dictionary
        device: Device to train on
    
    Returns:
        Dictionary with training history
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    dead_threshold=config['dead_threshold'] 
    corr_threshold=config['corr_threshold'] 
    saturation_threshold=config['saturation_threshold'] 
    saturation_percentage=config['saturation_percentage']

    # Create module filter function
    def module_filter(name):
        return 'linear' in name or '.mlp' in name or 'fc' in name or name.endswith('.proj')
    
    # For monitoring metrics
    train_monitor = NetworkMonitor(model, module_filter)
    val_monitor = NetworkMonitor(model, module_filter)
    
    # History tracking
    history = {
        'tasks': {}
    }
    
    print(f"Starting continual learning with {len(task_dataloaders)} tasks...")

    def analyze_callback(monitor, fixed_batch, fixed_targets):
        return analyze_fixed_batch(
            model, monitor, fixed_batch, fixed_targets, criterion, 
            device=device, 
            dead_threshold=dead_threshold, 
            corr_threshold=corr_threshold, 
            saturation_threshold=saturation_threshold, 
            saturation_percentage=saturation_percentage
        )
    
    for task_id, task_data in task_dataloaders.items():
        print(f"\n{'='*50}")
        print(f"Starting Task {task_id}: Classes {task_data['current']['classes']}")
        print(f"{'='*50}")
        
        current_train_loader = task_data['current']['train']
        current_val_loader = task_data['current']['val']
        current_fixed_train = task_data['current']['fixed_train']
        current_fixed_val = task_data['current']['fixed_val']
        
        task_history = {
            'epochs': [],
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'training_metrics_history': defaultdict(lambda: defaultdict(list)),
            'validation_metrics_history': defaultdict(lambda: defaultdict(list))
        }
        
        # Get a fixed batch for metrics
        try:
            fixed_train_batch, fixed_train_targets = next(iter(current_fixed_train))
            fixed_val_batch, fixed_val_targets = next(iter(current_fixed_val))
            
            fixed_train_batch = fixed_train_batch.to(device)
            fixed_train_targets = fixed_train_targets.to(device)
            fixed_val_batch = fixed_val_batch.to(device)
            fixed_val_targets = fixed_val_targets.to(device)
            
            # Initial metrics
            print("Measuring initial metrics...")
            
            train_metrics = analyze_callback(train_monitor, fixed_train_batch, fixed_train_targets)
            val_metrics = analyze_callback(val_monitor, fixed_val_batch, fixed_val_targets)
            
            for layer_name, metrics in train_metrics.items():
                for metric_name, value in metrics.items():
                    task_history['training_metrics_history'][layer_name][metric_name].append(value)
            
            for layer_name, metrics in val_metrics.items():
                for metric_name, value in metrics.items():
                    task_history['validation_metrics_history'][layer_name][metric_name].append(value)
        except StopIteration:
            print("Warning: Not enough samples for fixed batch metrics")
        
        # Evaluate on current task before training
        current_train_loss, current_train_acc = evaluate_model(model, current_train_loader, criterion, device)
        current_val_loss, current_val_acc = evaluate_model(model, current_val_loader, criterion, device)
        
        print(f"Initial performance on current task:")
        print(f"  Train Loss: {current_train_loss:.4f}, Train Acc: {current_train_acc:.2f}%")
        print(f"  Val Loss: {current_val_loss:.4f}, Val Acc: {current_val_acc:.2f}%")
        
        # Record initial metrics
        task_history['epochs'].append(0)
        task_history['train_loss'].append(current_train_loss)
        task_history['train_acc'].append(current_train_acc)
        task_history['val_loss'].append(current_val_loss)
        task_history['val_acc'].append(current_val_acc)
        
        # Training loop for this task
        start_time = time.time()
        for epoch in range(1, config["epochs_per_task"] + 1):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for inputs, targets in current_train_loader:
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
            
            epoch_train_loss = running_loss / len(current_train_loader)
            epoch_train_acc = 100. * correct / total
            
            # Evaluate on current task
            current_val_loss, current_val_acc = evaluate_model(model, current_val_loader, criterion, device)
            
            # Record current task metrics
            task_history['epochs'].append(epoch)
            task_history['train_loss'].append(epoch_train_loss)
            task_history['train_acc'].append(epoch_train_acc)
            task_history['val_loss'].append(current_val_loss)
            task_history['val_acc'].append(current_val_acc)
            
            # Periodically collect network metrics
            if epoch % config["metrics_frequency"] == 0 or epoch == config["epochs_per_task"]:
                try:
                    train_monitor.clear_data()
                    val_monitor.clear_data()
                    
                    train_metrics = analyze_callback(train_monitor, fixed_train_batch, fixed_train_targets)
                    val_metrics = analyze_callback(val_monitor, fixed_val_batch, fixed_val_targets)
                    
                    # Log current fixed batch metrics to wandb
                    fixed_metrics_log = {"task": task_id, "epoch": epoch}
                    for layer_name, metrics in train_metrics.items():
                        for metric_name, value in metrics.items():
                            fixed_metrics_log[f"fixed_train/{layer_name}/{metric_name}"] = value
                    for layer_name, metrics in val_metrics.items():
                        for metric_name, value in metrics.items():
                            fixed_metrics_log[f"fixed_val/{layer_name}/{metric_name}"] = value
                    
                    # Log all metrics to wandb
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
            
            # Log to wandb
            log_data = {
                "task": task_id,
                "epoch": epoch,
                "train_loss": epoch_train_loss,
                "train_acc": epoch_train_acc,
                "val_loss": current_val_loss,
                "val_acc": current_val_acc
            }
            
            wandb.log(log_data)
            
            # Print progress
            elapsed = time.time() - start_time
            print(f'Task {task_id}, Epoch {epoch}/{config["epochs_per_task"]}:')
            print(f'  Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%, '
                 f'Val Loss: {current_val_loss:.4f}, Val Acc: {current_val_acc:.2f}%')
            print(f'  Time: {elapsed:.2f}s')
        
        # Store task history
        history['tasks'][task_id] = {
            'classes': task_data['current']['classes'],
            'history': task_history
        }
    
    return history