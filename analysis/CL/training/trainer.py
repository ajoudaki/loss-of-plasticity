"""
Training utilities for continual learning experiments.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import copy

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from .metrics import (
    compute_accuracy,
    compute_rank_metrics,
    compute_connected_components
)


class Trainer:
    """
    General trainer for supervised learning models.
    """
    def __init__(self, 
                 model, 
                 train_loader, 
                 val_loader=None, 
                 criterion=None,
                 optimizer=None,
                 scheduler=None,
                 device=None,
                 metrics_config=None,
                 checkpoint_dir='./checkpoints',
                 use_wandb=False,
                 wandb_project='continual_learning',
                 wandb_entity=None,
                 wandb_config=None):
        """
        Initialize the trainer.
        
        Parameters:
            model (nn.Module): Model to train
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader, optional): Validation data loader
            criterion (callable, optional): Loss function (defaults to CrossEntropyLoss)
            optimizer (torch.optim.Optimizer, optional): Optimizer (defaults to Adam)
            scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Learning rate scheduler
            device (torch.device, optional): Device to use (defaults to cuda if available, otherwise cpu)
            metrics_config (dict, optional): Configuration for additional metrics
            checkpoint_dir (str): Directory to save checkpoints
            use_wandb (bool): Whether to log with Weights & Biases
            wandb_project (str): W&B project name
            wandb_entity (str, optional): W&B entity name
            wandb_config (dict, optional): Additional W&B configuration
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Set up criterion, optimizer, and scheduler
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
        self.optimizer = optimizer if optimizer is not None else optim.Adam(model.parameters(), lr=0.001)
        self.scheduler = scheduler
        
        # Set up device
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Set up metrics configuration
        default_metrics_config = {
            'compute_rank_metrics': False,
            'compute_component_metrics': False,
            'compute_activation_stats': False,
            'compute_weight_stats': False,
            'track_gradient_stats': False,
            'record_activations': False,
            'record_activations_freq': 10,  # epochs between recording
            'record_sample_size': 100,  # number of samples for recording
            'record_activations_for_layers': None,  # record all layers if None
            'record_gradients_for_layers': None,  # record all layers if None
        }
        self.metrics_config = default_metrics_config.copy()
        if metrics_config:
            self.metrics_config.update(metrics_config)
            
        # Set up checkpointing
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Set up W&B
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        if self.use_wandb:
            # Initialize wandb if it hasn't been initialized
            if not wandb.run:
                wandb.init(project=wandb_project, entity=wandb_entity, config=wandb_config)
            # Log the model as an artifact
            wandb.watch(model, log='all', log_freq=100)
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
        }
        
        # Storage for metrics
        self.sample_activations = {}
        self.rank_metrics = {}
        self.component_metrics = {}
        self.gradients = {}
    
    def train_epoch(self):
        """
        Train the model for one epoch.
        
        Returns:
            tuple: (epoch loss, epoch accuracy)
        """
        self.model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        # Track gradients if configured
        track_gradients = self.metrics_config['track_gradient_stats']
        record_gradients = self.metrics_config['record_gradients_for_layers']
        
        record_activations = (
            self.metrics_config['record_activations'] and 
            (self.current_epoch % self.metrics_config['record_activations_freq'] == 0)
        )
        
        # Train loop with progress bar
        loop = tqdm(enumerate(self.train_loader), total=len(self.train_loader), leave=False)
        for batch_idx, (inputs, targets) in loop:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Zero the parameter gradients
            self.optimizer.zero_grad()
            
            # Forward pass 
            store_activations = record_activations and batch_idx == 0
            
            # Check if model has store_activations parameter
            if hasattr(self.model, 'record_activations'):
                # Temporarily enable activation recording if needed
                old_record_setting = self.model.record_activations
                self.model.record_activations = store_activations
                outputs = self.model(inputs, store_activations=store_activations)
                # Check if outputs is a tuple (contains both predictions and activations)
                if isinstance(outputs, tuple):
                    outputs, activations = outputs
                    if store_activations:
                        self.sample_activations[f'epoch_{self.current_epoch}'] = activations
                self.model.record_activations = old_record_setting
            else:
                # Model doesn't support activation recording
                if hasattr(self.model, 'forward') and 'store_activations' in self.model.forward.__code__.co_varnames:
                    # Model's forward accepts store_activations parameter
                    outputs = self.model(inputs, store_activations=store_activations)
                    if store_activations and isinstance(outputs, tuple):
                        outputs, activations = outputs
                        self.sample_activations[f'epoch_{self.current_epoch}'] = activations
                else:
                    # Regular forward pass
                    outputs = self.model(inputs)
            
            # Compute loss
            loss = self.criterion(outputs, targets)
            
            # Backward and optimize
            loss.backward()
            
            # Collect gradient statistics if configured
            if track_gradients:
                self._collect_gradient_stats(batch_idx == 0)
            
            self.optimizer.step()
            
            # Update statistics
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == targets).item()
            total_samples += inputs.size(0)
            
            # Update progress bar
            loop.set_description(f"Epoch {self.current_epoch+1}")
            loop.set_postfix(loss=loss.item(), acc=running_corrects/total_samples)
        
        # Update learning rate
        if self.scheduler is not None:
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                pass  # This scheduler is stepped in validate()
            else:
                self.scheduler.step()
        
        # Compute epoch statistics
        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects / total_samples
        
        # Compute additional metrics if configured
        if self.metrics_config['compute_rank_metrics']:
            self.rank_metrics[f'epoch_{self.current_epoch}'] = compute_rank_metrics(self.model)
        
        if self.metrics_config['compute_component_metrics']:
            self.component_metrics[f'epoch_{self.current_epoch}'] = compute_connected_components(self.model)
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        """
        Validate the model.
        
        Returns:
            tuple: (validation loss, validation accuracy)
        """
        if self.val_loader is None:
            return None, None
        
        self.model.eval()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                if isinstance(outputs, tuple):
                    outputs, _ = outputs
                loss = self.criterion(outputs, targets)
                
                # Update statistics
                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == targets).item()
                total_samples += inputs.size(0)
        
        # Compute validation statistics
        val_loss = running_loss / total_samples
        val_acc = running_corrects / total_samples
        
        # Step LR scheduler if it's ReduceLROnPlateau
        if self.scheduler is not None and isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(val_loss)
        
        return val_loss, val_acc
    
    def train(self, num_epochs=100, early_stopping=None):
        """
        Train the model for multiple epochs.
        
        Parameters:
            num_epochs (int): Number of epochs to train
            early_stopping (int, optional): Early stopping patience in epochs
        
        Returns:
            dict: Training history
        """
        print(f"Training on device: {self.device}")
        best_model_state = None
        early_stopping_counter = 0
        
        # Training loop
        start_time = time.time()
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            
            if val_loss is not None:
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
                
                # Check for improvement
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_val_loss = val_loss
                    best_model_state = copy.deepcopy(self.model.state_dict())
                    early_stopping_counter = 0
                    
                    # Save best model checkpoint
                    self._save_checkpoint('best')
                else:
                    early_stopping_counter += 1
            
            # Save epoch checkpoint
            self._save_checkpoint(f'epoch_{epoch}')
            
            # Print progress
            val_loss_str = f"{val_loss:.4f}" if val_loss is not None else "N/A"
            val_acc_str = f"{val_acc:.4f}" if val_acc is not None else "N/A"
            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}, "
                  f"Val loss: {val_loss_str}, Val acc: {val_acc_str}")
            
            # Log to W&B
            if self.use_wandb:
                log_dict = {
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                }
                if val_loss is not None:
                    log_dict.update({
                        'val_loss': val_loss,
                        'val_acc': val_acc
                    })
                
                # Log rank metrics if available
                if f'epoch_{epoch}' in self.rank_metrics:
                    for k, v in self.rank_metrics[f'epoch_{epoch}'].items():
                        log_dict[f'rank/{k}'] = v
                
                # Log component metrics if available
                if f'epoch_{epoch}' in self.component_metrics:
                    for k, v in self.component_metrics[f'epoch_{epoch}'].items():
                        log_dict[f'components/{k}'] = v
                
                wandb.log(log_dict)
            
            # Early stopping
            if early_stopping is not None and early_stopping_counter >= early_stopping:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Training completed
        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.2f} seconds")
        
        # Load best model if available
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"Loaded best model with validation accuracy: {self.best_val_acc:.4f}")
        
        return self.history
    
    def _collect_gradient_stats(self, save_gradients=False):
        """
        Collect gradient statistics.
        
        Parameters:
            save_gradients (bool): Whether to save the full gradients
        """
        gradient_stats = {}
        layers_to_record = self.metrics_config['record_gradients_for_layers']
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                # Skip if we're only recording specific layers and this isn't one
                if layers_to_record is not None and not any(layer in name for layer in layers_to_record):
                    continue
                
                # Compute statistics
                grad_data = param.grad.data
                grad_mean = grad_data.mean().item()
                grad_std = grad_data.std().item()
                grad_abs_mean = grad_data.abs().mean().item()
                grad_norm = grad_data.norm().item()
                
                # Store statistics
                gradient_stats[name] = {
                    'mean': grad_mean,
                    'std': grad_std,
                    'abs_mean': grad_abs_mean,
                    'norm': grad_norm
                }
                
                # Save full gradients if requested
                if save_gradients:
                    epoch_key = f'epoch_{self.current_epoch}'
                    if epoch_key not in self.gradients:
                        self.gradients[epoch_key] = {}
                    self.gradients[epoch_key][name] = grad_data.cpu().clone()
        
        return gradient_stats
    
    def _save_checkpoint(self, name):
        """
        Save a checkpoint of the model.
        
        Parameters:
            name (str): Name of the checkpoint
        """
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(self.checkpoint_dir, f'{name}.pt')
        
        torch.save({
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'history': self.history
        }, checkpoint_path)
    
    def load_checkpoint(self, path):
        """
        Load a checkpoint.
        
        Parameters:
            path (str): Path to the checkpoint
            
        Returns:
            dict: Checkpoint data
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_acc = checkpoint['best_val_acc']
        self.history = checkpoint['history']
        
        return checkpoint
    
    def evaluate(self, test_loader=None):
        """
        Evaluate the model on a test set.
        
        Parameters:
            test_loader (DataLoader, optional): Test data loader (defaults to validation loader)
            
        Returns:
            tuple: (test loss, test accuracy)
        """
        if test_loader is None:
            test_loader = self.val_loader
            
        if test_loader is None:
            raise ValueError("No test_loader provided and no val_loader set during initialization")
        
        self.model.eval()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                if isinstance(outputs, tuple):
                    outputs, _ = outputs
                loss = self.criterion(outputs, targets)
                
                # Update statistics
                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == targets).item()
                total_samples += inputs.size(0)
        
        # Compute test statistics
        test_loss = running_loss / total_samples
        test_acc = running_corrects / total_samples
        
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
        
        return test_loss, test_acc


class ContinualTrainer:
    """
    Trainer for continual learning experiments.
    """
    def __init__(self, 
                 model, 
                 continual_data_loader,
                 criterion=None,
                 optimizer_factory=None,
                 scheduler_factory=None,
                 device=None,
                 metrics_config=None,
                 checkpoint_dir='./checkpoints',
                 use_wandb=False,
                 wandb_project='continual_learning',
                 wandb_entity=None,
                 wandb_config=None):
        """
        Initialize the continual trainer.
        
        Parameters:
            model (nn.Module): Model to train
            continual_data_loader (ContinualDataLoader): Continual learning dataloader
            criterion (callable, optional): Loss function (defaults to CrossEntropyLoss)
            optimizer_factory (callable, optional): Function that takes model and returns optimizer
            scheduler_factory (callable, optional): Function that takes optimizer and returns scheduler
            device (torch.device, optional): Device to use
            metrics_config (dict, optional): Configuration for additional metrics
            checkpoint_dir (str): Directory to save checkpoints
            use_wandb (bool): Whether to log with Weights & Biases
            wandb_project (str): W&B project name
            wandb_entity (str, optional): W&B entity name
            wandb_config (dict, optional): Additional W&B configuration
        """
        self.model = model
        self.continual_data_loader = continual_data_loader
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
        
        # Set up optimizer factory
        if optimizer_factory is None:
            def default_optimizer_factory(model):
                return optim.Adam(model.parameters(), lr=0.001)
            self.optimizer_factory = default_optimizer_factory
        else:
            self.optimizer_factory = optimizer_factory
        
        # Set up scheduler factory
        self.scheduler_factory = scheduler_factory
        
        # Set up device
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Set up metrics configuration
        default_metrics_config = {
            'compute_rank_metrics': False,
            'compute_component_metrics': False,
            'compute_activation_stats': False,
            'compute_weight_stats': False,
            'track_gradient_stats': False,
            'record_activations': False,
            'record_activations_freq': 10,  # epochs between recording
            'record_sample_size': 100,  # number of samples for recording
            'compute_forgetting': True,  # track performance on previous tasks
            'record_activations_for_layers': None,  # record all layers if None
            'record_gradients_for_layers': None,  # record all layers if None
        }
        self.metrics_config = default_metrics_config.copy()
        if metrics_config:
            self.metrics_config.update(metrics_config)
            
        # Set up checkpointing
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Set up W&B
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        if self.use_wandb:
            # Initialize wandb if it hasn't been initialized
            if not wandb.run:
                wandb.init(project=wandb_project, entity=wandb_entity, config=wandb_config)
            # Log the model as an artifact
            wandb.watch(model, log='all', log_freq=100)
        
        # Training state
        self.current_task = 0
        self.current_epoch = 0
        self.history = {
            'tasks': []
        }
        
        # Task performance tracking
        self.task_accuracies = {}
    
    def train_task(self, task_idx, num_epochs=10, early_stopping=None):
        """
        Train the model on a specific task.
        
        Parameters:
            task_idx (int): Index of the task to train on
            num_epochs (int): Number of epochs to train
            early_stopping (int, optional): Early stopping patience in epochs
            
        Returns:
            dict: Task training history
        """
        print(f"Training on Task {task_idx}")
        self.current_task = task_idx
        
        # Get data loaders for the current task
        train_loader = self.continual_data_loader.get_task_loader(task_idx, train=True)
        val_loader = self.continual_data_loader.get_task_loader(task_idx, train=False)
        
        # Create optimizer and scheduler
        optimizer = self.optimizer_factory(self.model)
        scheduler = self.scheduler_factory(optimizer) if self.scheduler_factory is not None else None
        
        # Create trainer for this task
        task_trainer = Trainer(
            model=self.model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=self.criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=self.device,
            metrics_config=self.metrics_config,
            checkpoint_dir=os.path.join(self.checkpoint_dir, f'task_{task_idx}'),
            use_wandb=self.use_wandb
        )
        
        # Train on the task
        task_history = task_trainer.train(num_epochs=num_epochs, early_stopping=early_stopping)
        
        # Evaluate on all previous tasks to measure forgetting
        self.evaluate_all_tasks(up_to_task=task_idx+1)
        
        # Save task history
        self.history['tasks'].append({
            'task_idx': task_idx,
            'task_history': task_history,
            'task_accuracies': self.task_accuracies.copy()
        })
        
        # Save task checkpoint
        self._save_checkpoint(f'after_task_{task_idx}')
        
        return task_history
    
    def train_all_tasks(self, num_epochs_per_task=10, early_stopping=None):
        """
        Train the model on all tasks sequentially.
        
        Parameters:
            num_epochs_per_task (int): Number of epochs to train per task
            early_stopping (int, optional): Early stopping patience in epochs
            
        Returns:
            dict: Training history
        """
        num_tasks = len(self.continual_data_loader)
        
        for task_idx in range(num_tasks):
            self.train_task(task_idx, num_epochs=num_epochs_per_task, early_stopping=early_stopping)
        
        return self.history
    
    def evaluate_all_tasks(self, up_to_task=None):
        """
        Evaluate the model on all tasks seen so far.
        
        Parameters:
            up_to_task (int, optional): Evaluate only up to this task index
            
        Returns:
            dict: Task accuracies
        """
        num_tasks = up_to_task if up_to_task is not None else len(self.continual_data_loader)
        
        self.model.eval()
        accuracies = {}
        
        for task_idx in range(num_tasks):
            test_loader = self.continual_data_loader.get_task_loader(task_idx, train=False)
            acc = compute_accuracy(self.model, test_loader, self.device)
            
            task_key = f'task_{task_idx}'
            accuracies[task_key] = acc
            
            print(f"Task {task_idx} Accuracy: {acc:.4f}")
            
            # Log to W&B
            if self.use_wandb:
                wandb.log({
                    f'task_accuracy/{task_key}': acc,
                    'current_task': self.current_task
                })
        
        # Compute average accuracy across all tasks
        avg_acc = sum(accuracies.values()) / len(accuracies)
        accuracies['average'] = avg_acc
        print(f"Average Accuracy: {avg_acc:.4f}")
        
        # Log to W&B
        if self.use_wandb:
            wandb.log({
                'average_accuracy': avg_acc,
                'current_task': self.current_task
            })
        
        # Update task accuracies
        self.task_accuracies.update(accuracies)
        
        return accuracies
    
    def compute_forgetting(self):
        """
        Compute forgetting for each task.
        
        Returns:
            dict: Forgetting metrics for each task
        """
        if not self.metrics_config['compute_forgetting']:
            return {}
        
        forgetting = {}
        
        for task_idx in range(self.current_task):
            task_key = f'task_{task_idx}'
            
            # Find best accuracy achieved on this task before switching to next task
            best_acc = 0
            for task_history in self.history['tasks']:
                if task_history['task_idx'] == task_idx:
                    # Best accuracy achieved during training on this task
                    if 'val_acc' in task_history['task_history']:
                        best_acc = max(task_history['task_history']['val_acc'])
                    break
            
            # Current accuracy on this task
            current_acc = self.task_accuracies.get(task_key, 0)
            
            # Compute forgetting
            forgetting[task_key] = max(0, best_acc - current_acc)
            
            print(f"Forgetting for Task {task_idx}: {forgetting[task_key]:.4f}")
        
        # Compute average forgetting
        if forgetting:
            avg_forgetting = sum(forgetting.values()) / len(forgetting)
            forgetting['average'] = avg_forgetting
            print(f"Average Forgetting: {avg_forgetting:.4f}")
            
            # Log to W&B
            if self.use_wandb:
                wandb.log({
                    'average_forgetting': avg_forgetting,
                    'current_task': self.current_task
                })
        
        return forgetting
    
    def _save_checkpoint(self, name):
        """
        Save a checkpoint of the model.
        
        Parameters:
            name (str): Name of the checkpoint
        """
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(self.checkpoint_dir, f'{name}.pt')
        
        torch.save({
            'current_task': self.current_task,
            'model_state_dict': self.model.state_dict(),
            'task_accuracies': self.task_accuracies,
            'history': self.history
        }, checkpoint_path)

    
    def load_checkpoint(self, path):
        """
        Load a checkpoint.
        
        Parameters:
            path (str): Path to the checkpoint
            
        Returns:
            dict: Checkpoint data
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.current_task = checkpoint['current_task']
        self.task_accuracies = checkpoint['task_accuracies']
        self.history = checkpoint['history']
        
        return checkpoint

    def __len__(self):
        """Number of tasks in the sequence."""
        return len(self.train_loaders)
    
    def get_all_tasks_loader(self, train=True):
        """
        Get a single dataloader containing data from all tasks.
        
        Parameters:
            train (bool): Whether to get the training dataloader
            
        Returns:
            DataLoader: Dataloader with all tasks
        """
        # Combine all task datasets
        all_tasks = ConcatDataset(
            self.task_sequence.train_tasks if train else self.task_sequence.test_tasks
        )
        
        # Create a dataloader
        return DataLoader(
            all_tasks, 
            batch_size=self.batch_size, 
            shuffle=self.shuffle if train else False, 
            num_workers=self.num_workers
        )