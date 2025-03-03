"""
Continual learning experiment focusing on rank analysis.
This script demonstrates how to track and analyze the rank of layer representations during training.
"""

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from utils.config import ExperimentConfig, create_model_from_config
from data.datasets import ContinualTaskSequence, ContinualDataLoader
from training.trainer import Trainer
from training.metrics import compute_rank_metrics, compute_effective_rank, get_layer_activations
from analysis.rank_analysis import compute_layer_ranks, compute_feature_diversity, analyze_rank_dynamics
import utils.visualization as viz

class RankAnalysisTrainer(Trainer):
    """Extended trainer to track rank metrics during training."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rank_history = {
            'epochs': [],
            'layer_ranks': {},
            'activation_ranks': {}
        }
        self.model_checkpoints = []
    
    def train_epoch(self):
        """Override train_epoch to track rank metrics."""
        epoch_loss, epoch_acc = super().train_epoch()
        
        # Compute rank metrics for model weights
        layer_ranks = compute_layer_ranks(self.model)
        
        # Store epoch information
        self.rank_history['epochs'].append(self.current_epoch)
        
        # Initialize layer entries if needed
        for layer_name in layer_ranks:
            if layer_name not in self.rank_history['layer_ranks']:
                self.rank_history['layer_ranks'][layer_name] = {
                    'effective_rank': [],
                    'stable_rank': [],
                    'rank_95': []
                }
            
            # Store rank metrics
            self.rank_history['layer_ranks'][layer_name]['effective_rank'].append(
                layer_ranks[layer_name]['effective_rank'])
            self.rank_history['layer_ranks'][layer_name]['stable_rank'].append(
                layer_ranks[layer_name]['stable_rank'])
            self.rank_history['layer_ranks'][layer_name]['rank_95'].append(
                layer_ranks[layer_name]['rank_95'])
        
        # Save model checkpoint
        if self.current_epoch % 5 == 0:  # Save every 5 epochs
            self.model_checkpoints.append((self.current_epoch, 
                                          {k: v.cpu() for k, v in self.model.state_dict().items()}))
        
        return epoch_loss, epoch_acc
    
    def compute_activation_ranks(self, loader):
        """Compute rank metrics for activations."""
        # Get activations for sample batch
        activations = get_layer_activations(self.model, loader, device=self.device)
        
        # Compute effective rank for each layer's activations
        activation_ranks = {}
        for layer_name, activation in activations.items():
            if isinstance(activation, torch.Tensor) and activation.dim() > 1:
                # Reshape to 2D if needed
                if activation.dim() > 2:
                    act_flat = activation.reshape(activation.shape[0], -1)
                else:
                    act_flat = activation
                
                # Compute effective rank
                eff_rank = compute_effective_rank(act_flat)
                activation_ranks[layer_name] = eff_rank
                
                # Store in history
                if layer_name not in self.rank_history['activation_ranks']:
                    self.rank_history['activation_ranks'][layer_name] = []
                
                self.rank_history['activation_ranks'][layer_name].append(eff_rank)
        
        return activation_ranks


def main():
    # Generate timestamp for experiment
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create configuration for MLP experiment
    config = ExperimentConfig(
        experiment_name=f"rank_analysis_{timestamp}",
        random_seed=42,
        dataset_name="cifar10",
        task_type="pairs",
        num_tasks=5,
        model_type="mlp",
        model_config={
            "hidden_sizes": [512, 256, 128],
            "activation": "relu",
            "dropout_p": 0.0,  # No dropout to better analyze rank
            "normalization": None  # No normalization for clearer rank analysis
        },
        optimizer_type="adam",
        optimizer_config={"lr": 0.001, "weight_decay": 0.0},
        batch_size=128,
        num_epochs_per_task=30,
        metrics_config={
            "compute_rank_metrics": True,
            "compute_component_metrics": True,
            "compute_activation_stats": True,
            "record_activations": True
        },
        results_dir=f"./results/rank_analysis_{timestamp}",
        checkpoint_dir=f"./checkpoints/rank_analysis_{timestamp}"
    )
    
    # Create directories
    os.makedirs(config.results_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    torch.manual_seed(config.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.random_seed)
    
    # Create task sequence
    task_sequence = ContinualTaskSequence(
        dataset_name=config.dataset_name,
        task_type=config.task_type,
        n_tasks=config.num_tasks,
        root=config.dataset_root
    )
    
    # We'll track rank for one task at a time
    task_idx = 0  # First task
    train_loader = task_sequence.get_task(task_idx, train=True)
    val_loader = task_sequence.get_task(task_idx, train=False)
    
    # Create dataloaders
    train_dataloader = torch.utils.data.DataLoader(
        train_loader, 
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers
    )
    
    val_dataloader = torch.utils.data.DataLoader(
        val_loader, 
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers
    )
    
    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() and config.use_cuda else "cpu")
    model = create_model_from_config(config)
    model.to(device)
    
    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.optimizer_config["lr"],
        weight_decay=config.optimizer_config.get("weight_decay", 0)
    )
    
    # Create specialized rank analysis trainer
    trainer = RankAnalysisTrainer(
        model=model,
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        criterion=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        device=device,
        metrics_config=config.metrics_config,
        checkpoint_dir=config.checkpoint_dir
    )
    
    # Train model
    print(f"Training model on Task {task_idx}...")
    trainer.train(num_epochs=config.num_epochs_per_task)
    
    # Compute final activation ranks
    print("Computing final activation ranks...")
    activation_ranks = trainer.compute_activation_ranks(val_dataloader)
    
    # Print rank metrics
    print("\nFinal layer weight rank metrics:")
    for layer_name, ranks in trainer.rank_history['layer_ranks'].items():
        final_eff_rank = ranks['effective_rank'][-1]
        print(f"{layer_name}: Effective Rank = {final_eff_rank:.2f}")
    
    print("\nFinal activation rank metrics:")
    for layer_name, final_rank in activation_ranks.items():
        print(f"{layer_name}: Effective Rank = {final_rank:.2f}")
    
    # Visualize rank evolution over training
    print("\nGenerating rank evolution visualizations...")
    
    # Plot weight rank evolution
    for layer_name, ranks in trainer.rank_history['layer_ranks'].items():
        plt.figure(figsize=(10, 6))
        plt.plot(trainer.rank_history['epochs'], ranks['effective_rank'], marker='o')
        plt.title(f'Effective Rank Evolution for {layer_name}')
        plt.xlabel('Epoch')
        plt.ylabel('Effective Rank')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(config.results_dir, f"{layer_name}_rank_evolution.png"))
    
    # Plot activation rank evolution
    for layer_name, ranks in trainer.rank_history['activation_ranks'].items():
        if len(ranks) > 1:  # Only plot if we have multiple points
            plt.figure(figsize=(10, 6))
            plt.plot(trainer.rank_history['epochs'][:len(ranks)], ranks, marker='o')
            plt.title(f'Activation Rank Evolution for {layer_name}')
            plt.xlabel('Epoch')
            plt.ylabel('Effective Rank')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(config.results_dir, f"{layer_name}_activation_rank_evolution.png"))
    
    # Load stored model checkpoints and analyze singular value spectrum
    print("\nAnalyzing singular value spectrum evolution...")
    
    # Convert checkpoint format to what analyze_rank_dynamics expects
    model_checkpoints = []
    for epoch, state_dict in trainer.model_checkpoints:
        model_copy = create_model_from_config(config)
        model_copy.load_state_dict(state_dict)
        model_checkpoints.append((epoch, model_copy))
    
    # Analyze rank dynamics
    for layer_name in trainer.rank_history['layer_ranks'].keys():
        # Extract key layers
        if 'fc' in layer_name or 'linear' in layer_name:
            # Plot singular value spectrum for a few checkpoints
            spectrum_epochs = [0, len(model_checkpoints)//2, -1]  # First, middle, last
            spectrum_epochs = [idx for idx in spectrum_epochs if idx < len(model_checkpoints) and idx >= 0]
            
            if len(spectrum_epochs) > 0:
                epoch_indices = [spectrum_epochs[i] for i in range(len(spectrum_epochs))]
                
                # Create model list with just the selected epochs
                selected_models = [(model_checkpoints[idx][0], model_checkpoints[idx][1]) 
                                  for idx in epoch_indices]
                
                # Analyze rank dynamics
                rank_data = analyze_rank_dynamics(selected_models, [layer_name])
                
                # Plot singular value spectrum
                viz.plot_singular_values_spectrum(
                    rank_data,
                    layer_name=layer_name,
                    figsize=(10, 6),
                    log_scale=True,
                    save_path=os.path.join(config.results_dir, f"{layer_name}_singular_value_spectrum.png")
                )
    
    print(f"\nExperiment complete! Results saved to {config.results_dir}")

if __name__ == "__main__":
    main()